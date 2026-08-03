#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
compare_models.py — сравнение двух моделей на РЕАЛЬНОМ видео. Бок о бок.

ЗАЧЕМ: метрики (mAP/recall) на твоём val'е НЕ МОГУТ ответить на главный вопрос —
val размечен старой 640-моделью, поэтому если новая найдёт дальнюю точку, которую
учитель не разметил, это засчитается как ЛОЖНОЕ срабатывание и ПОНИЗИТ метрику.
Модель наказывается ровно за то, ради чего её учили.

Единственный честный тест — глазами на реальном видео с дальней целью.
Этот скрипт кладёт два прогона рядом в один кадр и считает статистику.

ЗАПУСКАТЬ НА НОУТЕ (там ultralytics и веса .pt).

  # главный тест: старая 640 против новой 1280
  python compare_models.py --video test.mp4 ^
      --model-a best_640.pt --imgsz-a 640 ^
      --model-b best_1280_n.pt --imgsz-b 1280

  # с сохранением результата
  python compare_models.py --video test.mp4 --model-a ... --model-b ... --save out.mp4

  # одна модель (просто посмотреть)
  python compare_models.py --video test.mp4 --model-a best_1280_n.pt --imgsz-a 1280

  # FPV-камера из дрон-комплекта: она САМА шлёт RTP на твой IP -> нужен .sdp
  python compare_models.py --video camera.sdp --model-a best_1280_n.pt --imgsz-a 1280

  # IP-камера по RTSP (Ethernet)
  python compare_models.py --video "rtsp://USER:PASS@CAM_IP:554/Streaming/Channels/101" ^
      --model-a best_1280_n.pt --imgsz-a 1280

  # веб-камера
  python compare_models.py --video 0 --model-a best_1280_n.pt --imgsz-a 1280

Управление в окне:  ПРОБЕЛ — пауза/пуск, стрелки <- -> покадрово (на паузе), Q — выход.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

# ВАЖНО: OPENCV_FFMPEG_CAPTURE_OPTIONS читается при ИНИЦИАЛИЗАЦИИ FFmpeg-бэкенда,
# то есть ДО первого VideoCapture. Если выставить её позже — не подхватится, и
# будет ошибка "Protocol 'rtp' not on whitelist 'file,crypto,data'".
# Поэтому ставим здесь, до import cv2.
#   protocol_whitelist — разрешаем rtp/udp (нужно для SDP push-потока)
#   rtsp_transport;tcp — для RTSP-камер: без потерь пакетов (для rtp игнорируется)
#   fifo_size          — буфер под всплески UDP, меньше дропов
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "protocol_whitelist;file,crypto,data,rtp,udp,tcp,rtsp,http,https,tls|"
    "rtsp_transport;tcp|"
    "fifo_size;50000000|"      # большой буфер сокета: меньше потерь UDP -> меньше «сыпи»
    "max_delay;100000|"        # 0.1с, не копим задержку
    "reorder_queue_size;0|"
    "fflags;nobuffer|"
    "flags;low_delay"
)

from typing import Any

import cv2
import numpy as np



class LatestFrame(threading.Thread):
    """Непрерывно вычерпывает поток и держит ТОЛЬКО свежий кадр.

    Зачем: если читать кадры медленнее, чем камера их шлёт (а инференс на 1280
    небыстрый), кадры копятся в буфере -> растёт задержка, буфер переполняется ->
    UDP-пакеты теряются -> картинка «сыпется» квадратами.
    Этот поток всегда забирает последний кадр, старые выбрасывает: задержка не
    накапливается, сокет не переполняется.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.fid = 0
        self.running = True
        self.dropped = 0

    def run(self) -> None:
        while self.running:
            ok = self.cap.grab()          # быстро: только забрать, не декодировать
            if not ok:
                time.sleep(0.005)
                continue
            ok, f = self.cap.retrieve()
            if not ok:
                continue
            with self.lock:
                if self.frame is not None:
                    self.dropped += 1     # прошлый кадр не успели обработать
                self.frame = f
                self.fid += 1

    def get(self) -> tuple[np.ndarray | None, int]:
        with self.lock:
            if self.frame is None:
                return None, 0
            f, i = self.frame, self.fid
            self.frame = None             # помечаем как забранный
            return f, i

    def stop(self) -> None:
        self.running = False



def apply_crop(frame: np.ndarray, spec: str) -> np.ndarray:
    """Вырезать часть кадра. Для подвесов, где RGB и тепловизор в одном кадре."""
    if not spec:
        return frame
    h, w = frame.shape[:2]
    presets = {
        "left":   (0.0, 0.0, 0.5, 1.0),
        "right":  (0.5, 0.0, 1.0, 1.0),
        "top":    (0.0, 0.0, 1.0, 0.5),
        "bottom": (0.0, 0.5, 1.0, 1.0),
    }
    if spec in presets:
        x1, y1, x2, y2 = presets[spec]
    else:
        try:
            x1, y1, x2, y2 = [float(v) for v in spec.split(",")]
        except Exception:
            print(f"[!] Cannot parse --crop '{spec}'. Format: left|right|top|bottom "
                  f"or x1,y1,x2,y2 in fractions 0..1")
            return frame
    return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def draw_grid(frame: np.ndarray) -> np.ndarray:
    """Сетка с долями — чтобы понять, какую область резать."""
    h, w = frame.shape[:2]
    for f in (0.25, 0.5, 0.75):
        cv2.line(frame, (int(w * f), 0), (int(w * f), h), (0, 255, 255), 1)
        cv2.line(frame, (0, int(h * f)), (w, int(h * f)), (0, 255, 255), 1)
        cv2.putText(frame, f"{f}", (int(w * f) + 3, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"{f}", (3, int(h * f) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
    return frame


def draw_dets(frame: np.ndarray, res: Any, color: tuple[int, int, int],
              label: str) -> tuple[int, list[float]]:
    """Рисует рамки. Возвращает (кол-во целей, список conf)."""
    confs = []
    boxes = res.boxes
    n = 0 if boxes is None else len(boxes)
    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        c = float(boxes.conf[i])
        confs.append(c)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # для дальних «точек» рамка крошечная — рисуем ещё и метку-указатель
        cv2.putText(frame, f"{c:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if (x2 - x1) < 20 and (y2 - y1) < 20:
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 18, color, 1)
    return n, confs


def banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Стартовый баннер и панель конфига (rich, soft-import — без rich работает плейн-текстом)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.status import Status
    from rich.spinner import SPINNERS
    from rich.table import Table
    from rich.theme import Theme
    _HAS_RICH = True
    _console = Console(theme=Theme({"default": "#FFA500"}))
    SPINNERS["propeller"] = {"interval": 80, "frames": ["|", "/", "—", "\\"]}
except ImportError:
    _HAS_RICH = False
    _console = None


# Силуэт стража-воина с копьём (из 3.jpg → ASCII). Цвет наследует Theme (#FFA500).
_LOGO = (
    "                                  ⢀⣾⡂ ⣼⡆\n"
    "                                ⢀⣴⣿⡟ ⢰⣿⡇\n"
    "                             ⢀⣠⣴⣿⣿⠋⠁ ⠈⢻⡇\n"
    "                         ⣠⣴⣶⣾⡿⠟⠁     ⠘⢿⡇\n"
    "                      ⣀⣴⣿⣿⡿⠟⠋         ⢼⡇\n"
    "                   ⣀⣤⣾⣿⠿⠛⠉            ⠈⠇\n"
    "                 ⢀⣼⣿⡿⠋\n"
    "                ⢠⣿⣿⡿⠁\n"
    "               ⣰⣿⣿⣿⣄⣀⣀\n"
    "               ⣿⣿⣿⣿⣿⣿⣿⣤⣀⣀\n"
    "            ⢀⣄ ⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⡀           ⡆\n"
    "           ⣴⣿⣿⣿⣿⡿⠛⠉⠉⠁ ⠈⠁⠉⠉⠙⠻⢿⣷⡶        ⢠⠇\n"
    "          ⢰⣿⣿⣿⣿⠟  ⣀⣤⣤⣀⡀      ⠈⠉        ⡼\n"
    "         ⣠⣿⣿⣿⣿⣿⡇ ⠘⢿⣿⣿⣿⣿⣦               ⡇\n"
    "        ⢠⣿⣿⣿⣿⣿⡿    ⠉⠉⠉⠙⠛⢷⡄   ⣀⣤⣄       ⠁\n"
    "       ⢀⣾⣿⣿⣿⣿⣿⡇        ⢀⡾⠁  ⠺⠿⣿⣿⣿⡄\n"
    "      ⣴⣿⣿⣿⣿⣿⣿⣿⠁       ⣠⣿⠃     ⢿⣿⣿⡟\n"
    "    ⢀⡼⠻⠿⠁⠈⣿⣿⣿⣿⡄      ⢾⣿⡇       ⠈⠉ ⣠⣴     ⡄   ⢱⠄\n"
    "   ⣀⣿⡄    ⠛⣿⣿⣿⣇       ⠿⠿⠦        ⣴⣿⣿⣦⣀⣠⣤⣤⣿⣶⠈⠙⠛⠳⠶\n"
    "⢀⣴⣿⣿⣿⡇   ⡀ ⣿⣿⣿⣿⠄  ⠢⣴⣤⣀⡀        ⣠⣾⣿⣿⡿⠛⠿⣿⣿⣿⠉⠹⡄\n"
    "⣿⣿⣿⣿⣿⡇  ⢰⡇ ⠻⡿⠿⣿⣇    ⠉⠙⠛⠶⡀     ⣼⣿⣿⣿⣿⡇   ⠈⠋ ⢠⡿\n"
    "⢿⣿⣿⣿⡿⠁  ⣼⣧  ⠁ ⠸⣿⣆          ⣠⣴⣾⣿⣿⣿⡿⠛⠁\n"
    "⠈⠉⢿⡿    ⣹⠿     ⠙⢿⣷⣀    ⢀⣠⣴⣾⣿⣿⣿⣿⣿⠿⠁\n"
    "  ⢸⡇              ⠸⣿⣶⣶⡿⠛⠉ ⣈⠿⢿⡿⠉⠉\n"
    "  ⠈⡇               ⠙⠋⠁   ⠐⠋  ⠁\n"
    "\n"
    "  [bold]GRIDEN · DRONE · COMPARE[/] — two eyes on the range"
)


def print_banner() -> None:
    """Стартовый блоб. Печатает сразу при запуске, пока грузятся модели."""
    if _HAS_RICH:
        try:
            _console.print(_LOGO, style="#FFA500")
            return
        except Exception:
            pass            # rich не смог (консоль без reconfigure) -- ниже ASCII-заглушка
    print("DRONE . COMPARE  --  two eyes on the range")


def _spinner(msg: str) -> Any:
    """Спиннер rich (оранжевый 'пропеллер') на время долгой операции; без rich -- plain print."""
    if _HAS_RICH:
        return Status(f"[bold #FFA500]{msg}[/]", spinner="propeller")
    print(msg)
    from contextlib import nullcontext
    return nullcontext()


def print_config(a: argparse.Namespace, src_label: str,
                 w: int, h: int, live: bool) -> None:
    """Сводка запуска одним блоком: сразу видно чем запустил."""
    rows: list[tuple[str, str]] = [
        ("A", f"{Path(a.model_a).name}  @ imgsz {a.imgsz_a}"),
    ]
    if a.model_b:
        rows.append(("B", f"{Path(a.model_b).name}  @ imgsz {a.imgsz_b}"))
    rows.append(("source", src_label))
    if w and h:
        rows.append(("frame", f"{w}x{h}"))
    rows.append(("device", str(a.device)))
    rows.append(("conf / iou", f"{a.conf} / {a.iou}"))
    if a.crop:
        rows.append(("crop", a.crop))
    if live:
        rows.append(("mode", "live — freshest frame only"))
    if a.save:
        rows.append(("save", a.save))

    if _HAS_RICH:
        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="bold cyan", no_wrap=True)
        tbl.add_column()
        for k, v in rows:
            tbl.add_row(k, v)
        _console.print(Panel(
            tbl, title="[bold]run config[/]",
            border_style="cyan",
            subtitle="[dim]SPACE pause · Q exit[/]",
            expand=False,
        ))
    else:
        print("-" * 54)
        for k, v in rows:
            print(f"  {k:<10} {v}")
        print("-" * 54)


# ===================== СТРАЖ-НА-OSD =====================
# Силуэт «стража на стенах крепости» — знак системы. PNG с альфой считается
# один раз и кэшируется; накладывается в левый нижний угол кадра (in-place).
_STRAZH_PNG = r"C:\Users\yaraz\strazh.png"
_strazh_cache: dict = {}

# Янтарный цвет силуэта в PNG (BGR) и общая «проявка» для силуэта и подписи —
# чтобы знак и надпись были одного цвета и одной яркости.
_STR_BGR: tuple[int, int, int] = (63, 132, 235)
_STR_ALPHA: float = 0.85          # было 0.7 — подняли, страж стал ярче


def draw_strazh(frame: np.ndarray, height_frac: float = 0.24, alpha: float = _STR_ALPHA,
                margin: int = 24) -> None:
    ov = _strazh_cache.get("img")
    if ov is None:
        loaded = cv2.imread(_STRAZH_PNG, cv2.IMREAD_UNCHANGED)
        if loaded is None:
            return                  # ассета нет — тихо пропускаем, вывод не ломаем
        ov = loaded
        _strazh_cache["img"] = ov
    fh = frame.shape[0]
    th = max(1, int(fh * height_frac))
    tw = max(1, int(ov.shape[1] * th / ov.shape[0]))
    layer = cv2.resize(ov, (tw, th), interpolation=cv2.INTER_AREA)
    x0, y0 = margin, fh - th - margin
    roi = frame[y0:y0 + th, x0:x0 + tw]
    # тёмная обводка по контуру -- страж читается на светлом небе (толщина по высоте кадра)
    mask_s = (layer[:, :, 3] > 127).astype(np.uint8)
    ring_px = max(1, round(fh / 500))            # 1620 -> ~3px, 540 -> 1px
    ring = cv2.dilate(mask_s, np.ones((3, 3), np.uint8), iterations=ring_px) - mask_s
    ring_m = ring.astype(np.float32) * 0.5
    for c in range(3):
        roi[:, :, c] = np.clip(roi[:, :, c] * (1 - ring_m), 0, 255).astype(np.uint8)
    m = (layer[:, :, 3] > 127).astype(np.float32) * alpha
    src = layer[:, :, :3].astype(np.float32)
    for c in range(3):
        roi[:, :, c] = np.clip(roi[:, :, c] * (1 - m) + src[:, :, c] * m,
                               0, 255).astype(np.uint8)

    # Подпись над силуэтом — тот же янтарь и та же alpha, что у стража:
    # текст рисуем на чёрной плашке, берём его как маску и блендим с кадром,
    # иначе непрозрачный putText всегда ярче полупрозрачного силуэта.
    text = "GRIDEN // drone-watch"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = fh / 1000.0                # масштаб по высоте кадра: на нативе 1620 -- крупно
    thick = 2 if scale >= 1.0 else 1
    pad = max(4, thick * 3)
    (w, h), bl = cv2.getTextSize(text, font, scale, thick)
    tl_w, tl_h = w + 2 * pad, h + bl + 2 * pad
    tl = np.zeros((tl_h, tl_w, 3), dtype=np.uint8)
    cv2.putText(tl, text, (pad, pad + h), font, scale, _STR_BGR, thick, cv2.LINE_AA)
    tm = (tl.astype(np.float32).sum(axis=2) > 0).astype(np.float32) * _STR_ALPHA
    gap = max(12, int(fh * 0.018))     # отступ подписи от верха силуэта
    ty0 = y0 - gap - tl_h              # низ плашки на gap px выше силуэта
    croi = frame[ty0:ty0 + tl_h, x0:x0 + tl_w]
    if ty0 >= 0 and ty0 + tl_h <= fh and croi.shape[0] == tl_h and croi.shape[1] == tl_w:
        col = np.array(_STR_BGR, dtype=np.float32)
        for c in range(3):
            croi[:, :, c] = np.clip(croi[:, :, c] * (1 - tm) + col[c] * tm,
                                    0, 255).astype(np.uint8)


# ===================== TRAIL (затухающий след детекций) =====================
# След из прошлых точек, чтобы глаз не терял дальнюю цель между кадрами.
# Цвет trail — ТЕМНЕЕ цвета модели, чтобы не путать с живой рамкой:
#   A=(0,255,0) зелёная  ->  TRAIL_BGR_A=(0,90,0)   тёмно-зелёный
#   B=(0,200,255) оранж. ->  TRAIL_BGR_B=(0,90,0) тёмно-оранжевый
TRAIL_BGR_A: tuple[int, int, int] = (0, 90, 0)
TRAIL_BGR_B: tuple[int, int, int] = (0, 90, 0)
TRAIL_TTL: float = 2.0                              # сек: сколько жить точке trail
TRAIL_R: int = 6                                    # радиус точки trail (меньше бокса)


def _draw_trail(panel: np.ndarray, trail: list[tuple[int, int, float]],
                color: tuple[int, int, int], now: float,
                ttl: float = TRAIL_TTL, r: int = TRAIL_R) -> None:
    """Рисует затухающий след прошлых детекций на панели (in-place, alpha-бленд кружков).

    Рисует только — НЕ фильтрует и НЕ аппендит. Вызывающий код сам отбрасывает
    истекшие точки до вызова и докладывает центры текущих детекций после.
    """
    H, W = panel.shape[:2]
    for tx, ty, tt in trail:
        _a = (1.0 - (now - tt) / ttl) * 0.6
        if _a <= 0.02:
            continue
        _cx0, _cy0 = max(0, tx - r), max(0, ty - r)
        _cx1, _cy1 = min(W, tx + r), min(H, ty + r)
        _croi = panel[_cy0:_cy1, _cx0:_cx1]
        _sub = np.zeros_like(_croi)
        cv2.circle(_sub, (tx - _cx0, ty - _cy0), r, color, -1)
        _m = (_sub.astype(np.float32).sum(axis=2) > 0).astype(np.float32) * _a
        for _c in range(3):
            _croi[:, :, _c] = np.clip(_croi[:, :, _c] * (1 - _m) + color[_c] * _m,
                                      0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare models on real video")
    ap.add_argument("--video", required=True, help="path to video (or 0 = webcam)")
    ap.add_argument("--model-a", required=True, help="model A (usually old 640)")
    ap.add_argument("--imgsz-a", type=int, default=640)
    ap.add_argument("--model-b", default="", help="model B (usually new 1280); empty = A only")
    ap.add_argument("--imgsz-b", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="low threshold = weak far targets visible (for range test)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="0", help="0 = GPU, cpu = CPU")
    ap.add_argument("--save", default="", help="save result to .mp4 file")
    ap.add_argument("--width", type=int, default=960, help="width of one panel on screen")
    ap.add_argument("--start", type=int, default=0, help="start from frame N")
    ap.add_argument("--crop", default="",
                    help="crop frame BEFORE inference. For dual-sensor gimbals "
                         "(RGB+thermal in one frame): left|right|top|bottom, "
                         "or explicit 'x1,y1,x2,y2' in fractions 0..1 (e.g. 0,0,0.5,1 = left half)")
    ap.add_argument("--show-source", action="store_true",
                    help="show source frame with grid — see where everything sits")
    a = ap.parse_args()
    print_banner()

    from ultralytics import YOLO

    print(f"[A] {a.model_a}  @ imgsz={a.imgsz_a}")
    with _spinner(f"loading model A: {a.model_a}"):
        model_a = YOLO(a.model_a)
    model_b = None
    if a.model_b:
        print(f"[B] {a.model_b}  @ imgsz={a.imgsz_b}")
        with _spinner(f"loading model B: {a.model_b}"):
            model_b = YOLO(a.model_b)

    src = 0 if a.video == "0" else a.video
    is_rtsp = isinstance(src, str) and src.lower().startswith(("rtsp://", "http://"))

    is_sdp = isinstance(src, str) and src.lower().endswith(".sdp")

    if is_sdp:
        # RTP-push: камера САМА вещает поток на наш IP:порт (не мы к ней подключаемся).
        # Так устроены FPV-камеры из дрон-комплектов. SDP описывает, что и откуда слушать.
        # ВАЖНО: IP ноутбука ДОЛЖЕН совпадать с тем, что указан в SDP (c=IN IP4 ...),
        # иначе камера вещает «в пустоту» и кадров не будет.
        print(f"[SDP] Listening RTP stream per description: {src}")
        print("      No frames? -> laptop IP = address in SDP, and open UDP port:")
        print('      netsh advfirewall firewall add rule name="RTP" dir=in '
              'action=allow protocol=UDP localport=8000')
        with _spinner(f"opening SDP/RTP stream: {src}"):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    elif is_rtsp:
        # RTSP поверх TCP: по UDP теряются пакеты -> кадры «рассыпаются» артефактами,
        # а для дальней «точки» любой артефакт = потерянная цель.
        print("[RTSP] Transport: TCP (no packet loss)")
        with _spinner(f"opening RTSP stream: {src}"):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
    else:
        with _spinner(f"opening source: {src}"):
            cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {a.video}")
        if is_sdp:
            print("        Check: 1) laptop IP = address from SDP (c=IN IP4 ...)")
            print("        2) camera is on and streaming  3) Windows firewall is not blocking UDP")
            print("        4) does this same .sdp open in VLC")
        if is_rtsp:
            print("        Check: 1) does the camera ping  2) is the RTSP-URL correct")
            print("        3) login/password in URL  4) does the link open in VLC")
        return

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w0 and h0:
        print(f"[SOURCE] {w0}x{h0}")
    else:
        # для SDP/RTP размер неизвестен до первого декодированного кадра — это норма
        print("[SOURCE] size becomes known after the first frame...")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if a.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)

    # для СЕТЕВЫХ источников (sdp/rtsp/веб-камера) — читаем в отдельном потоке,
    # всегда берём свежий кадр. Для файла это не нужно (там кадры ждут нас сами).
    live = is_sdp or is_rtsp or src == 0
    if src == 0:
        src_label = "webcam (0)"
    elif is_rtsp:
        src_label = f"RTSP·TCP  {a.video}"
    elif is_sdp:
        src_label = f"RTP/SDP push  {a.video}"
    else:
        src_label = f"file  {a.video}"
    puller = None
    if live:
        puller = LatestFrame(cap)
        puller.start()
        print("[STREAM] Live mode: taking only the freshest frame, dropping old ones")
        print("        (otherwise latency grows and artifacts appear)")
        time.sleep(0.5)

    print_config(a, src_label=src_label, w=w0, h=h0, live=live)

    writer = None
    stats = {"a": {"frames_hit": 0, "dets": 0, "confs": []},
             "b": {"frames_hit": 0, "dets": 0, "confs": []}}
    frames = 0
    paused = False
    warned = False
    t0 = time.time()

    # запись по R/K (как в field_test): одно side-by-side видео работы детектора
    recording = False
    rec_writer = None
    rec_path = ""
    RUN_KEYS = {ord("r"), ord("R"), 0x43A, 0x41A, 0xEA, 0xCA, 234, 202, 1082, 1050}

    # trail (затухающий след) — отдельный для каждой панели: A и B
    _trail_a: list[tuple[int, int, float]] = []
    _trail_b: list[tuple[int, int, float]] = []

    # FPS + latency инференса на каждой панели (свой замер для A и B)
    _lat_a = 0.0
    _lat_b = 0.0
    _fps_val_a = 0.0
    _fps_val_b = 0.0
    _fps_t0a = 0.0
    _fps_t0b = 0.0
    _fps_na = 0
    _fps_nb = 0

    print("\nSPACE — pause | R (K) — recording | Q — exit\n")

    while True:
        if not paused:
            if live:
                frame, _fid = puller.get()
                if frame is None:
                    if frames == 0 and (time.time() - t0) > 10 and not warned:
                        warned = True
                        print("\n[!] 10 seconds — not a single frame. Stream is not reaching us.")
                        print("    1) Is laptop IP = the address from SDP (c=IN IP4 line)?")
                        print("    2) Is UDP port 8000 open in the firewall?")
                        print("    3) Wireshark: are UDP packets coming from the camera to port 8000?")
                        print("    4) Does this same cam.sdp open in VLC?")
                    if (cv2.waitKeyEx(1) & 0xFFFFFF) in (ord('q'), ord('Q'), 0x439, 0x419, 0xE9, 0xC9, 1081, 1049, 233, 201):
                        break
                    time.sleep(0.005)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            frames += 1

            if a.show_source:
                # просто показать, что приходит, с сеткой долей
                v = draw_grid(frame.copy())
                sc = a.width / v.shape[1]
                cv2.imshow("SOURCE FRAME (grid in fractions). Q - exit",
                           cv2.resize(v, (a.width, int(v.shape[0] * sc))))
                if (cv2.waitKeyEx(1) & 0xFFFFFF) in (ord('q'), ord('Q'), 0x439, 0x419, 0xE9, 0xC9, 1081, 1049, 233, 201):
                    break
                frames += 1
                continue

            if frames == 1:
                fh, fw = frame.shape[:2]
                print(f"[SOURCE] real frame: {fw}x{fh}")
                if min(fw, fh) < a.imgsz_a:
                    print(f"[!] Frame {fw}x{fh} is SMALLER than network input {a.imgsz_a} on the short side.")
                    print("    The image gets upscaled — a far target will NOT become more detailed.")
                    print("    Raise the stream resolution in the camera settings,")
                    print(f"    otherwise {a.imgsz_a} is pointless (effectively running at {min(fw,fh)}).")

            frame = apply_crop(frame, a.crop)

            # --- модель A ---
            _t_inf = time.time()
            ra = model_a.predict(frame, imgsz=a.imgsz_a, conf=a.conf, iou=a.iou,
                                 device=a.device, verbose=False)[0]
            _lat_a = (time.time() - _t_inf) * 1000.0
            _fps_na += 1
            if not _fps_t0a:
                _fps_t0a = time.time()
            elif time.time() - _fps_t0a >= 0.5:
                _fps_val_a = _fps_na / (time.time() - _fps_t0a)
                _fps_t0a = time.time()
                _fps_na = 0

            _now = time.time()
            fa = frame.copy()
            # trail A: фильтр истекших → отрисовка ДО боксов → аппенд центров после draw_dets
            _trail_a[:] = [(tx, ty, tt) for tx, ty, tt in _trail_a if _now - tt < TRAIL_TTL]
            _draw_trail(fa, _trail_a, TRAIL_BGR_A, _now)
            na, ca = draw_dets(fa, ra, (0, 255, 0), "A")
            if ra.boxes is not None and len(ra.boxes) > 0:
                for b in ra.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = (int(v) for v in b)
                    _trail_a.append(((x1 + x2) // 2, (y1 + y2) // 2, _now))
            stats["a"]["dets"] += na
            stats["a"]["confs"] += ca
            if na:
                stats["a"]["frames_hit"] += 1

            # FPS + latency A — правый нижний угол панели A (тёмная подложка)
            _fps_txt_a = f"a {_fps_val_a:.0f}fps {_lat_a:.0f}ms"
            (_twA, _thA), _ = cv2.getTextSize(_fps_txt_a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(fa, (fa.shape[1] - _twA - 14, fa.shape[0] - _thA - 12),
                          (fa.shape[1] - 4, fa.shape[0] - 4), (0, 0, 0), -1)
            cv2.putText(fa, _fps_txt_a, (fa.shape[1] - _twA - 10, fa.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
            banner(fa, f"A: {Path(a.model_a).stem} @{a.imgsz_a}   TGT:{na}", (0, 255, 0))

            panels = [fa]

            # --- модель B ---
            if model_b is not None:
                _t_inf = time.time()
                rb = model_b.predict(frame, imgsz=a.imgsz_b, conf=a.conf, iou=a.iou,
                                     device=a.device, verbose=False)[0]
                _lat_b = (time.time() - _t_inf) * 1000.0
                _fps_nb += 1
                if not _fps_t0b:
                    _fps_t0b = time.time()
                elif time.time() - _fps_t0b >= 0.5:
                    _fps_val_b = _fps_nb / (time.time() - _fps_t0b)
                    _fps_t0b = time.time()
                    _fps_nb = 0

                _now = time.time()
                fb = frame.copy()
                # trail B: фильтр → отрисовка → аппенд (аналогично A)
                _trail_b[:] = [(tx, ty, tt) for tx, ty, tt in _trail_b if _now - tt < TRAIL_TTL]
                _draw_trail(fb, _trail_b, TRAIL_BGR_B, _now)
                nb, cb = draw_dets(fb, rb, (0, 200, 255), "B")
                if rb.boxes is not None and len(rb.boxes) > 0:
                    for b in rb.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = (int(v) for v in b)
                        _trail_b.append(((x1 + x2) // 2, (y1 + y2) // 2, _now))
                stats["b"]["dets"] += nb
                stats["b"]["confs"] += cb
                if nb:
                    stats["b"]["frames_hit"] += 1

                # FPS + latency B — правый нижний угол панели B
                _fps_txt_b = f"b {_fps_val_b:.0f}fps {_lat_b:.0f}ms"
                (_twB, _thB), _ = cv2.getTextSize(_fps_txt_b, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(fb, (fb.shape[1] - _twB - 14, fb.shape[0] - _thB - 12),
                              (fb.shape[1] - 4, fb.shape[0] - 4), (0, 0, 0), -1)
                cv2.putText(fb, _fps_txt_b, (fb.shape[1] - _twB - 10, fb.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
                banner(fb, f"B: {Path(a.model_b).stem} @{a.imgsz_b}   TGT:{nb}", (0, 200, 255))
                panels.append(fb)

            # --- склейка бок о бок ---
            h = int(panels[0].shape[0] * (a.width / panels[0].shape[1]))
            panels = [cv2.resize(p, (a.width, h)) for p in panels]
            view = np.hstack(panels) if len(panels) > 1 else panels[0]

            pos = f"{frames + a.start}" + (f"/{total}" if total else "")
            cv2.putText(view, f"frame {pos}", (8, view.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            draw_strazh(view)          # страж на view -- попадает и в --save, и в запись по R

            if a.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(a.save, fourcc, fps,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)

            # запись по R: одно side-by-side видео детектора (страж + боксы, без раскадровки)
            if recording:
                if rec_writer is None:
                    rec_path = f"compare_rec_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    rec_writer = cv2.VideoWriter(rec_path, fourcc, fps,
                                                 (view.shape[1], view.shape[0]))
                    print(f">>> RECORDING START -> {rec_path}")
                rec_writer.write(view)

            cv2.imshow("A | B   (SPACE - pause, R - recording, Q - exit)", view)

        k = cv2.waitKeyEx(1) & 0xFFFFFF
        if k in (ord('q'), ord('Q'), 0x439, 0x419, 0xE9, 0xC9, 1081, 1049, 233, 201):
            break
        if k == ord(' '):
            paused = not paused
        if k in RUN_KEYS:
            if not recording:
                recording = True
                _trail_a.clear()
                _trail_b.clear()
                print(">>> REC: recording from the next frame (R again -- stop)")
            else:
                recording = False
                if rec_writer is not None:
                    rec_writer.release()
                    print(f">>> STOP. Saved: {rec_path}")
                rec_writer = None

    if puller:
        puller.stop()
        time.sleep(0.1)
    cap.release()
    if writer:
        writer.release()
        print(f"[SAVED] {a.save}")
    if rec_writer is not None:
        rec_writer.release()
        print(f"[SAVED] {rec_path}")
    cv2.destroyAllWindows()

    # --- итог ---
    el = time.time() - t0
    title = f"RESULT over {frames} frames  ({el:.0f} s, {frames/max(el,1):.1f} proc.frame/s)"
    dropped = puller.dropped if puller else 0

    # собираем строки для таблицы и плейн-вывода: (model, name, frames hit, dets, avg conf)
    result_rows: list[tuple[str, str, str, str, str]] = []
    for key, name, imgsz in (("a", a.model_a, a.imgsz_a), ("b", a.model_b, a.imgsz_b)):
        if key == "b" and not a.model_b:
            continue
        st = stats[key]
        confs = st["confs"]
        avg = sum(confs) / len(confs) if confs else 0.0
        pct = 100.0 * st["frames_hit"] / frames if frames else 0
        hit_str = f"{st['frames_hit']}/{frames}  ({pct:.1f}%)"
        result_rows.append((key.upper(),
                            f"{Path(name).stem} @ {imgsz}",
                            hit_str, str(st["dets"]), f"{avg:.3f}"))

    if _HAS_RICH:
        tbl = Table(title=title)
        tbl.add_column("model", style="bold cyan")
        tbl.add_column("name", style="bold")
        tbl.add_column("frames hit", justify="right")
        tbl.add_column("dets", justify="right")
        tbl.add_column("avg conf", justify="right")
        for key, name, hit_str, dets, avg_str in result_rows:
            tbl.add_row(key, name, hit_str, dets, avg_str)
        _console.print(tbl)
        if dropped:
            _console.print(f"[dim]frames dropped (could not keep up): {dropped}[/]")
    else:
        print("\n" + "=" * 70)
        print(title)
        if dropped:
            print(f"Frames dropped (could not keep up): {dropped}")
        print("=" * 70)
        for key, name, hit_str, dets, avg_str in result_rows:
            print(f"\n[{key}] {name}")
            print(f"    frames with target: {hit_str}")
            print(f"    total detections:   {dets}")
            print(f"    average conf:       {avg_str}")

    if a.model_b:
        da = stats["a"]["frames_hit"]
        db = stats["b"]["frames_hit"]
        print("\n" + "-" * 70)
        if db > da:
            print(f"B found the target on {db - da} MORE frames than A  (+{100*(db-da)/max(1,da):.0f}%)")
            print("=> 1280 sees farther. Hypothesis confirmed.")
        elif db < da:
            print(f"A found the target on {da - db} more frames than B")
            print("=> 1280 gave NO gain. The problem is most likely in optics/camera,")
            print("   not in inference resolution. Too early to buy hardware.")
        else:
            print("Equal by frames-with-target — look at conf and at it with your eyes.")
        print("-" * 70)
        print("IMPORTANT: do not count numbers only. Scroll through where the target is FAR,")
        print("and watch with your eyes: which panel catches it and which loses it.")


if __name__ == "__main__":
    main()
