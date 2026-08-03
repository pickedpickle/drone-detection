#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
autolabel_sort.py — авторазметка своей моделью + сортировка по числу детекций.

Прогоняет best.pt по ВСЕМ фото из --src (рекурсивно, много подпапок) и
раскладывает результат по трём папкам в --out:

  1_single/   — кадры, где найден РОВНО ОДИН объект   -> images/ + labels/
  2_multi/    — кадры, где найдено ДВА И БОЛЕЕ         -> images/ + labels/
  3_empty/    — кадры БЕЗ детекций                     -> только images/ (без .txt)

Каждому кадру даётся СКВОЗНОЕ уникальное имя 000000, 000001, ... (не повторяется
нигде). Фото и его разметка получают ОДНО и то же имя: 000123.jpg / 000123.txt.
Разметка в формате YOLO, один класс -> id 0. Исходники КОПИРУЮТСЯ (оригиналы целы).
Пишется mapping.csv (новое_имя -> откуда взялось) для прослеживаемости.

ЗАПУСКАТЬ НА НОУТБУКЕ (там ultralytics и веса).

  python autolabel_sort.py --src "C:\Users\yaraz\Desktop\Новая папка (3)\no_label" --out "C:\yaraz\Desktop\sorted"

Модель по умолчанию — твоя drone_max_distance_v1. conf по умолчанию 0.25.
"""
import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_MODEL = r"C:\Users\yaraz\Desktop\dataset_final\runs\detect\drone_max_distance_v1\weights\best.pt"


def main():
    ap = argparse.ArgumentParser(description="Авторазметка + сортировка по числу детекций")
    ap.add_argument("--src", required=True, help="родительская папка с подпапками фото")
    ap.add_argument("--out", required=True, help="куда сложить 1_single / 2_multi / 3_empty")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="путь к best.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="порог уверенности (по умолч. 0.25)")
    ap.add_argument("--imgsz", type=int, default=640, help="размер входа (как при обучении = 640)")
    ap.add_argument("--device", default="0", help="0 = GPU, cpu = процессор")
    ap.add_argument("--pad", type=int, default=6, help="разрядность сквозного номера")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out)
    if not src.is_dir():
        print(f"[ОШИБКА] Нет папки: {src}")
        return

    # Готовим выходную структуру
    dirs = {
        "single": out / "1_single",
        "multi": out / "2_multi",
        "empty": out / "3_empty",
    }
    (dirs["single"] / "images").mkdir(parents=True, exist_ok=True)
    (dirs["single"] / "labels").mkdir(parents=True, exist_ok=True)
    (dirs["multi"] / "images").mkdir(parents=True, exist_ok=True)
    (dirs["multi"] / "labels").mkdir(parents=True, exist_ok=True)
    (dirs["empty"] / "images").mkdir(parents=True, exist_ok=True)

    # Собираем все картинки рекурсивно
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[ОШИБКА] В {src} не найдено изображений.")
        return
    print(f"[ИНФО] Найдено {len(images)} кадров. Загружаю модель...")

    model = YOLO(a.model)
    print(f"[ИНФО] Модель: {a.model}")
    print(f"[ИНФО] conf={a.conf}, imgsz={a.imgsz}, device={a.device}\n")

    counters = {"single": 0, "multi": 0, "empty": 0}
    map_rows = []
    n = 0

    for img_path in images:
        res = model.predict(str(img_path), conf=a.conf, imgsz=a.imgsz,
                            device=a.device, verbose=False)[0]

        boxes = res.boxes
        count = 0 if boxes is None else len(boxes)

        if count == 0:
            bucket = "empty"
        elif count == 1:
            bucket = "single"
        else:
            bucket = "multi"

        name = f"{n:0{a.pad}d}"
        ext = img_path.suffix.lower()

        # Копируем изображение
        dst_img = dirs[bucket] / "images" / f"{name}{ext}"
        shutil.copy2(img_path, dst_img)

        # Пишем разметку YOLO (кроме пустых)
        if bucket != "empty":
            xywhn = boxes.xywhn.cpu().numpy()  # cx, cy, w, h (нормированные)
            lines = []
            for (cx, cy, w, h) in xywhn:
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            dst_lbl = dirs[bucket] / "labels" / f"{name}.txt"
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        map_rows.append((name + ext, bucket, count, str(img_path)))
        counters[bucket] += 1
        n += 1

        if n % 200 == 0:
            print(f"  обработано {n}/{len(images)}  "
                  f"[1:{counters['single']}  2+:{counters['multi']}  0:{counters['empty']}]")

    # mapping.csv
    with open(out / "mapping.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["new_name", "bucket", "num_detections", "source_path"])
        wr.writerows(map_rows)

    print("\n" + "=" * 60)
    print("[ГОТОВО] Раскладка завершена.")
    print(f"   1_single (1 объект):    {counters['single']}")
    print(f"   2_multi  (2+ объекта):  {counters['multi']}")
    print(f"   3_empty  (без детекций):{counters['empty']}")
    print(f"   Всего:                  {n}")
    print(f"   mapping.csv: {out / 'mapping.csv'}")
    print("=" * 60)
    print("Дальше: выборочно проглядь 3_empty (нет ли пропущенных дальних точек)")
    print("и 2_multi (нет ли ложняков), потом собирай финальный датасет.")


if __name__ == "__main__":
    main()
