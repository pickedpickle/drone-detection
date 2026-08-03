#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
export_ncnn_1280.py — экспорт СУЩЕСТВУЮЩЕЙ best.pt в NCNN на вход 1280.

!!! ВАЖНО — ЧЕСТНО !!!
Это ЭКСПЕРИМЕНТ, не «портирование». Модель обучена на 640; переэкспорт в 1280
меняет только размер входа, но НЕ дообучает веса под мелкие дальние цели.
Дальность может вырасти, а может и нет — это грубая проверка гипотезы.
Правильный путь к дальности: дообучить best.pt на imgsz=1280 (train, датасет тот же).

Экспорт запускать на НОУТБУКЕ (где ultralytics + best.pt).

  python export_ncnn_1280.py --model best.pt
  python export_ncnn_1280.py --model best.pt --imgsz 1280 --val data.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Экспорт best.pt в NCNN на 1280 (эксперимент)")
    ap.add_argument("--model", required=True, help="путь к best.pt")
    ap.add_argument("--imgsz", type=int, default=1280, help="размер входа (1280 для теста дальности)")
    ap.add_argument("--half", action="store_true", default=True, help="fp16 (по умолчанию вкл)")
    ap.add_argument("--no-half", dest="half", action="store_false", help="выключить fp16")
    ap.add_argument("--val", default="", help="data.yaml — прогнать валидацию после экспорта (опц.)")
    a = ap.parse_args()

    from ultralytics import YOLO

    model_path = Path(a.model)
    if not model_path.exists():
        print(f"[ОШИБКА] Не найдена модель: {model_path}")
        sys.exit(1)

    print(f"[ЭКСПОРТ] {model_path} -> NCNN, imgsz={a.imgsz}, half={a.half}")
    print("          ВНИМАНИЕ: веса от 640-обучения, это грубый тест дальности.")
    model = YOLO(str(model_path))
    out = model.export(format="ncnn", imgsz=a.imgsz, half=a.half)
    print(f"[ОК] Экспортировано: {out}")
    print(f"     Папка NCNN обычно: {model_path.parent}/{model_path.stem}_ncnn_model/")
    print(f"     Внутри: model.ncnn.param + model.ncnn.bin")
    print(f"     На Pi поставь INPUT_SIZE = {a.imgsz} (ОБЯЗАНО совпасть с этим экспортом!)")

    # Необязательная валидация в отдельном процессе (чтобы избежать глюка torch.cuda после export)
    if a.val:
        print(f"\n[VAL] Прогоняю валидацию на {a.val} (imgsz={a.imgsz})...")
        code = (
            "from ultralytics import YOLO; "
            f"m=YOLO(r'{out}'); "
            f"r=m.val(data=r'{a.val}', imgsz={a.imgsz}); "
            "print('mAP50=%.4f mAP50-95=%.4f'%(r.box.map50, r.box.map))"
        )
        subprocess.run([sys.executable, "-c", code])


if __name__ == "__main__":
    main()
