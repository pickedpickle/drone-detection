# Drone Detection Post

Early-warning **drone detection post** — a stationary observation point that detects aerial drones (quadcopters, fixed-wing such as Shahed/Geran) on RTSP camera streams and raises a hardware alert. **Defensive: detection + alerting only, no countermeasure.**

Personal, non-commercial research project by a self-taught developer. Work in progress.

## Why
Catch a drone at the edge of visibility, where it occupies just a few pixels. Priority is **recall** — a false alarm is cheaper than a missed drone. The whole pipeline is tuned for long-range detection on telephoto cameras (imgsz=1280 to give a far target ~4× more pixels than 640).

## What's here
Tooling around the detection pipeline (the live detector — `GRIDEN` — is being added):

- **`compare_models.py`** — run two YOLO models side-by-side on live video / RTSP / SDP: crop, distance estimation, on-screen overlay. The main field-test tool.
- **`export_ncnn_1280.py`** — export `.pt` → NCNN for edge deployment (Raspberry Pi 5, etc.).
- **`autolabel_sort.py`** — sort / auto-label frames for dataset building.
- **`download/`** — HuggingFace / Kaggle dataset downloaders.

## What's NOT here (by design)
- **Model weights** (`.pt`, NCNN) — not in the repo.
- **Datasets** — not in the repo.

Both are available **on request** to verified researchers and enthusiasts working on defensive early-warning. Open an issue describing who you are and what you're building — requests are reviewed individually.

## Approach & honest findings
- **Public drone datasets are mostly close-up shots at 640px.** Long-range (3–5px) targets barely exist in them. The real lever is **own telephoto footage with hand-labeled distant targets**, not more downloads.
- **Detection range is limited by data, not model capacity or resolution alone.** We measure target-size distribution before trusting any dataset — otherwise it just reinforces the close-up bias.
- **Val metrics can lie about range:** a val set labeled by a weaker teacher model penalizes the student for detecting distant targets the teacher missed. Trust field tests on real video.
- Going 640 → 1280 helps (effective stride drops, far target gets grid cells), but the ceiling is the absence of long-range targets in the training data.

## Stack
YOLOv8 (n / s) @ imgsz=1280, ultralytics, NCNN for edge. Python 3.13, OpenCV, PyTorch (CUDA). ESP32 for the hardware alert (one-shot per-drone signaling across 4 lamps).

## Status
Personal research, active development. Not production. Expect rough edges — they're documented in commits and notes.

## Disclaimer
Defensive early-warning system: **detection and alerting only**. No targeting, no countermeasure, no offensive use. Built for observation-post and civil-protection scenarios.
