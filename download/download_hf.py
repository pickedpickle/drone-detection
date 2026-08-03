"""
Скачивание датасетов с HuggingFace
"""

from huggingface_hub import snapshot_download
from pathlib import Path

# Датасеты для скачивания
datasets = [
    "lgrzybowski/seraphim-drone-detection-dataset",  # 83,483 images
    "pathikg/drone-detection-dataset",                # varying scales
    # "silveroupti/VisDrone",                      # пропускаем (уже в датасете)
]

output_dir = Path("./huggingface_downloads")

for dataset_id in datasets:
    print(f"\nСкачиваю {dataset_id}...")

    try:
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=output_dir / dataset_id.split("/")[-1],
            local_dir_use_symlinks=False,
            token=True  # если требуется авторизация
        )
        print(f"✅ Готово: {dataset_id}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

print(f"\nВсе датасеты скачаны в: {output_dir}")
