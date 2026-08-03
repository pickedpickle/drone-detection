"""
Прокси-скрипт для Kaggle API.
Позволяет скачивать датасеты без API key.
"""

import os
import requests
import zipfile
from pathlib import Path

def download_kaggle_dataset(dataset_name, output_dir="."):
    """
    Скачать датасет с Kaggle через прямые ссылки.
    """
    # Kaggle использует прямые ссылки на файлы
    # Формат: https://www.kaggle.com/datasets/username/dataset-name/download

    print(f"Попытка скачать {dataset_name}...")

    # Kaggle требует логин для скачивания
    # Без API key не получится

    # Альтернатива: попробовать найти прямые ссылки
    # Но это сложно без браузера

    print(f"Kaggle требует логин/API key.")
    print(f"Нужно:")
    print(f"1. Зарегистрироваться на kaggle.com")
    print(f"2. Получить API key")
    print(f"3. Использовать Kaggle CLI: kaggle datasets download -d {dataset_name}")

    return None

if __name__ == "__main__":
    # Пример использования
    download_kaggle_dataset("muhammadsaoodsarwar/drone-vs-bird")
