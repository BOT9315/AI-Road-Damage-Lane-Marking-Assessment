"""
download_dataset.py
====================
Helper for obtaining a public road-damage dataset to train/test with.

This project is dataset-agnostic (see src/dataset.py for VOC/COCO
converters), but the recommended starting point is RDD2022
(Road Damage Dataset 2022), used in the CRDDC'2022 challenge, which
contains ~47,000 labeled images of road damage across multiple
countries in Pascal-VOC format with the exact class taxonomy used
in config.yaml (D00, D10, D20, D40, D43, D44).

This script does NOT hardcode a download URL (dataset hosting moves
around / requires accepting terms), so it just prints instructions
and validates that a manually-downloaded copy is laid out correctly.

Steps to obtain data:
  1. Visit: https://github.com/sekilab/RoadDamageDetector
  2. Download the country subset(s) you want (e.g. Japan, India, Czech, USA).
  3. Extract into data/raw/ so you have:
        data/raw/images/*.jpg
        data/raw/annotations/*.xml
  4. Run: python src/dataset.py --format voc \
              --images data/raw/images \
              --annotations data/raw/annotations \
              --out data/processed

Alternative smaller datasets (good for quick prototyping):
  - Kaggle "Pothole Detection Dataset" (COCO format, potholes only)
  - Kaggle "Crack Detection Dataset" (classification, needs bbox re-labeling)
"""

import sys
from pathlib import Path


def check_layout(raw_dir="data/raw"):
    images_dir = Path(raw_dir) / "images"
    annotations_dir = Path(raw_dir) / "annotations"

    ok = True
    if not images_dir.exists() or not any(images_dir.glob("*")):
        print(f"[MISSING] {images_dir} is empty or does not exist.")
        ok = False
    else:
        n_images = len(list(images_dir.glob("*")))
        print(f"[OK] Found {n_images} files in {images_dir}")

    if not annotations_dir.exists() or not any(annotations_dir.glob("*")):
        print(f"[MISSING] {annotations_dir} is empty or does not exist.")
        ok = False
    else:
        n_annots = len(list(annotations_dir.glob("*")))
        print(f"[OK] Found {n_annots} files in {annotations_dir}")

    if ok:
        print("\nDataset layout looks correct. Next step:")
        print("  python src/dataset.py --format voc --images data/raw/images "
              "--annotations data/raw/annotations --out data/processed")
    else:
        print("\nSee the docstring at the top of this file for download instructions.")
    return ok


if __name__ == "__main__":
    check_layout()
    sys.exit(0)
