"""
dataset.py
==========
Prepares raw road-damage imagery + annotations into a YOLO-format
dataset (images/, labels/ with train/val/test splits).

Supports two common input annotation formats:
  1. Pascal VOC XML  (RDD2022 default format)
  2. COCO JSON

Usage
-----
    python src/dataset.py --format voc \
        --images data/raw/images \
        --annotations data/raw/annotations \
        --out data/processed

    python src/dataset.py --format coco \
        --images data/raw/images \
        --annotations data/raw/annotations.json \
        --out data/processed
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CLASS_MAP = {
    "D00": 0, "D01": 0,             # longitudinal crack variants
    "D10": 1, "D11": 1,             # transverse crack variants
    "D20": 2,                       # alligator crack
    "D40": 3,                       # pothole
    "D43": 4,                       # crosswalk blur
    "D44": 5,                       # white line blur
}


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def voc_to_yolo_box(size, box):
    """Convert (xmin, ymin, xmax, ymax) -> normalized YOLO (x_c, y_c, w, h)."""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x_c = (box[0] + box[2]) / 2.0 * dw
    y_c = (box[1] + box[3]) / 2.0 * dh
    w = (box[2] - box[0]) * dw
    h = (box[3] - box[1]) * dh
    return x_c, y_c, w, h


def convert_voc_annotation(xml_path, out_label_path):
    """Parse one Pascal-VOC XML file and write a YOLO .txt label file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        cls_id = CLASS_MAP.get(cls_name)
        if cls_id is None:
            continue  # skip unknown / background classes
        bnd = obj.find("bndbox")
        box = (
            float(bnd.find("xmin").text),
            float(bnd.find("ymin").text),
            float(bnd.find("xmax").text),
            float(bnd.find("ymax").text),
        )
        x_c, y_c, bw, bh = voc_to_yolo_box((w, h), box)
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

    with open(out_label_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


def convert_coco_annotations(coco_json_path, out_label_dir, images_dir):
    """Convert a single COCO annotations JSON into per-image YOLO txt files."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    cat_lookup = {c["id"]: c["name"] for c in coco["categories"]}

    per_image_labels = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]  # COCO: x,y,width,height (top-left origin)
        cls_name = cat_lookup[ann["category_id"]]
        cls_id = CLASS_MAP.get(cls_name)
        if cls_id is None:
            continue
        x_c = (x + bw / 2) / w
        y_c = (y + bh / 2) / h
        nw, nh = bw / w, bh / h
        per_image_labels[ann["image_id"]].append(
            f"{cls_id} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}"
        )

    count = 0
    for img_id, lines in per_image_labels.items():
        fname = Path(images[img_id]["file_name"]).stem + ".txt"
        with open(Path(out_label_dir) / fname, "w") as f:
            f.write("\n".join(lines))
        count += len(lines)
    return count


def build_splits(image_files, train_split, val_split, seed=42):
    random.seed(seed)
    train_files, temp_files = train_test_split(
        image_files, train_size=train_split, random_state=seed
    )
    relative_val = val_split / (1 - train_split)
    val_files, test_files = train_test_split(
        temp_files, train_size=relative_val, random_state=seed
    )
    return train_files, val_files, test_files


def organize_dataset(images_dir, labels_dir, out_dir, splits):
    """Copy images/labels into images/{train,val,test} and labels/{train,val,test}."""
    for split_name, files in zip(["train", "val", "test"], splits):
        img_out = Path(out_dir) / "images" / split_name
        lbl_out = Path(out_dir) / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(files, desc=f"Organizing {split_name}"):
            img_path = Path(img_path)
            label_path = Path(labels_dir) / (img_path.stem + ".txt")

            shutil.copy(img_path, img_out / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, lbl_out / label_path.name)
            else:
                # create empty label file (image with no damage / background)
                (lbl_out / (img_path.stem + ".txt")).touch()


def main():
    parser = argparse.ArgumentParser(description="Prepare road damage dataset for YOLO training")
    parser.add_argument("--format", choices=["voc", "coco"], required=True)
    parser.add_argument("--images", required=True, help="Path to raw images folder")
    parser.add_argument("--annotations", required=True,
                         help="VOC: folder of .xml files. COCO: path to json file")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)["dataset"]
    tmp_labels = Path(args.out) / "_tmp_labels"
    tmp_labels.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    if args.format == "voc":
        xml_files = list(Path(args.annotations).glob("*.xml"))
        for xml_file in tqdm(xml_files, desc="Converting VOC -> YOLO"):
            out_path = tmp_labels / (xml_file.stem + ".txt")
            total_boxes += convert_voc_annotation(xml_file, out_path)
    else:
        total_boxes = convert_coco_annotations(args.annotations, tmp_labels, args.images)

    image_files = sorted(
        [p for p in Path(args.images).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    print(f"Found {len(image_files)} images, {total_boxes} annotated boxes.")

    splits = build_splits(image_files, cfg["train_split"], cfg["val_split"])
    organize_dataset(args.images, tmp_labels, args.out, splits)

    shutil.rmtree(tmp_labels)
    print(f"Dataset ready at: {args.out}")
    print(f"  train: {len(splits[0])} | val: {len(splits[1])} | test: {len(splits[2])}")


if __name__ == "__main__":
    main()
