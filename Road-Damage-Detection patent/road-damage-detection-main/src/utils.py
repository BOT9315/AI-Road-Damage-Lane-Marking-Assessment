"""
utils.py
========
Shared helper functions: config loading, severity scoring,
drawing detections, and simple report generation.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import yaml

# Fixed color per class (BGR) for consistent visualization
CLASS_COLORS = {
    0: (66, 135, 245),   # longitudinal crack - blue
    1: (66, 245, 194),   # transverse crack - teal
    2: (245, 195, 66),   # alligator crack - orange
    3: (36, 28, 237),    # pothole - red (most severe, visually)
    4: (200, 200, 200),  # crosswalk blur - gray
    5: (150, 150, 255),  # whiteline blur - light red
}


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def bbox_area_ratio(box, img_w, img_h):
    """box = (x1, y1, x2, y2) in pixel coords. Returns fraction of image area covered."""
    x1, y1, x2, y2 = box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    return box_area / float(img_w * img_h)


def estimate_severity(class_id, area_ratio):
    """
    Heuristic severity scoring combining damage type and relative size.
    Potholes and alligator cracks are inherently more dangerous than
    faded markings, so class type shifts the base severity level.
    """
    base_weight = {0: 1.0, 1: 1.0, 2: 1.6, 3: 2.0, 4: 0.4, 5: 0.4}.get(class_id, 1.0)
    score = area_ratio * 100 * base_weight

    if score < 0.5:
        return "Low"
    elif score < 2.0:
        return "Medium"
    elif score < 6.0:
        return "High"
    else:
        return "Critical"


def draw_detections(image, detections, class_names):
    """
    detections: list of dicts with keys:
        class_id, confidence, box (x1,y1,x2,y2), severity
    """
    img = image.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        color = CLASS_COLORS.get(det["class_id"], (255, 255, 255))
        label = f"{class_names[det['class_id']]} {det['confidence']:.2f} [{det['severity']}]"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def summarize_detections(detections, class_names):
    """Aggregate counts per class and severity for reporting."""
    summary = {
        "total_detections": len(detections),
        "by_class": {},
        "by_severity": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
    }
    for det in detections:
        cname = class_names[det["class_id"]]
        summary["by_class"][cname] = summary["by_class"].get(cname, 0) + 1
        summary["by_severity"][det["severity"]] += 1
    return summary


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
