"""
Basic unit tests for src/utils.py.
Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import bbox_area_ratio, estimate_severity, summarize_detections


def test_bbox_area_ratio_full_image():
    ratio = bbox_area_ratio((0, 0, 100, 100), 100, 100)
    assert abs(ratio - 1.0) < 1e-6


def test_bbox_area_ratio_partial():
    ratio = bbox_area_ratio((0, 0, 50, 50), 100, 100)
    assert abs(ratio - 0.25) < 1e-6


def test_estimate_severity_low():
    assert estimate_severity(class_id=0, area_ratio=0.001) == "Low"


def test_estimate_severity_pothole_critical():
    # Pothole (class 3) has higher weight, so a mid-size box should
    # escalate faster than a crack of the same relative size.
    severity = estimate_severity(class_id=3, area_ratio=0.05)
    assert severity in ("High", "Critical")


def test_summarize_detections_counts():
    class_names = {0: "crack", 3: "pothole"}
    dets = [
        {"class_id": 0, "severity": "Low"},
        {"class_id": 3, "severity": "Critical"},
        {"class_id": 3, "severity": "High"},
    ]
    summary = summarize_detections(dets, class_names)
    assert summary["total_detections"] == 3
    assert summary["by_class"]["pothole"] == 2
    assert summary["by_severity"]["Critical"] == 1
    assert summary["by_severity"]["High"] == 1
