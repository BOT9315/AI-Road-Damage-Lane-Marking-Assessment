"""
evaluate.py
===========
Evaluates a trained checkpoint against the validation/test split and
produces a metrics report (precision, recall, mAP50, mAP50-95, F1
per class) as JSON + a confusion-matrix style summary plot.

Usage
-----
    python src/evaluate.py --weights models/best.pt --data data/road_damage.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.model import RoadDamageModel
from src.utils import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate road damage detection model")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--data", default="data/road_damage.yaml")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = list(cfg["dataset"]["classes"].values())
    report_dir = ensure_dir(cfg["evaluate"]["report_dir"])

    model = RoadDamageModel(weights=args.weights, device=cfg["project"]["device"])
    metrics = model.validate(args.data, img_size=cfg["dataset"]["img_size"])

    # Ultralytics DetMetrics object exposes per-class and aggregate results
    results = {
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": {},
    }

    for i, cname in enumerate(class_names):
        try:
            results["per_class"][cname] = {
                "precision": float(metrics.box.p[i]),
                "recall": float(metrics.box.r[i]),
                "mAP50": float(metrics.box.ap50[i]),
            }
        except (IndexError, TypeError):
            continue

    report_path = Path(report_dir) / "eval_metrics.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(json.dumps(results["overall"], indent=2))
    print(f"\nFull report saved to: {report_path}")

    # Simple bar chart of mAP50 per class
    if results["per_class"]:
        names = list(results["per_class"].keys())
        maps = [v["mAP50"] for v in results["per_class"].values()]
        plt.figure(figsize=(9, 5))
        plt.bar(names, maps, color="#3b82f6")
        plt.ylabel("mAP@0.5")
        plt.title("Per-Class mAP@0.5")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plot_path = Path(report_dir) / "per_class_map50.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
