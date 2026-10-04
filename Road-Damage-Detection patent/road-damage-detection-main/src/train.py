"""
train.py
========
Trains the road damage detection model (YOLOv8) using settings
defined in config.yaml. Saves best/last checkpoints and logs
metrics (precision, recall, mAP) via Ultralytics' built-in logger.

Usage
-----
    python src/train.py --config config.yaml
    python src/train.py --config config.yaml --epochs 50 --batch 8   # override
"""

import argparse
import shutil
from pathlib import Path

from src.model import RoadDamageModel
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train road damage detection model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default="data/road_damage.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    epochs = args.epochs or train_cfg["epochs"]
    batch = args.batch or train_cfg["batch_size"]
    img_size = args.img_size or train_cfg["img_size"]

    print("=" * 60)
    print("Road Damage Detection — Training")
    print("=" * 60)
    print(f"Architecture : {model_cfg['architecture']} ({model_cfg['variant']})")
    print(f"Data config  : {args.data}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {batch}")
    print(f"Image size   : {img_size}")
    print(f"Device       : {cfg['project']['device']}")
    print("=" * 60)

    model = RoadDamageModel(weights=model_cfg["variant"], device=cfg["project"]["device"])

    results = model.train(
        data_yaml=args.data,
        epochs=epochs,
        img_size=img_size,
        batch_size=batch,
        project_dir=train_cfg["save_dir"],
        lr0=train_cfg["lr0"],
        optimizer=train_cfg["optimizer"],
        patience=train_cfg["patience"],
        augment=train_cfg["augment"],
        mosaic=train_cfg["mosaic"],
        mixup=train_cfg["mixup"],
    )

    # Copy best weights to models/best.pt for easy downstream use
    run_dir = Path(train_cfg["save_dir"]) / "road_damage_run" / "weights"
    best_weights = run_dir / "best.pt"
    if best_weights.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy(best_weights, "models/best.pt")
        print(f"\nBest weights copied to models/best.pt")

    print("\nTraining complete. Metrics and plots saved under:")
    print(f"  {train_cfg['save_dir']}/road_damage_run/")


if __name__ == "__main__":
    main()
