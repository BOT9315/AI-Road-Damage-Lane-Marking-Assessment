"""
inference.py
============
Run trained model on a single image, a folder of images, or a video,
draw annotated detections, estimate severity, and optionally export
a JSON/CSV summary report.

Usage
-----
    # single image
    python src/inference.py --source path/to/image.jpg --weights models/best.pt

    # folder of images
    python src/inference.py --source path/to/folder --weights models/best.pt

    # video
    python src/inference.py --source path/to/video.mp4 --weights models/best.pt --video
"""

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from src.model import RoadDamageModel
from src.utils import (draw_detections, ensure_dir, load_config,
                        summarize_detections)


def run_on_image(model, image_path, class_names, cfg, out_dir):
    image = cv2.imread(str(image_path))
    results = model.predict(
        source=str(image_path),
        conf=cfg["inference"]["conf_threshold"],
        iou=cfg["inference"]["iou_threshold"],
        img_size=cfg["dataset"]["img_size"],
        max_det=cfg["inference"]["max_detections"],
    )
    detections = RoadDamageModel.parse_results(results, class_names)
    annotated = draw_detections(image, detections, class_names)

    out_path = Path(out_dir) / f"annotated_{Path(image_path).name}"
    cv2.imwrite(str(out_path), annotated)

    summary = summarize_detections(detections, class_names)
    return detections, summary, out_path


def run_on_video(model, video_path, class_names, cfg, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(out_dir) / f"annotated_{Path(video_path).stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_detections = []
    pbar = tqdm(total=total_frames, desc="Processing video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(
            source=frame,
            conf=cfg["inference"]["conf_threshold"],
            iou=cfg["inference"]["iou_threshold"],
            img_size=cfg["dataset"]["img_size"],
            max_det=cfg["inference"]["max_detections"],
        )
        detections = RoadDamageModel.parse_results(results, class_names)
        all_detections.extend(detections)
        annotated = draw_detections(frame, detections, class_names)
        writer.write(annotated)
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()

    summary = summarize_detections(all_detections, class_names)
    return all_detections, summary, out_path


def main():
    parser = argparse.ArgumentParser(description="Run road damage inference")
    parser.add_argument("--source", required=True, help="image, folder, or video path")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", action="store_true", help="treat --source as a video file")
    parser.add_argument("--out", default="outputs/predictions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = cfg["dataset"]["classes"]
    weights = args.weights or cfg["inference"]["weights_path"]
    out_dir = ensure_dir(args.out)

    model = RoadDamageModel(weights=weights, device=cfg["project"]["device"])

    if args.video:
        detections, summary, out_path = run_on_video(model, args.source, class_names, cfg, out_dir)
        results_payload = {"source": args.source, "summary": summary}
    else:
        source_path = Path(args.source)
        if source_path.is_dir():
            all_summaries = {}
            for img_file in sorted(source_path.glob("*")):
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                _, summary, out_path = run_on_image(model, img_file, class_names, cfg, out_dir)
                all_summaries[img_file.name] = summary
                print(f"{img_file.name}: {summary['total_detections']} detections -> {out_path}")
            results_payload = {"source": args.source, "per_image_summary": all_summaries}
        else:
            detections, summary, out_path = run_on_image(model, source_path, class_names, cfg, out_dir)
            results_payload = {"source": args.source, "summary": summary}
            print(f"Detections: {summary['total_detections']}")
            print(f"Saved annotated image to: {out_path}")

    report_path = Path(out_dir) / "report.json"
    with open(report_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
