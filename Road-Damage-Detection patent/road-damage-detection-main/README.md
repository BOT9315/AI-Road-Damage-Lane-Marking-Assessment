# 🛣️ AI-Powered Road Damage Detection Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

An end-to-end computer vision system that detects and classifies road
surface damage — potholes, cracks, and faded lane markings — from
images and video, estimates severity, and presents results through an
interactive web app. Built on **YOLOv8** (Ultralytics) for real-time
object detection.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Damage Taxonomy](#-damage-taxonomy)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference (CLI)](#-running-inference-cli)
- [Web App Demo](#-web-app-demo)
- [Severity Estimation Logic](#-severity-estimation-logic)
- [Model Export](#-model-export-deployment)
- [Limitations & Future Work](#-limitations--future-work)
- [Testing](#-testing)
- [License & Attribution](#-license--attribution)

---

## 🎯 Problem Statement

Manual road inspection is slow, expensive, and inconsistent across
inspectors. This project automates the process: a camera-equipped
vehicle, drone, or even a smartphone can capture road footage, and the
system automatically detects damage, classifies its type, and scores
its severity — producing a prioritized maintenance report for city or
highway authorities.

## 🎯 Objectives

- Detect and localize road damage in images/video using bounding boxes
- Classify damage into standard categories (cracks, potholes, faded markings)
- Estimate severity (Low / Medium / High / Critical) from damage size and type
- Provide a usable demo interface (web app) for non-technical stakeholders
- Provide a reproducible training/evaluation pipeline for improving the model over time

## 🏷️ Damage Taxonomy

This project follows the standard taxonomy from the RDD2022 (Road
Damage Dataset) / CRDDC benchmark:

| Class ID | Code | Description            |
|----------|------|-------------------------|
| 0        | D00  | Longitudinal crack      |
| 1        | D10  | Transverse crack        |
| 2        | D20  | Alligator (mesh) crack  |
| 3        | D40  | Pothole                 |
| 4        | D43  | Crosswalk blur          |
| 5        | D44  | White line blur         |

---

## 📁 Project Structure

```
road-damage-detection/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.yaml                # Central configuration (paths, hyperparameters)
├── check_setup.py             # Verifies environment before running anything
├── .gitignore                 # Excludes venv, datasets, weights from Git
│
├── data/
│   ├── raw/                   # Place downloaded raw images + annotations here
│   ├── processed/             # Auto-generated YOLO-format train/val/test split
│   ├── road_damage.yaml       # YOLO dataset descriptor (classes, paths)
│   └── download_dataset.py    # Instructions + validation for dataset setup
│
├── src/
│   ├── dataset.py             # VOC/COCO -> YOLO format conversion + split
│   ├── model.py                # YOLOv8 wrapper (train/predict/export/parse)
│   ├── train.py                # Training entry point
│   ├── inference.py           # Run detection on image / folder / video (CLI)
│   ├── evaluate.py            # Compute precision/recall/mAP
│   └── utils.py                # Severity scoring, drawing, reporting helpers
│
├── app/
│   └── app.py                  # Streamlit web app (image / video / batch demo)
│
├── models/                     # Trained weights land here (best.pt, last.pt)
├── outputs/                    # Predictions, evaluation reports, plots
├── tests/
│   └── test_utils.py          # Unit tests for severity scoring / utilities
└── notebooks/                  # Optional exploratory analysis notebooks
```

---

## 🏗️ System Architecture

```
 ┌────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐
 │  Raw Data   │───▶│  Preprocess   │───▶│  YOLOv8 Model  │───▶│  Post-processing │
 │ (images +   │    │ (VOC/COCO ->  │    │   Training     │    │ (severity score, │
 │  XML/JSON)  │    │  YOLO format) │    │  (src/train.py)│    │  drawing boxes)  │
 └────────────┘    └──────────────┘    └───────┬───────┘    └────────┬─────────┘
                                                │                     │
                                                ▼                     ▼
                                        models/best.pt      ┌──────────────────┐
                                                │            │   Streamlit App   │
                                                └───────────▶│  (app/app.py) /    │
                                                             │  CLI inference.py  │
                                                             └──────────────────┘
```

**Why YOLOv8?** Road damage detection needs to run on constrained
hardware (dashcams, edge devices on inspection vehicles) while staying
accurate on small objects like cracks. YOLOv8 offers a strong
speed/accuracy tradeoff with multiple model sizes (n/s/m/l/x), native
augmentation (mosaic, mixup), and straightforward ONNX/TorchScript
export for deployment.

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/BOT9315/road-damage-detection.git
cd road-damage-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify everything is installed correctly
python check_setup.py
```

**Requirements:** Python 3.9+, and ideally a CUDA-capable GPU for
training (CPU works for inference at reduced speed). Set
`project.device` in `config.yaml` to `"cpu"` if no GPU is available.

---

## 📊 Dataset Setup

1. Download a road damage dataset. Recommended: **RDD2022**
   (https://github.com/sekilab/RoadDamageDetector) — ~47K labeled
   images across multiple countries, Pascal-VOC format, matches the
   class taxonomy above out of the box.
2. Arrange raw files as:
   ```
   data/raw/images/*.jpg
   data/raw/annotations/*.xml
   ```
3. Validate the layout:
   ```bash
   python data/download_dataset.py
   ```
4. Convert to YOLO format with a train/val/test split (75/15/10 by default):
   ```bash
   python src/dataset.py --format voc \
       --images data/raw/images \
       --annotations data/raw/annotations \
       --out data/processed
   ```
   COCO-format datasets are also supported via `--format coco`.

This produces:
```
data/processed/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

> **Note:** Raw datasets are NOT included in this repo (see `.gitignore`) due to size. You must download them separately.

---

## 🏋️ Training

```bash
python src/train.py --config config.yaml
```

Key hyperparameters (edit in `config.yaml` under `train:`):

| Parameter     | Default | Notes                                    |
|---------------|---------|-------------------------------------------|
| epochs        | 100     | Use `patience: 20` for early stopping     |
| batch_size    | 16      | Reduce if you hit GPU OOM                 |
| img_size      | 640     | Larger sizes help detect small cracks     |
| optimizer     | SGD     | AdamW also works well for smaller datasets|
| mosaic/mixup  | on      | Strong augmentation for class imbalance   |

Override any value from the CLI:
```bash
python src/train.py --epochs 50 --batch 8 --img-size 416
```

Training outputs (loss curves, PR curves, confusion matrix, weights)
are saved under `models/runs/road_damage_run/`. The best checkpoint is
automatically copied to `models/best.pt`.

---

## 📈 Evaluation

```bash
python src/evaluate.py --weights models/best.pt --data data/road_damage.yaml
```

Outputs precision, recall, mAP@0.5, and mAP@0.5:0.95 overall and per
class, saved to `outputs/eval_reports/eval_metrics.json`, plus a
per-class mAP bar chart.

---

## 🔍 Running Inference (CLI)

```bash
# Single image
python -m src.inference --source path/to/image.jpg --weights models/best.pt

# Folder of images
python -m src.inference --source path/to/folder --weights models/best.pt

# Video
python -m src.inference --source path/to/video.mp4 --weights models/best.pt --video
```

Annotated outputs and a JSON summary report are written to
`outputs/predictions/`.

---

## 🌐 Web App Demo

```bash
streamlit run app/app.py
```

Features:
- **Single Image tab** — upload a photo, view side-by-side original vs. annotated result, per-detection table
- **Video tab** — upload a clip, get a fully annotated output video
- **Batch tab** — upload multiple images at once for a summary table
- Adjustable confidence / IoU thresholds in the sidebar

---

## ⚠️ Severity Estimation Logic

Since raw bounding boxes alone don't convey urgency, `src/utils.py`
combines **damage type** and **relative size** (bounding box area as a
fraction of the image) into a severity score:

- Potholes and alligator cracks are weighted higher (more hazardous)
  than faded lane markings.
- The weighted area-ratio score is bucketed into `Low / Medium / High / Critical`.

This is a heuristic, not a physical measurement — for production use,
it should be calibrated against known road conditions or combined with
depth/LiDAR data.

---

## 📦 Model Export (Deployment)

```python
from src.model import RoadDamageModel
model = RoadDamageModel(weights="models/best.pt")
model.export(format="onnx")          # or "torchscript"
```

ONNX export enables deployment to edge devices (Jetson, mobile) or
serving via ONNX Runtime without a full PyTorch dependency.

---

## 🚧 Limitations & Future Work

- **Severity scoring is heuristic**, not physically calibrated. Future work: fuse with stereo depth or LiDAR.
- **Class imbalance**: potholes are typically rarer than cracks — consider focal loss or targeted oversampling.
- **Weather/lighting robustness**: should be validated across rain, glare, and night-time conditions.
- **Geo-tagging**: pair detections with GPS metadata to plot damage on a map for maintenance crews.
- **Tracking across video frames**: adding a tracker (e.g. ByteTrack) would avoid double-counting the same damage across frames.

---

## 🧪 Testing

```bash
pytest tests/
```

Covers severity-scoring logic and detection-summary aggregation.

---

## 📄 License & Attribution

This project scaffold is provided for educational/research use. If
using the RDD2022 dataset, cite:

> Arya, D. et al. "RDD2022: A multi-national image dataset for
> automatic Road Damage Detection." (2022)

YOLOv8 is provided by Ultralytics under AGPL-3.0 / commercial license
— review their licensing terms before commercial deployment.

---

## 🙋 Contributing

Issues and pull requests are welcome. For major changes, please open
an issue first to discuss what you'd like to change.
