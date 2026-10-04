# AI Road Damage & Lane Marking Assessment

A computer-vision system that inspects road images and video frames for **pavement damage** (potholes, cracks) and **lane marking wear**, converts pixel measurements into approximate real-world units, scores overall road condition, links the result to a road/contractor record, and exports a PDF audit report.

It combines three things:

1. **YOLOv8** object detection for road damage.
2. A **classical OpenCV pipeline** (morphology, edge and threshold analysis) that finds cracks and measures lane-marking wear.
3. **Inverse Perspective Mapping (IPM)** to estimate distance and surface area in metres / cm² from a single camera, with no depth sensor.

The repository ships two front ends: a **Flask web app** with a REST API (`main.py`) and a **Streamlit dashboard** (`app/app.py`). It also includes a full training and evaluation pipeline for custom YOLOv8 models.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running the apps](#running-the-apps)
- [REST API](#rest-api)
- [Training your own model](#training-your-own-model)
- [Command-line inference](#command-line-inference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Dual-stream damage detection.** YOLOv8 inference first; if it finds nothing, a morphological crack detector (black-hat, Canny and adaptive threshold, merged and filtered) runs as a fallback.
- **Lane marking quality assessment.** Compares the brightness of a road strip against neighbouring asphalt to compute a contrast ratio and a wear %, then assigns a serviceability status (Compliant / Moderate Degradation / Critical Maintenance Required).
- **IPM-based measurement.** Converts detection boxes to estimated distance (m), width / length (m) and surface area (cm²) using camera height, focal length and pitch.
- **Pavement Condition Index (PCI).** A simple 0-100 score derived from detected defects.
- **Location handling.** Reads GPS from photo EXIF, falls back to browser GPS sent by the client, then to a fixed placeholder.
- **Road asset registry.** Maps a road name and section (for example "NH-44, Km 142") to a contractor / authority / warranty record.
- **PDF audit reports.** Generated with ReportLab, including asset identity, PCI, annotated image, lane-quality table and an itemised defect inventory.
- **Live inspection.** Webcam and phone-camera modes, plus a standalone phone-to-laptop frame streamer.
- **Interactive map** (Streamlit): pydeck map of damage spots with severity colours and a 3D orbit view.
- **Training pipeline** for RDD2022-style data (VOC / COCO converters, train, evaluate, annotated inference on images, folders and video).

---

## How it works

### Detection pipeline (`main.py`, `POST /api/v1/inspect`)

```
Uploaded image
   │
   ├─ EXIF GPS extraction ──────────────► location (exif → client GPS → placeholder)
   │
   ├─ Stream 1: YOLOv8 (conf 0.15) ─────► each box counted as a pothole
   │
   ├─ Stream 2: morphological cracks ───► only runs if YOLO found 0 potholes
   │      black-hat + Canny + adaptive threshold → contours → up to 4 boxes
   │      large box → "Pothole (D40)"; wide → "Transverse Crack (D10)";
   │      tall → "Longitudinal Crack (D00)"
   │
   ├─ IPM: bbox → distance / width / length / area
   │
   ├─ Stream 3: lane quality (LaneQualityEngine) on a fixed road strip
   │
   ├─ PCI = clamp(100 − (potholes × 35 + cracks × 15), 0, 100)
   │
   └─ Road registry lookup → JSON response → optional PDF report
```

### Lane marking quality (`src/lane_quality.py`)

The engine converts the image to grayscale, takes the given strip as the "lane" region, and samples a neighbouring strip of asphalt (to the left if there is room, otherwise to the right, otherwise the full row band) for comparison.

| Metric | Definition |
|---|---|
| Contrast ratio | `(mean_lane − mean_asphalt) / (mean_lane + mean_asphalt)`, floored at 0 (range 0-1) |
| Wear % | share of lane-strip pixels darker than 1.15 × mean asphalt luminance |
| Critical Maintenance Required | wear > 35% **or** contrast < 0.20 |
| Moderate Degradation | wear > 15% |
| Compliant | otherwise |

The engine returns `contrast_ratio`, `wear_percentage` and `serviceability`. It also exposes `pixel_to_ground_metric(u, v, img_w, img_h, pitch_rad)`, which maps a pixel to ground-plane X / Y metres. The thresholds are heuristic and have not been validated against any road-marking standard, so treat them as indicative. Note that this contrast ratio is on a 0-1 scale, whereas the API's `lane_quality_report.contrast_ratio` is filled in by rules in `main.py` on a different scale (see [Known limitations](#known-limitations)).

### Inverse Perspective Mapping (`src/metric_ipm.py`)

A pinhole ground-plane model. Defaults: camera height **1.2 m**, focal length **800 px**, nominal pitch **-12°**. The distance is clamped to 1-50 m, and metres-per-pixel at that distance is used to compute width, length and area. Accuracy depends entirely on those camera parameters matching your real setup. An optional `imu_pitch_deg` argument compensates for device tilt.

### Severity (`src/utils.py`)

For CLI inference, severity is `Low / Medium / High / Critical`, based on the box's share of the image area, weighted by class (pothole ×2.0, alligator crack ×1.6, longitudinal / transverse crack ×1.0, marking blur ×0.4).

---

## Repository layout

```
.
├── README.md
├── .gitignore
└── Road-Damage-Detection patent/road-damage-detection-main/    # project root (the parent folder name contains a space)
    ├── main.py                    # Flask server + REST API (port 8000)
    ├── config.yaml                # Central config: classes, training, inference, eval
    ├── requirements.txt
    ├── check_setup.py             # Verifies required packages are importable
    │                              # (no .pt weight files are tracked: *.pt is gitignored)
    │
    ├── app/
    │   ├── app.py                 # Streamlit dashboard (port 8501)
    │   ├── mobile_stream.py       # Standalone Flask server: phone camera → laptop (port 5000)
    │   └── tempCodeRunnerFile.py  # Empty editor leftover, safe to delete
    │
    ├── src/
    │   ├── model.py               # RoadDamageModel: thin wrapper over Ultralytics YOLO
    │   ├── train.py               # Training entry point
    │   ├── evaluate.py            # Precision / recall / mAP report + per-class plot
    │   ├── inference.py           # CLI inference for image, folder or video
    │   ├── dataset.py             # VOC / COCO → YOLO conversion and train/val/test split
    │   ├── lane_quality.py        # LaneQualityEngine
    │   ├── metric_ipm.py          # DynamicIPMEngine
    │   ├── road_registry.py       # Road / contractor asset lookup
    │   ├── report_generator.py    # ReportLab PDF builder
    │   └── utils.py               # Config loader, severity, drawing, summaries
    │
    ├── data/
    │   ├── download_dataset.py    # Prints RDD2022 instructions and validates layout
    │   └── road_damage.yaml       # YOLO dataset descriptor (6 classes)
    │
    ├── models/rdd_exp/args.yaml   # Args of an earlier 5-epoch yolov8n training run
    ├── static/index.html          # Web UI for main.py (Tailwind + Leaflet via CDN)
    ├── test_images/sample.jpg
    └── tests/test_utils.py
```

### Damage classes (RDD2022 taxonomy)

| ID | Class | Meaning |
|---|---|---|
| 0 | `D00_longitudinal_crack` | Cracks along the direction of travel |
| 1 | `D10_transverse_crack` | Cracks across the road |
| 2 | `D20_alligator_crack` | Interconnected fatigue cracking |
| 3 | `D40_pothole` | Pothole / rutting / bump |
| 4 | `D43_crosswalk_blur` | Faded crosswalk |
| 5 | `D44_whiteline_blur` | Faded white lane line |

---

## Installation

**Prerequisites:** Python 3.9-3.11 is a safe choice for the pinned versions. A GPU is optional; CPU is enough for inference and the default `config.yaml` uses `device: "cpu"`.

```bash
git clone https://github.com/AayushSingh34/AI-Road-Damage-Lane-Marking-Assessment.git
cd "AI-Road-Damage-Lane-Marking-Assessment/Road-Damage-Detection patent/road-damage-detection-main"

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Packages missing from `requirements.txt`

The code imports several packages that `requirements.txt` does not list. Install them too:

```bash
pip install flask reportlab pytest
pip install "qrcode[pil]"         # optional: only used by the Streamlit app
```

| Package | Needed by |
|---|---|
| `flask` | `main.py`, `app/mobile_stream.py` |
| `reportlab` | `src/report_generator.py` and the Streamlit PDF export |
| `pytest` | running the tests |
| `qrcode` | optional, guarded import in `app/app.py` |

`fpdf2` is listed in `requirements.txt` but is not used anywhere in the code, and `pydeck` ships as a Streamlit dependency.

Verify the environment:

```bash
python check_setup.py
```

---

## Running the apps

All commands run from `Road-Damage-Detection patent/road-damage-detection-main/` (the project root).

### Option A: Flask web app (full pipeline, API and PDF reports)

```bash
python main.py
```

Open <http://localhost:8000>. The UI lets you pick **Highway** or **Local road**, enter a road name and section, upload an image or use the live camera, then view detections, the lane-quality assessment, the map and the PDF export.

Model loading: the app tries `models/best.pt` first and falls back to `yolov8n.pt`, which Ultralytics downloads automatically on first use (so the first run needs internet). See [Known limitations](#known-limitations) for why that matters.

### Option B: Streamlit dashboard

```bash
streamlit run app/app.py
```

Opens at <http://localhost:8501>.

- **Sidebar:** choose `yolov8n.pt` or `yolov8s.pt`, input source (Webcam, Upload Video / Image, Mobile Camera), confidence threshold (0.10-1.00, default 0.40), and a QR code to open the app from a phone on the same Wi-Fi.
- **Detection Feed:** live webcam with on-demand scan, phone snap-and-scan, or file upload, with an itemised anomaly table and a downloadable PDF.
- **Live Damage Map:** pydeck map with severity-coloured spots and a 3D orbit view. The spots are seeded sample points around Chandigarh / Mohali.
- **Incident Logs** and **About & Strategic Impact** tabs.

The QR code is rendered by the public `api.qrserver.com` service, so your local URL is sent to that service.

### Option C: Phone camera streamer (standalone)

```bash
python app/mobile_stream.py
```

Starts a server on port 5000. Open `http://<laptop-ip>:5000` on a phone on the same network, tap **Start Broadcasting**, and the rear camera sends JPEG frames (about 10 FPS) to `/upload_frame`. Frames are held in memory and exposed through `get_current_frame()` for other code to consume. This module is not currently imported by the Streamlit app. Most mobile browsers only allow camera access over HTTPS (or localhost), so plain `http://<ip>` may be blocked.

---

## REST API

Served by `main.py` on port 8000.

### `POST /api/v1/inspect`

Multipart form data:

| Field | Type | Description |
|---|---|---|
| `file` | file, required | Road image |
| `road_type` | text | `highway` (default) or any other value for a local road |
| `road_name` | text | For example `NH-44` |
| `road_section` | text | Section / chainage keywords, for example `Km 142 Ramban` |
| `lat`, `lng` | float | Client GPS, used if the image has no EXIF GPS |

Response (abridged):

```json
{
  "success": true,
  "processing_time": 0.412,
  "image_metadata": { "width": 1280, "height": 720 },
  "summary": { "potholes": 1, "cracks": 2, "lane_defects": 1, "pci_index": 35 },
  "location": { "lat": 30.7, "lng": 76.7, "source": "exif_hardware_gps" },
  "road_asset": { "road_id": "...", "name": "...", "contractor": "...", "authority": "..." },
  "structural_detections": [
    { "type": "structural", "class_name": "Pothole (D40)", "confidence": 0.81,
      "bbox": [120.0, 340.5, 260.2, 410.0],
      "metrics": { "estimated_distance_m": 6.4, "width_m": 0.31, "length_m": 0.16, "surface_area_cm2": 496.0 } }
  ],
  "lane_evaluations": [ { "class_name": "Road Marking Segment (D44)", "bbox": [], "metrics": {} } ],
  "lane_quality_report": {
    "wear_percentage": 80.0, "continuity_percentage": 20.0, "contrast_ratio": 1.8,
    "serviceability": "Critical (Markings Broken / Degraded)",
    "recommended_action": "Priority Thermoplastic Re-striping Required"
  }
}
```

`location.source` is one of `exif_hardware_gps`, `client_realtime_gps`, or `simulated_station_gps`.

### `POST /api/v1/inspect-video-frame`

Lightweight per-frame check for live camera mode. Send the same road-context fields plus `frame` (image file). Returns `condition`, `pci_index`, `wear_percentage`, `continuity_percentage`, `contrast_ratio`, `road_asset` and `defects_in_frame` (potholes / cracks). This endpoint uses only the OpenCV pipeline, not YOLO.

### `POST /api/v1/export-report`

Form fields: `data` (the JSON from `/inspect`, as a string) and optional `image` (annotated image). Returns a PDF download named `Road_Audit_Report_<timestamp>.pdf`.

### `GET /`

Serves `static/index.html`.

Quick test:

```bash
curl -X POST http://localhost:8000/api/v1/inspect \
  -F "file=@test_images/sample.jpg" \
  -F "road_type=highway" -F "road_name=NH-44" -F "road_section=Ludhiana"
```

---

## Training your own model

The pipeline is built for RDD2022-style data but accepts anything convertible to YOLO format. Run the `src` modules from the project root with `python -m` (they use `from src...` imports, so `python src/train.py` fails with `ModuleNotFoundError`).

**1. Get data.** `data/download_dataset.py` does not download anything; it prints instructions and checks your layout. Get RDD2022 from <https://github.com/sekilab/RoadDamageDetector>, then arrange:

```
data/raw/images/*.jpg
data/raw/annotations/*.xml
```

```bash
python data/download_dataset.py        # validates the layout
```

**2. Convert to YOLO format and split** (default 75% train / 15% val / 10% test, seed 42):

```bash
python -m src.dataset --format voc \
    --images data/raw/images --annotations data/raw/annotations --out data/processed

# or for COCO JSON
python -m src.dataset --format coco \
    --images data/raw/images --annotations data/raw/annotations.json --out data/processed
```

Labels are mapped via `D00/D01→0`, `D10/D11→1`, `D20→2`, `D40→3`, `D43→4`, `D44→5`; other classes are skipped. Images without labels get empty label files (treated as background).

**3. Train:**

```bash
python -m src.train --config config.yaml
python -m src.train --epochs 50 --batch 8 --img-size 640      # overrides
```

Run output goes to `models/runs/road_damage_run/`, and the best checkpoint is copied to `models/best.pt`, which `main.py` loads automatically.

**4. Evaluate:**

```bash
python -m src.evaluate --weights models/best.pt --data data/road_damage.yaml
```

Writes `outputs/eval_reports/eval_metrics.json` (overall and per-class precision, recall, mAP50, mAP50-95) and `per_class_map50.png`.

> `models/rdd_exp/args.yaml` records an earlier experiment (yolov8n, 5 epochs, `data/rdd2020.yaml`). It is a log of that run, not the current training configuration.

---

## Command-line inference

```bash
python -m src.inference --source test_images/sample.jpg --weights models/best.pt   # one image
python -m src.inference --source path/to/folder --weights models/best.pt           # a folder
python -m src.inference --source path/to/video.mp4 --weights models/best.pt --video
```

Output goes to `outputs/predictions/` (override with `--out`): annotated images or video (`annotated_*`) and a `report.json` with counts by class and severity.

---

## Configuration

`config.yaml` is the single source of truth for the CLI tools:

| Section | Key settings |
|---|---|
| `project` | `seed: 42`, `device: "cpu"` (`cuda`, `cpu`, or `mps`) |
| `dataset` | split ratios, `img_size: 640`, the 6-class map |
| `model` | `variant: yolov8s.pt`, `num_classes: 6`, export formats |
| `train` | 100 epochs, batch 16, SGD, `lr0: 0.01`, early-stopping patience 20, mosaic 1.0, mixup 0.1 |
| `inference` | `conf_threshold: 0.35`, `iou_threshold: 0.45`, `max_detections: 100`, `weights_path: yolov8s.pt` |
| `evaluate` | metrics list, `report_dir: outputs/eval_reports` |

Settings **not** driven by `config.yaml`: the Flask app's camera parameters (`camera_height=1.2`, `focal_length=800`), its YOLO confidence (0.15), and the Streamlit sidebar values are hard-coded in `main.py` and `app/app.py`. Edit them there to match your camera.

---

## Testing

```bash
pytest tests/
```

`tests/test_utils.py` covers `bbox_area_ratio`, `estimate_severity` and `summarize_detections`. The detection engines, API and report generator have no automated tests yet.

---

## Known limitations

These are worth knowing before relying on the output.

- **No road-damage model is included.** This repo tracks no `.pt` files (`*.pt` is gitignored), so there is no trained `models/best.pt`. Until you train one, `main.py` falls back to the generic COCO-pretrained `yolov8n.pt` (auto-downloaded by Ultralytics), and its detections will not be meaningful road-damage results. Likewise, the `yolov8n.pt` / `yolov8s.pt` choices in the Streamlit sidebar and `weights_path: yolov8s.pt` in `config.yaml` are generic weights fetched on demand.
- **`main.py` labels every YOLO detection "Pothole (D40)"** regardless of the predicted class, and counts it as a pothole in the PCI.
- **Some values are fixed, not measured.** Heuristic detections report confidence 0.88, the lane-assessment entry reports 0.92, and the PDF defect table shows 91% for lane rows. In `/api/v1/inspect`, the lane engine's measured wear is only a starting point: `main.py` then sets continuity, contrast, serviceability and recommended action from fixed rules based on the pothole / crack count (and floors the wear value when defects exist), so the reported lane metrics are largely rule-driven rather than purely measured. The engine's own output (the table above) is only used unmodified if you call `LaneQualityEngine` directly.
- **Placeholder location.** With no EXIF or client GPS, the API returns fixed coordinates (28.6139, 77.2090) labelled `simulated_station_gps`.
- **Road registry data is illustrative.** `src/road_registry.py` contains a small hard-coded set of NHAI package records. For any road not in it, contractor, dates, warranty and pavement mix are synthesised from a hash of the road name. Do not treat these as authoritative asset or liability records without replacing them with real data.
- **IPM accuracy** depends on correct camera height, focal length and pitch. Defaults suit a ~1.2 m mounted camera, not an arbitrary photo.
- **Two PCI formulas.** The image endpoint and the video-frame endpoint score PCI differently, so values are not directly comparable.
- **Lane strip is fixed.** Lane quality is computed on a fixed region of the lower image, not on detected lane lines.
- **`requirements.txt` is incomplete** (see [Installation](#installation)).
- **Network use.** The web UI loads Tailwind and Leaflet from CDNs and may call `ipapi.co` for an IP-location fallback, so it needs internet access.
- **Development server.** `main.py` runs Flask with `debug=True` on `0.0.0.0`. Do not expose it publicly as-is.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from `Road-Damage-Detection patent/road-damage-detection-main/` and use `python -m src.<module>` |
| `cd` fails with "too many arguments" | The parent folder name contains a space; wrap the path in quotes |
| `No module named 'flask'` / `'reportlab'` | `pip install flask reportlab` |
| Streamlit PDF export unavailable | Install `reportlab` |
| No QR code in the sidebar | Install `qrcode[pil]`; the QR image also needs internet |
| Phone camera won't start | Browsers require HTTPS or localhost for camera access; try a tunnel or local HTTPS |
| Training can't find the dataset | Check `path:` in `data/road_damage.yaml` points at your `data/processed` folder |
| Slow inference | Set `device: "cuda"` in `config.yaml` if you have a GPU, or use `yolov8n.pt` |

---

## License

No license file is included in this repository. Until one is added, all rights are reserved by the author. Add a `LICENSE` file (for example MIT) to permit reuse.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [RDD2022 / Road Damage Detector](https://github.com/sekilab/RoadDamageDetector) dataset and taxonomy
- OpenCV, Flask, Streamlit, pydeck, Leaflet and ReportLab
