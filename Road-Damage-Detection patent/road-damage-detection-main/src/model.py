"""
model.py
========
Thin wrapper around Ultralytics YOLOv8 so the rest of the codebase
(train.py, inference.py, evaluate.py, the Streamlit app) doesn't
depend directly on the third-party API. Swapping architectures later
(e.g. YOLOv9, RT-DETR) only requires changes here.
"""

from ultralytics import YOLO


class RoadDamageModel:
    def __init__(self, weights="yolov8s.pt", device="cuda"):
        self.device = device
        self.model = YOLO(weights)

    def train(self, data_yaml, epochs, img_size, batch_size, project_dir,
              lr0=0.01, optimizer="SGD", patience=20, augment=True,
              mosaic=1.0, mixup=0.1, **kwargs):
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project=project_dir,
            name="road_damage_run",
            lr0=lr0,
            optimizer=optimizer,
            patience=patience,
            augment=augment,
            mosaic=mosaic,
            mixup=mixup,
            **kwargs,
        )
        return results

    def validate(self, data_yaml, img_size=640):
        return self.model.val(data=data_yaml, imgsz=img_size, device=self.device)

    def predict(self, source, conf=0.35, iou=0.45, img_size=640, max_det=100):
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            max_det=max_det,
            device=self.device,
            verbose=False,
        )

    def export(self, format="onnx"):
        return self.model.export(format=format)

    @staticmethod
    def parse_results(results, class_names):
        """Convert Ultralytics Results object -> list of plain dicts."""
        from src.utils import bbox_area_ratio, estimate_severity

        detections = []
        for r in results:
            h, w = r.orig_shape
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                if cls_id not in class_names:
                    continue  # skip classes outside our road-damage taxonomy
                conf = float(box.conf[0])
                area_ratio = bbox_area_ratio((x1, y1, x2, y2), w, h)
                severity = estimate_severity(cls_id, area_ratio)
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_names[cls_id],
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "severity": severity,
                })
        return detections