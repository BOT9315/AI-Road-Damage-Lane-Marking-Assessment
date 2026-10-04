import numpy as np
import cv2

class LaneQualityEngine:
    def __init__(self, camera_height=1.2, focal_length=800):
        self.h = float(camera_height)
        self.f = float(focal_length)

    def pixel_to_ground_metric(self, u, v, img_w, img_h, pitch_rad=0.0):
        y_center = v - (img_h / 2.0)
        x_center = u - (img_w / 2.0)

        alpha = np.arctan2(y_center, self.f)
        total_angle = pitch_rad + alpha

        if total_angle <= 0.05:
            return None, None

        ground_y = self.h / np.tan(total_angle)
        ground_x = (x_center * ground_y) / self.f
        return round(float(ground_x), 2), round(float(ground_y), 2)

    def evaluate_marking_quality(self, img_bgr, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = img_bgr.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            return {"contrast_ratio": 0.0, "wear_percentage": 100.0, "serviceability": "Invalid Region"}

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lane_roi = gray[y1:y2, x1:x2]

        pad = max(5, (x2 - x1) // 2)
        if x1 - pad >= 0:
            asphalt_roi = gray[y1:y2, x1-pad:x1]
        elif x2 + pad <= w_img:
            asphalt_roi = gray[y1:y2, x2:x2+pad]
        else:
            asphalt_roi = gray[y1:y2, :]

        mean_lane_lum = float(np.mean(lane_roi))
        mean_asphalt_lum = float(np.mean(asphalt_roi))

        contrast = (mean_lane_lum - mean_asphalt_lum) / (mean_lane_lum + mean_asphalt_lum + 1e-5)
        contrast = max(0.0, contrast)

        faded_pixels = np.sum(lane_roi < (mean_asphalt_lum * 1.15))
        total_pixels = lane_roi.size
        wear_pct = (float(faded_pixels) / float(total_pixels)) * 100.0 if total_pixels > 0 else 100.0

        if wear_pct > 35.0 or contrast < 0.20:
            status = "Critical Maintenance Required"
        elif wear_pct > 15.0:
            status = "Moderate Degradation"
        else:
            status = "Compliant"

        return {
            "contrast_ratio": round(contrast, 3),
            "wear_percentage": round(wear_pct, 1),
            "serviceability": status
        }