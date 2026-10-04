import numpy as np

class DynamicIPMEngine:
    def __init__(self, camera_height_m: float = 1.2, focal_length_px: float = 800.0, nominal_pitch_deg: float = -12.0):
        self.h = camera_height_m
        self.f = focal_length_px
        self.theta_nominal = np.radians(nominal_pitch_deg)

    def estimate_physical_metrics(self, bbox: list, img_height: int, img_width: int, imu_pitch_deg: float = 0.0):
        """
        Transforms pixel bounding box coordinates to real-world ground-plane metrics (meters / sq cm)
        using inverse perspective mapping compensated by camera pitch.
        """
        x1, y1, x2, y2 = bbox
        theta_actual = self.theta_nominal + np.radians(imu_pitch_deg)

        # Contact line on the ground plane relative to optical center
        v_ground = max(1.0, y2 - (img_height / 2.0))

        # Pinhole ground-plane projection
        denom = np.sin(abs(theta_actual)) + (v_ground / self.f) * np.cos(abs(theta_actual))
        distance_z = abs(self.h / max(0.05, denom))
        distance_z = max(1.0, min(distance_z, 50.0))  # Realistic clamp: 1m to 50m

        # Ground sampling distance (meters per pixel at this distance)
        gsd = distance_z / self.f

        width_m = round(float(abs(x2 - x1) * gsd), 3)
        length_m = round(float(abs(y2 - y1) * gsd), 3)
        area_cm2 = round(float(width_m * length_m * 10000), 1)

        return {
            "estimated_distance_m": round(float(distance_z), 2),
            "width_m": width_m,
            "length_m": length_m,
            "surface_area_cm2": area_cm2
        }