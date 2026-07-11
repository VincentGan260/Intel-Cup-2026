"""Expose real GPS and LD2451 values without loading any model."""

from __future__ import annotations

import time


def build_real_sensor_state(camera_available: bool, radar_reader, gps_reader) -> dict:
    radar = radar_reader.read_once()
    gps = gps_reader.read_once()
    targets = [{
        "target_id": int(t.target_id),
        "distance_m": round(float(t.distance_m), 2),
        "relative_speed_mps": round(float(t.relative_speed_mps), 2),
        "angle_deg": round(float(t.angle_deg), 1),
        "confidence": round(float(t.confidence), 3),
    } for t in radar.targets]
    radar_connected = getattr(radar_reader, "_serial", None) is not None
    gps_connected = getattr(gps_reader, "_serial", None) is not None
    return {
        "timestamp": time.time(), "risk_score": 0.0, "risk_level": 0,
        "risk_label": "模型未启用",
        "risk_items": {"obs": 0.0, "dist": 0.0, "pose": 0.0, "speed": 0.0},
        "weights": {"obs": 0.0, "dist": 0.0, "pose": 0.0, "speed": 0.0},
        "sensors": {"camera": bool(camera_available), "vision": "off",
                    "radar": "real" if radar_connected else "off",
                    "gps": "real" if gps_connected else "off"},
        "mode": "real-sensors",
        "message": "真实传感器显示模式（模型未启用，IMU不使用）",
        "radar_data": {"connected": radar_connected, "valid": bool(radar.valid),
                       "target_count": len(targets),
                       "nearest_distance_m": round(float(radar.nearest_distance_m), 2),
                       "min_ttc_s": round(float(radar.min_ttc), 2), "targets": targets},
        "gps_data": {"connected": gps_connected, "valid": bool(gps.valid),
                     "speed_kmh": round(float(gps.speed_kmh), 2),
                     "latitude": round(float(gps.latitude), 7),
                     "longitude": round(float(gps.longitude), 7),
                     "fix_quality": int(gps.fix_quality), "satellites": int(gps.satellites)},
        "performance": {"vision_infer_ms": 0.0},
        "vision_details": {"valid": False, "object_count": 0, "person_count": 0,
                           "vehicle_count": 0, "max_confidence": 0.0,
                           "drivable_area_ratio": 0.0, "max_visual_risk": 0.0,
                           "objects": [], "frame_size": {"width": 640, "height": 480}},
    }
