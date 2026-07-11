"""Create a live state from camera, GPS, LD2451 and optional vision models."""

from __future__ import annotations

import time


def build_real_sensor_state(camera_available: bool, radar_reader, gps_reader,
                            frame=None, frame_capture_monotonic_ns: int = 0,
                            camera_frame_id: int = -1,
                            vision_adapter=None, fusion_engine=None,
                            recorder=None, sync_thresholds=None) -> dict:
    sync_thresholds = sync_thresholds or {"radar_max_delta_ms": 100.0, "gps_max_delta_ms": 1000.0}
    radar_start_ns = time.monotonic_ns()
    radar = radar_reader.read_once()
    radar_end_ns = time.monotonic_ns()
    gps_start_ns = time.monotonic_ns()
    gps = gps_reader.read_once()
    gps_end_ns = time.monotonic_ns()
    camera_ns = frame_capture_monotonic_ns or radar_start_ns
    radar_delta_ms = abs(radar_end_ns - camera_ns) / 1_000_000.0
    gps_delta_ms = abs(gps_end_ns - camera_ns) / 1_000_000.0
    radar_fresh = radar_delta_ms <= float(sync_thresholds["radar_max_delta_ms"])
    gps_fresh = gps_delta_ms <= float(sync_thresholds["gps_max_delta_ms"])
    vision = None
    fusion = None
    inference_ms = 0.0
    vision_start_ns = 0
    vision_finish_ns = 0
    if vision_adapter is not None and frame is not None:
        vision_start_ns = time.monotonic_ns()
        vision = vision_adapter.process(frame)
        vision_finish_ns = time.monotonic_ns()
        inference_ms = (vision_finish_ns - vision_start_ns) / 1_000_000.0
        raw_result = vision_adapter.get_latest_vision_result()
        if raw_result is not None and fusion_engine is not None and radar_fresh:
            fusion = fusion_engine.fuse_vision_result(raw_result, radar)
    timestamps = {
        "frame_capture_monotonic_ns": camera_ns,
        "radar_read_start_monotonic_ns": radar_start_ns,
        "radar_read_end_monotonic_ns": radar_end_ns,
        "gps_read_start_monotonic_ns": gps_start_ns,
        "gps_read_end_monotonic_ns": gps_end_ns,
        "vision_start_monotonic_ns": vision_start_ns,
        "vision_finish_monotonic_ns": vision_finish_ns,
        "radar_delta_ms": radar_delta_ms,
        "gps_delta_ms": gps_delta_ms,
        "vision_latency_ms": inference_ms,
    }
    if recorder is not None:
        recorder.write(frame, radar, gps, vision, fusion, inference_ms, timestamps,
                       camera_frame_id=camera_frame_id,
                       radar_valid=bool(radar.valid) and radar_fresh,
                       gps_valid=bool(gps.valid) and gps_fresh)

    targets = [{"target_id": int(t.target_id), "distance_m": round(float(t.distance_m), 2),
                "relative_speed_mps": round(float(t.relative_speed_mps), 2),
                "angle_deg": round(float(t.angle_deg), 1),
                "confidence": round(float(t.confidence), 3)} for t in radar.targets]
    objects = [] if vision is None else [
        {"class_name": o.class_name, "risk_class": o.risk_class,
         "confidence": round(float(o.confidence), 3), "bbox": list(o.bbox),
         "in_drivable_area": o.in_drivable_area}
        for o in vision.objects]
    radar_connected = getattr(radar_reader, "_serial", None) is not None
    gps_connected = getattr(gps_reader, "_serial", None) is not None
    vision_valid = bool(vision is not None and vision.valid)
    return {
        "timestamp": time.time(), "risk_score": 0.0, "risk_level": 0,
        "risk_label": "风险模型未启用",
        "risk_items": {"obs": 0.0, "dist": 0.0, "pose": 0.0, "speed": 0.0},
        "weights": {"obs": 0.0, "dist": 0.0, "pose": 0.0, "speed": 0.0},
        "sensors": {"camera": bool(camera_available),
                    "vision": "real" if vision_valid else "off",
                    "radar": "real" if radar_connected else "off",
                    "gps": "real" if gps_connected else "off"},
        "mode": "real-recording" if recorder is not None else "real-sensors",
        "message": "真实传感器 + 视觉特征；IMU不使用，风险模型未启用",
        "radar_data": {"connected": radar_connected, "valid": bool(radar.valid) and radar_fresh,
                       "target_count": len(targets),
                       "nearest_distance_m": round(float(radar.nearest_distance_m), 2),
                       "min_ttc_s": round(float(radar.min_ttc), 2), "targets": targets},
        "gps_data": {"connected": gps_connected, "valid": bool(gps.valid) and gps_fresh,
                     "speed_kmh": round(float(gps.speed_kmh), 2),
                     "latitude": round(float(gps.latitude), 7),
                     "longitude": round(float(gps.longitude), 7),
                     "fix_quality": int(gps.fix_quality)},
        "fusion_data": {"valid": fusion is not None,
                        "vision_radar_count": fusion.n_vision_radar if fusion else 0,
                        "vision_only_count": fusion.n_vision_only if fusion else 0,
                        "radar_only_count": fusion.n_radar_only if fusion else 0},
        "performance": {"vision_infer_ms": round(inference_ms, 2)},
        "timestamps": timestamps,
        "vision_details": {"valid": vision_valid, "object_count": len(objects),
                           "person_count": vision.person_count if vision else 0,
                           "vehicle_count": vision.vehicle_count if vision else 0,
                           "max_confidence": float(vision.max_confidence) if vision else 0.0,
                           "drivable_area_ratio": float(vision.drivable_area_ratio) if vision else 0.0,
                           "objects": objects,
                           "frame_size": {"width": frame.shape[1] if frame is not None else 640,
                                          "height": frame.shape[0] if frame is not None else 480}},
    }
