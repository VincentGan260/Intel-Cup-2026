"""Adapt XGBoost runtime data to the existing Dashboard/cloud state contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _module_share(contributions: dict, name: str) -> float | None:
    module = contributions.get(name)
    if not isinstance(module, dict):
        return None
    try:
        return max(0.0, min(1.0, float(module.get("importance_pct", 0.0)) / 100.0))
    except (TypeError, ValueError):
        return None


def _compat_module_level(
    overall_level: int | None,
    contributions: dict,
    name: str,
    *,
    sensor_valid: bool,
) -> int | None:
    """Keep the old level type without pretending XGBoost has old rule levels."""
    if not sensor_valid:
        return None
    module = contributions.get(name)
    if overall_level not in (1, 2) or not isinstance(module, dict):
        return 0
    return overall_level if module.get("direction") == "raises" else 0


def _vision_objects(vision: Any) -> list[dict]:
    objects = []
    for item in list(getattr(vision, "objects", []) or []):
        objects.append({
            "class_name": str(getattr(item, "class_name", "")),
            "risk_class": str(getattr(item, "risk_class", "")),
            "confidence": float(getattr(item, "confidence", 0.0) or 0.0),
            "bbox": [float(value) for value in getattr(item, "bbox", (0, 0, 0, 0))],
            "in_drivable_area": getattr(item, "in_drivable_area", None),
            "risk": float(getattr(item, "visual_risk", 0.0) or 0.0),
        })
    return objects


def build_dashboard_state(
    *,
    timestamp: float,
    status: str,
    prediction: dict | None,
    prediction_error: str,
    features: dict,
    modules: dict,
    feature_window: dict,
    sensor_states: dict,
    runtime: dict,
    motor_control: bool,
    motor: dict,
    radar: Any,
    gps: Any,
    imu: Any,
    vision: Any,
    camera_available: bool,
    frame_width: int,
    frame_height: int,
    started_at: float,
) -> dict:
    """Return one state that serves XGBoost diagnostics and the old UI/cloud."""
    contributions = (
        prediction.get("module_contributions", {})
        if isinstance(prediction, dict) else {}
    )
    level = prediction.get("level") if isinstance(prediction, dict) else None
    level = int(level) if level in (0, 1, 2) else None
    risk_score = (
        float(prediction["risk_score"])
        if isinstance(prediction, dict) and prediction.get("risk_score") is not None
        else None
    )
    confidence = (
        float(prediction["confidence"])
        if isinstance(prediction, dict) and prediction.get("confidence") is not None
        else None
    )

    radar_valid = bool(getattr(radar, "valid", False))
    gps_valid = bool(getattr(gps, "valid", False))
    imu_valid = bool(getattr(imu, "valid", False))
    vision_valid = bool(getattr(vision, "valid", False))
    radar_targets = list(getattr(radar, "targets", []) or [])
    vision_objects = _vision_objects(vision)

    radar_level = _compat_module_level(
        level, contributions, "radar", sensor_valid=radar_valid
    )
    vision_level = _compat_module_level(
        level, contributions, "vision", sensor_valid=vision_valid
    )
    ranked_modules = sorted(
        (
            (name, _module_share(contributions, name) or 0.0)
            for name in ("radar", "vision", "imu", "gps")
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant = ranked_modules[0][0] if prediction and ranked_modules else "none"
    warning_reason = (
        f"xgboost:{prediction.get('label', level)};"
        f"confidence={confidence:.3f};dominant={dominant}"
        if prediction is not None and confidence is not None
        else f"xgboost:{status}"
    )
    risk_status = (
        "unknown" if prediction is None
        else "degraded" if status == "low_confidence"
        else "normal"
    )
    system_status = (
        "unknown" if prediction is None
        else "degraded" if status != "active"
        else "normal"
    )

    motor_status = (
        "disabled" if not motor_control
        else "unavailable" if motor.get("faulted") or not motor.get("connected")
        else "mock" if motor.get("mode") == "mock"
        else "active"
    )
    radar_status = sensor_states.get("radar", {}).get("status", "unknown")
    if radar_valid:
        radar_status = "tracking" if radar_targets else "no_target"
    gps_status = sensor_states.get("gps", {}).get("status", "unknown")
    if not gps_valid and gps_status in {"invalid", "active"}:
        gps_status = "no_fix"

    module_scores = {
        name: _module_share(contributions, name)
        for name in ("radar", "vision", "imu", "gps")
    }
    state = {
        "schema_version": "xgb-risk-v2",
        "timestamp": float(timestamp),
        "status": status,
        "decision_engine": "xgboost-only",
        "old_rules_loaded": False,
        "motor_control": bool(motor_control),
        "motor": motor,
        "prediction": prediction,
        "prediction_error": prediction_error,
        "features": features,
        "modules": modules,
        "feature_window": feature_window,
        "risk_score": risk_score,
        "risk_level": level,
        "risk_label": prediction.get("label") if prediction else "等待模型",
        "risk_status": risk_status,
        "system_status": system_status,
        "warning_reason": warning_reason,
        "radar_level": radar_level,
        "vision_level": vision_level,
        "radar_score": module_scores["radar"],
        "vision_score": module_scores["vision"],
        "imu_score": module_scores["imu"],
        "gps_score": module_scores["gps"],
        "risk_items": {
            "obs": module_scores["vision"] or 0.0,
            "dist": module_scores["radar"] or 0.0,
            "pose": module_scores["imu"] or 0.0,
            "speed": module_scores["gps"] or 0.0,
        },
        "weights": {
            "obs": module_scores["vision"] or 0.0,
            "dist": module_scores["radar"] or 0.0,
            "pose": module_scores["imu"] or 0.0,
            "speed": module_scores["gps"] or 0.0,
        },
        "sensors": {
            "camera": "real" if camera_available else "off",
            "vision": "real" if vision_valid else "invalid",
            "radar": "real" if radar_valid else "invalid",
            "imu": "real" if imu_valid else "invalid",
            "gps": "real" if gps_valid else "invalid",
            "motor": (
                "real" if motor_status == "active"
                else "mock" if motor_status == "mock"
                else "off"
            ),
        },
        "hardware_status": {
            "camera": {
                "status": "active" if camera_available else "unavailable",
                "reason": "camera frame producer",
            },
            "vision": {
                **sensor_states.get("vision", {}),
                "status": sensor_states.get("vision", {}).get("status", "disabled"),
            },
            "radar": {
                **sensor_states.get("radar", {}),
                "status": radar_status,
            },
            "gps": {
                **sensor_states.get("gps", {}),
                "status": gps_status,
            },
            "imu": {
                **sensor_states.get("imu", {}),
                "status": sensor_states.get("imu", {}).get("status", "unknown"),
            },
            "motor": {
                "status": motor_status,
                "reason": motor.get("gate_reason", ""),
            },
        },
        "radar_data": {
            "valid": radar_valid,
            "target_count": len(radar_targets),
            "nearest_distance_m": float(
                getattr(radar, "nearest_distance_m", -1.0) or 0.0
            ),
            "min_ttc_s": float(getattr(radar, "min_ttc", -1.0) or 0.0),
            "age_ms": sensor_states.get("radar", {}).get("age_ms"),
            "status": radar_status,
        },
        "gps_data": {
            "valid": gps_valid,
            "latitude": float(getattr(gps, "latitude", 0.0) or 0.0),
            "longitude": float(getattr(gps, "longitude", 0.0) or 0.0),
            "speed_kmh": max(0.0, float(getattr(gps, "speed_kmh", 0.0) or 0.0)),
            "fix_quality": int(getattr(gps, "fix_quality", 0) or 0),
            "satellites": int(getattr(gps, "satellites", 0) or 0),
            "status": gps_status,
        },
        "imu_data": {
            "valid": imu_valid,
            "roll": float(getattr(imu, "body_roll", None)
                          if getattr(imu, "body_roll", None) is not None
                          else getattr(imu, "roll", 0.0) or 0.0),
            "pitch": float(getattr(imu, "body_pitch", None)
                           if getattr(imu, "body_pitch", None) is not None
                           else getattr(imu, "pitch", 0.0) or 0.0),
            "yaw": float(getattr(imu, "yaw", 0.0) or 0.0),
            "acc_x": float(getattr(imu, "acc_x", 0.0) or 0.0),
            "acc_y": float(getattr(imu, "acc_y", 0.0) or 0.0),
            "acc_z": float(getattr(imu, "acc_z", 0.0) or 0.0),
            "gyro_x": float(getattr(imu, "gyro_x", 0.0) or 0.0),
            "gyro_y": float(getattr(imu, "gyro_y", 0.0) or 0.0),
            "gyro_z": float(getattr(imu, "gyro_z", 0.0) or 0.0),
            "brake_score": float(getattr(imu, "brake_score", 0.0) or 0.0),
            "bump_score": float(getattr(imu, "bump_score", 0.0) or 0.0),
            "tilt_score": float(getattr(imu, "tilt_score", 0.0) or 0.0),
        },
        "vision_details": {
            "valid": vision_valid,
            "object_count": len(vision_objects),
            "person_count": int(getattr(vision, "person_count", 0) or 0),
            "vehicle_count": int(getattr(vision, "vehicle_count", 0) or 0),
            "drivable_area_ratio": (
                float(getattr(vision, "drivable_area_ratio", 0.0))
                if vision_valid else None
            ),
            "objects": vision_objects,
            "frame_size": {"width": int(frame_width), "height": int(frame_height)},
        },
        "mode": "xgboost-real",
        "message": (
            f"XGBoost {prediction.get('label')} · 主导贡献 {dominant.upper()}"
            if prediction else "XGBoost 特征窗口预热中"
        ),
        "runtime": {
            **runtime,
            "backend_uptime_sec": round(max(0.0, timestamp - started_at), 2),
            "last_update_time": datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "actual_sample_hz": runtime.get("state_hz", 0.0),
            "vision_infer_ms": float(
                getattr(vision, "pipeline_inference_ms", 0.0) or 0.0
            ),
        },
    }
    return state
