"""Adapt XGBoost runtime data to the existing Dashboard/cloud state contract."""

from __future__ import annotations

import time
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
    effective_decision: dict | None = None,
    decision_source: str = "xgboost",
    missing_sensors: tuple[str, ...] = (),
    degraded_reasons: tuple[str, ...] = (),
) -> dict:
    """Return one state that serves XGBoost diagnostics and the old UI/cloud."""
    contributions = (
        prediction.get("module_contributions", {})
        if isinstance(prediction, dict) else {}
    )
    decision = (
        effective_decision
        if isinstance(effective_decision, dict)
        else prediction
    )
    level = decision.get("level") if isinstance(decision, dict) else None
    level = int(level) if level in (0, 1, 2) else None
    risk_score = (
        float(decision["risk_score"])
        if isinstance(decision, dict) and decision.get("risk_score") is not None
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
    raw_min_ttc = getattr(radar, "min_ttc", -1.0)
    raw_nearest_distance = getattr(radar, "nearest_distance_m", -1.0)
    radar_min_ttc = float(
        raw_min_ttc if raw_min_ttc is not None else -1.0
    )
    radar_nearest_distance = float(
        raw_nearest_distance
        if raw_nearest_distance is not None else -1.0
    )
    serialized_radar_targets = [
        {
            "target_id": int(getattr(item, "target_id", 0) or 0),
            "distance_m": float(getattr(item, "distance_m", 0.0) or 0.0),
            "relative_speed_mps": float(
                getattr(item, "relative_speed_mps", 0.0) or 0.0
            ),
            "angle_deg": float(getattr(item, "angle_deg", 0.0) or 0.0),
            "confidence": float(getattr(item, "confidence", 0.0) or 0.0),
        }
        for item in radar_targets
    ]
    vision_objects = _vision_objects(vision)

    fallback_scores = (
        decision.get("modality_scores", {})
        if decision_source != "xgboost" and isinstance(decision, dict)
        else {}
    )
    fallback_levels = (
        decision.get("modality_levels", {})
        if decision_source != "xgboost" and isinstance(decision, dict)
        else {}
    )
    if decision_source == "xgboost":
        radar_level = _compat_module_level(
            level, contributions, "radar", sensor_valid=radar_valid
        )
        vision_level = _compat_module_level(
            level, contributions, "vision", sensor_valid=vision_valid
        )
        imu_level = _compat_module_level(
            level, contributions, "imu", sensor_valid=imu_valid
        )
    else:
        radar_level = fallback_levels.get("radar") if radar_valid else None
        vision_level = fallback_levels.get("vision") if vision_valid else None
        imu_level = fallback_levels.get("imu") if imu_valid else None
    ranked_modules = sorted(
        (
            (name, _module_share(contributions, name) or 0.0)
            for name in ("radar", "vision", "imu", "gps")
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    dominant = ranked_modules[0][0] if prediction and ranked_modules else "none"
    if decision_source == "xgboost":
        warning_reason = (
            f"xgboost:{prediction.get('label', level)};"
            f"confidence={confidence:.3f};dominant={dominant}"
            if prediction is not None and confidence is not None
            else f"xgboost:{status}"
        )
    else:
        warning_reason = (
            f"{decision_source}:{decision.get('reason', 'fallback')}"
            if isinstance(decision, dict)
            else f"{decision_source}:unavailable"
        )
    is_degraded = bool(degraded_reasons or missing_sensors)
    risk_status = (
        "unknown" if decision is None
        else "degraded" if is_degraded or status != "active"
        else "normal"
    )
    system_status = (
        "unknown" if decision is None
        else "degraded" if is_degraded or status != "active"
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

    sensor_validity = {
        "radar": radar_valid,
        "vision": vision_valid,
        "imu": imu_valid,
        "gps": gps_valid,
    }
    if decision_source == "xgboost":
        module_scores = {
            name: (
                _module_share(contributions, name)
                if sensor_validity[name] else None
            )
            for name in ("radar", "vision", "imu", "gps")
        }
    else:
        module_scores = {
            name: (
                fallback_scores.get(name)
                if sensor_validity[name] else None
            )
            for name in ("radar", "vision", "imu")
        }
        module_scores["gps"] = None
    now_monotonic_ns = time.monotonic_ns()
    evidence_sources = (
        list(decision.get("evidence_sources", ()))
        if isinstance(decision, dict) and decision_source != "xgboost"
        else [
            name for name in ("radar", "vision", "imu", "gps")
            if sensor_validity[name]
            and isinstance(contributions.get(name), dict)
            and contributions[name].get("direction") == "raises"
        ]
    )
    score_state = (
        "unknown" if risk_score is None
        else "degraded" if is_degraded
        else "current"
    )
    radar_score_status = (
        "active" if module_scores["radar"] is not None else "unavailable"
    )
    vision_score_status = (
        "active" if module_scores["vision"] is not None else "unavailable"
    )
    imu_score_status = (
        "active" if module_scores["imu"] is not None else "unavailable"
    )
    gps_context_status = "active" if gps_valid else "invalid"
    startup_readiness = (
        "ready"
        if (
            camera_available
            and radar_valid
            and vision_valid
            and imu_valid
            and motor_status == "active"
            and gps_valid
        )
        else "degraded"
        if (
            camera_available
            and radar_valid
            and vision_valid
            and imu_valid
            and motor_status == "active"
        )
        else "not_ready"
    )
    risk_labels = {0: "低风险", 1: "中风险", 2: "高风险"}
    legacy_risk_labels = {0: "low", 1: "mid", 2: "high"}
    state = {
        "schema_version": "xgb-risk-v2",
        "timestamp": float(timestamp),
        "status": status,
        "decision_engine": "xgboost-with-deterministic-degradation",
        "decision_source": decision_source,
        "old_rules_loaded": True,
        "degraded": is_degraded,
        "missing_sensors": list(missing_sensors),
        "degradation": {
            "active": decision_source != "xgboost",
            "reasons": list(degraded_reasons),
            "missing_sensors": list(missing_sensors),
        },
        "motor_control": bool(motor_control),
        "motor": motor,
        "prediction": prediction,
        "prediction_error": prediction_error,
        "features": features,
        "modules": modules,
        "feature_window": feature_window,
        "risk_score": risk_score,
        "raw_risk_score": risk_score,
        "risk_score_state": score_state,
        "risk_decision_monotonic_ns": now_monotonic_ns,
        "risk_effective_updated_monotonic_ns": now_monotonic_ns,
        "risk_source_timing": {},
        "raw_risk_source_timing": {},
        "risk_timestamp_alignment": "xgboost_feature_window",
        "warning_rule_config": {},
        "risk_score_semantics": "xgboost_expected_ordinal_risk",
        "risk_level": level,
        "warning_level": level,
        "final_level": level,
        "raw_final_level": level,
        "last_known_level": level,
        "risk_label": legacy_risk_labels.get(level, "unknown"),
        "risk_status": risk_status,
        "system_status": system_status,
        "risk_reason": warning_reason,
        "warning_reason": warning_reason,
        "raw_warning_reason": warning_reason,
        "radar_level": radar_level,
        "vision_level": vision_level,
        "imu_level": imu_level,
        "radar_score": module_scores["radar"],
        "vision_score": module_scores["vision"],
        "imu_score": module_scores["imu"],
        "gps_score": module_scores["gps"],
        "radar_status": radar_status,
        "radar_safety_status": risk_status,
        "vision_status": sensor_states.get("vision", {}).get(
            "status", "disabled"
        ),
        "imu_status": sensor_states.get("imu", {}).get(
            "status", "disabled"
        ),
        "radar_score_status": radar_score_status,
        "vision_score_status": vision_score_status,
        "imu_score_status": imu_score_status,
        "gps_context_status": gps_context_status,
        "gps_speed_factor": 1.0,
        "vision_proximity_score": module_scores["vision"],
        "vision_proximity_adjusted_score": module_scores["vision"],
        "evidence_sources": evidence_sources,
        "both_modalities_active": radar_valid and vision_valid,
        "risk_rule": {
            "status": risk_status,
            "reason": warning_reason,
            "critical_ttc_s": (
                radar_min_ttc
                if radar_valid and radar_min_ttc >= 0.0
                else None
            ),
            "urgent_ttc_s": None,
            "critical_distance_m": (
                radar_nearest_distance
                if radar_valid
                and radar_nearest_distance >= 0.0
                else None
            ),
            "critical_lateral_m": None,
            "point_gate_half_width_m": None,
            "path_target_count": int(
                features.get("radar_path_target_count", 0.0) or 0
            ),
            "raw_target_count": len(radar_targets),
            "valid_target_count": len(radar_targets),
            "invalid_target_count": 0,
            "imu_level": imu_level,
            "imu_score": module_scores["imu"],
            "imu_status": imu_score_status,
            "imu_details": {},
        },
        "risk_items": {
            "obs": module_scores["vision"] or 0.0,
            "dist": module_scores["radar"] or 0.0,
            "pose": module_scores["imu"] or 0.0,
            "speed": module_scores["gps"] or 0.0,
            "ttc_s": (
                radar_min_ttc
                if radar_valid and radar_min_ttc >= 0.0
                else None
            ),
            "urgent_ttc_s": None,
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
            "connected": sensor_states.get("radar", {}).get("status")
            not in {"waiting", "port_closed"},
            "valid": radar_valid,
            "fresh": sensor_states.get("radar", {}).get("status") == "active",
            "target_count": len(radar_targets),
            "nearest_distance_m": radar_nearest_distance,
            "min_ttc_s": radar_min_ttc,
            "age_ms": sensor_states.get("radar", {}).get("age_ms"),
            "status": radar_status,
            "reason": sensor_states.get("radar", {}).get("error", ""),
            "diagnostics": {},
            "communication_alive": radar_valid,
            "target_age_fresh": radar_valid,
            "target_sync_fresh": radar_valid,
            "targets": serialized_radar_targets,
        },
        "gps_data": {
            "connected": sensor_states.get("gps", {}).get("status")
            not in {"waiting", "port_closed"},
            "valid": gps_valid,
            "fresh": sensor_states.get("gps", {}).get("status") == "active",
            "latitude": float(getattr(gps, "latitude", 0.0) or 0.0),
            "longitude": float(getattr(gps, "longitude", 0.0) or 0.0),
            "speed_kmh": max(0.0, float(getattr(gps, "speed_kmh", 0.0) or 0.0)),
            "fix_quality": int(getattr(gps, "fix_quality", 0) or 0),
            "satellites": int(getattr(gps, "satellites", 0) or 0),
            "status": gps_status,
            "reason": sensor_states.get("gps", {}).get("error", ""),
            "diagnostics": {},
        },
        "imu_data": {
            "connected": sensor_states.get("imu", {}).get("status")
            not in {"waiting", "port_closed"},
            "valid": imu_valid,
            "roll": float(getattr(imu, "body_roll", None)
                          if getattr(imu, "body_roll", None) is not None
                          else getattr(imu, "roll", 0.0) or 0.0),
            "pitch": float(getattr(imu, "body_pitch", None)
                           if getattr(imu, "body_pitch", None) is not None
                           else getattr(imu, "pitch", 0.0) or 0.0),
            "raw_roll": float(getattr(imu, "roll", 0.0) or 0.0),
            "raw_pitch": float(getattr(imu, "pitch", 0.0) or 0.0),
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
            "risk_level": imu_level,
            "risk_score": module_scores["imu"],
            "risk_status": imu_score_status,
            "turn_compensation_status": "unavailable",
            "roll_error_deg": features.get("roll_error_deg"),
            "outward_rate_deg_s": features.get("outward_rate_deg_s"),
            "time_to_critical_s": None,
        },
        "vision_details": {
            "valid": vision_valid,
            "object_count": len(vision_objects),
            "person_count": int(getattr(vision, "person_count", 0) or 0),
            "vehicle_count": int(getattr(vision, "vehicle_count", 0) or 0),
            "max_confidence": float(
                getattr(vision, "max_confidence", 0.0) or 0.0
            ),
            "drivable_area_ratio": (
                float(getattr(vision, "drivable_area_ratio", 0.0))
                if vision_valid else None
            ),
            "max_visual_risk": float(
                getattr(vision, "max_visual_risk", 0.0) or 0.0
            ),
            "objects": vision_objects,
            "frame_size": {"width": int(frame_width), "height": int(frame_height)},
        },
        "startup_readiness": startup_readiness,
        "fusion_data": {
            "valid": False,
            "vision_radar_count": 0,
            "vision_only_count": 0,
            "radar_only_count": 0,
        },
        "performance": {
            "vision_infer_ms": float(
                getattr(vision, "pipeline_inference_ms", 0.0) or 0.0
            ),
            "pipeline_infer_ms": float(
                getattr(vision, "pipeline_inference_ms", 0.0) or 0.0
            ),
            "detection_infer_ms": float(
                getattr(vision, "detection_inference_ms", 0.0) or 0.0
            ),
            "segmentation_infer_ms": float(
                getattr(vision, "segmentation_inference_ms", 0.0) or 0.0
            ),
            "radar_to_motor_ms": None,
            "radar_parsed_to_motor_go_ms": None,
            "vision_runtime": runtime.get("vision") or {},
        },
        "timestamps": {
            "risk_decision_monotonic_ns": now_monotonic_ns,
            "risk_effective_updated_monotonic_ns": now_monotonic_ns,
            "radar_age_ms": sensor_states.get("radar", {}).get("age_ms"),
            "vision_latency_ms": float(
                getattr(vision, "pipeline_inference_ms", 0.0) or 0.0
            ),
        },
        "mode": "real-sensors",
        "message": (
            f"规则降级 {risk_labels.get(level, '等待决策')}"
            if decision_source != "xgboost" and decision is not None
            else f"XGBoost {prediction.get('label')} · 主导贡献 {dominant.upper()}"
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
