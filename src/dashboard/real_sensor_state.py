"""Build live Dashboard state with a radar-first TTC warning fast path."""
from __future__ import annotations

import time


def _dispatch_motor(motor, decision) -> int:
    if motor is None or decision.status in {"unknown", "degraded"}:
        return 0
    if decision.level == 0:
        motor.alert_low()
    elif decision.level == 1:
        motor.alert_medium()
    else:
        motor.alert_high()
    if bool(getattr(motor, "last_command_was_dispatched", False)):
        return int(getattr(motor, "last_dispatch_monotonic_ns", 0) or 0)
    return 0


def _radar_diagnostic_state(
    *,
    connected: bool,
    raw_valid: bool,
    fresh: bool,
    communication_alive: bool,
    target_count: int,
    diagnostics: dict,
    age_ms: float | None,
) -> tuple[str, str]:
    if not connected:
        return "port_closed", "radar serial port is not open"
    if not raw_valid:
        if int(diagnostics.get("bytes_received", 0) or 0) <= 0:
            return "no_bytes", "serial port is open but no radar bytes have arrived"
        if int(diagnostics.get("valid_frame_count", 0) or 0) <= 0:
            return "no_valid_frame", "bytes arrived but no complete valid radar frame was decoded"
        return "read_invalid", "latest radar read did not produce a valid frame"
    if not fresh:
        if communication_alive:
            return "waiting", "radar communication is alive; awaiting the next target or no-target report"
        return "stale", f"last valid radar frame is stale ({age_ms:.0f} ms)" if age_ms is not None else "no radar sample timestamp"
    if target_count <= 0:
        return "no_target", "radar is healthy and currently reports no target"
    return "tracking", "radar is healthy and tracking target(s)"


def _gps_diagnostic_state(
    *,
    connected: bool,
    raw_valid: bool,
    sync_fresh: bool,
    fix_quality: int,
    diagnostics: dict,
) -> tuple[str, str]:
    if not connected:
        return "port_closed", "gps serial port is not open"
    if not raw_valid:
        if int(diagnostics.get("bytes_received", 0) or 0) <= 0:
            return "no_bytes", "serial port is open but no gps bytes have arrived"
        if int(diagnostics.get("valid_sentence_count", 0) or 0) <= 0:
            return "no_valid_sentence", "bytes arrived but no checksum-valid nmea sentence was decoded"
        if not bool(diagnostics.get("gga_fresh", False)):
            return "stale_gga", "gps position sentence is stale or missing"
        if not bool(diagnostics.get("rmc_fresh", False)):
            return "stale_rmc", "gps speed/status sentence is stale or missing"
        if fix_quality <= 0:
            return "no_fix", "gps is receiving data but has no positioning fix"
        return "invalid_fix", "gps data is present but not usable yet"
    if not sync_fresh:
        return "sync_stale", "gps sample is valid but not synchronized with the camera frame"
    return "active", "gps is healthy and synchronized"


def _camera_diagnostic_state(camera_available: bool) -> tuple[str, str]:
    if camera_available:
        return "active", "camera is open and frames are being produced"
    return "unavailable", "camera is not available or no frame can be read"


def _vision_diagnostic_state(vision_adapter, vision_valid: bool, vision) -> tuple[str, str]:
    if vision_adapter is None:
        return "disabled", "vision pipeline is not configured"
    if not bool(getattr(vision_adapter, "vision_enabled", False)):
        return "disabled", "vision pipeline is disabled or failed to initialize"
    if vision is None:
        return "waiting", "vision pipeline is enabled but no result has been produced yet"
    if vision_valid:
        return "active", "vision pipeline is producing valid perception results"
    return "invalid", "vision pipeline returned an invalid result"


def _motor_diagnostic_state(motor) -> tuple[str, str]:
    if motor is None:
        return "disabled", "motor controller is not configured"
    mode = str(getattr(motor, "mode", "unknown"))
    if mode == "mock":
        error = str(getattr(motor, "last_error", "") or "")
        return "mock", (f"real motor initialization failed: {error}"
                        if error else "motor controller is running in mock mode")
    if mode == "real" and getattr(motor, "_bus", None) is not None:
        return "active", "motor controller is connected to the haptic driver"
    if mode == "real":
        return "unavailable", "motor controller is in real mode but the i2c bus is not open"
    return "unknown", "motor controller state is unknown"


def build_real_sensor_state(
    camera_available: bool,
    radar_reader,
    gps_reader,
    imu_reader=None,
    frame=None,
    frame_capture_monotonic_ns: int = 0,
    camera_frame_id: int = -1,
    vision_adapter=None,
    vision_snapshot=None,
    process_vision: bool = True,
    fusion_engine=None,
    recorder=None,
    sync_thresholds=None,
    risk_rule=None,
    risk_model=None,
    classifier=None,
    motor=None,
    warning_system=None,
    imu_warning_rule=None,
    target_stale_ms: float = 500.0,
    radar_communication_watchdog_ms: float = 2000.0,
    record_sample: bool = True,
) -> dict:
    sync_thresholds = sync_thresholds or {"radar_max_delta_ms": 100.0, "gps_max_delta_ms": 1000.0}

    # Radar fast path: no GPS or vision operation may precede the decision.
    radar_start_ns = time.monotonic_ns()
    if warning_system is not None:
        radar = radar_reader.get_latest()
        if radar is None:
            from src.fusion.data_types import RadarData
            radar = RadarData(timestamp=time.time(), valid=False)
    else:
        radar = radar_reader.read_once()
    radar_end_ns = time.monotonic_ns()
    radar_sample_ns = int(getattr(radar_reader, "last_sample_monotonic_ns", 0) or 0)
    radar_age_ms = ((radar_end_ns - radar_sample_ns) / 1_000_000.0
                    if radar_sample_ns else None)
    radar_has_targets = bool(getattr(radar, "targets", []))
    radar_communication_alive = (
        radar_age_ms is not None
        and 0.0 <= radar_age_ms <= radar_communication_watchdog_ms
    )
    radar_target_age_fresh = (
        radar_age_ms is not None and 0.0 <= radar_age_ms <= target_stale_ms
    )
    radar_fresh = radar_communication_alive and (
        not radar_has_targets or radar_target_age_fresh
    )

    decision = (risk_rule.decide(radar, radar_fresh=radar_fresh)
                if risk_rule is not None and warning_system is None else None)
    risk_decision_ns = time.monotonic_ns()
    motor_command_ns = (_dispatch_motor(motor, decision)
                        if decision is not None and warning_system is None else 0)
    radar_to_motor_ms = ((motor_command_ns - radar_sample_ns) / 1_000_000.0
                         if motor_command_ns and radar_sample_ns else None)

    # Non-safety-critical sensors and vision run only after motor dispatch.
    gps_start_ns = time.monotonic_ns()
    gps = gps_reader.read_once()
    gps_end_ns = time.monotonic_ns()
    imu_start_ns = time.monotonic_ns()
    if imu_reader is not None:
        imu = imu_reader.read_once()
    else:
        from src.fusion.data_types import IMUData
        imu = IMUData(timestamp=time.time(), valid=False)
    imu_end_ns = time.monotonic_ns()
    imu_sample_ns = int(
        getattr(imu_reader, "last_sample_monotonic_ns", 0) or 0)
    gps_sample_ns = int(getattr(gps_reader, "last_sample_monotonic_ns", 0) or 0)
    camera_ns = frame_capture_monotonic_ns or radar_start_ns
    radar_delta_ms = ((radar_sample_ns - camera_ns) / 1_000_000.0 if radar_sample_ns else None)
    gps_delta_ms = ((gps_sample_ns - camera_ns) / 1_000_000.0 if gps_sample_ns else None)
    radar_target_sync_fresh = (
        radar_delta_ms is not None
        and abs(radar_delta_ms) <= float(sync_thresholds["radar_max_delta_ms"])
    )
    radar_measurement_fresh = radar_communication_alive and (
        not radar_has_targets or (radar_target_age_fresh and radar_target_sync_fresh)
    )
    gps_fresh = (gps_delta_ms is not None and
                 abs(gps_delta_ms) <= float(sync_thresholds["gps_max_delta_ms"]))
    if warning_system is not None:
        from src.fusion.warning_events import ModalityEvent

        gps_usable = bool(gps.valid) and gps_fresh
        warning_system.publish_gps(ModalityEvent(
            source="gps", source_id=str(gps_sample_ns), sequence=gps_sample_ns,
            capture_monotonic_ns=gps_sample_ns,
            completed_monotonic_ns=gps_end_ns,
            usable=gps_usable, level=0 if gps_usable else None,
            reason=("gps_speed_context" if gps_usable
                    else "gps_context_stale" if gps.valid else "gps_context_invalid"),
            status=("usable" if gps_usable else "stale" if gps.valid else "invalid"),
            details={"speed_kmh": float(gps.speed_kmh)},
        ))
        if imu_warning_rule is not None:
            imu_capture_ns = imu_sample_ns or imu_start_ns
            imu_gps_context_usable = bool(
                gps.valid and gps_sample_ns > 0
                and abs(imu_capture_ns - gps_sample_ns)
                <= float(sync_thresholds["gps_max_delta_ms"]) * 1_000_000.0)
            imu_event = imu_warning_rule.evaluate_event(
                imu,
                capture_monotonic_ns=imu_capture_ns,
                completed_monotonic_ns=imu_end_ns,
                sequence=imu_capture_ns,
                gps_speed_kmh=float(gps.speed_kmh),
                gps_usable=imu_gps_context_usable,
            )
            warning_system.publish_imu(imu_event, now_ns=imu_end_ns)

    vision = None
    fusion = None
    inference_ms = 0.0
    vision_start_ns = 0
    vision_finish_ns = 0
    if vision_adapter is not None:
        if vision_snapshot is not None:
            vision = vision_snapshot.vision_data
            raw_result = vision_snapshot.vision_result
            vision_start_ns = int(vision_snapshot.vision_start_monotonic_ns)
            vision_finish_ns = int(vision_snapshot.vision_finish_monotonic_ns)
            inference_ms = float(vision_snapshot.inference_ms)
        elif process_vision and frame is not None:
            vision_start_ns = time.monotonic_ns()
            vision = vision_adapter.process(frame)
            vision_finish_ns = time.monotonic_ns()
            inference_ms = float(getattr(vision, "pipeline_inference_ms", 0.0))
            if inference_ms <= 0.0:
                inference_ms = (vision_finish_ns - vision_start_ns) / 1_000_000.0
            raw_result = vision_adapter.get_latest_vision_result()
        else:
            vision = vision_adapter.get_latest()
            if vision is not None:
                inference_ms = float(getattr(vision, "pipeline_inference_ms", 0.0))
                if inference_ms <= 0.0:
                    inference_ms = (float(getattr(vision, "detection_inference_ms", 0.0)) +
                                    float(getattr(vision, "segmentation_inference_ms", 0.0)))
            raw_result = vision_adapter.get_latest_vision_result()
        if raw_result is not None and fusion_engine is not None and radar_measurement_fresh:
            fusion = fusion_engine.fuse_vision_result(raw_result, radar)

    timestamps = {
        "frame_capture_monotonic_ns": camera_ns,
        "radar_read_start_monotonic_ns": radar_start_ns,
        "radar_read_end_monotonic_ns": radar_end_ns,
        "radar_sample_monotonic_ns": radar_sample_ns,
        "risk_decision_monotonic_ns": risk_decision_ns,
        "motor_command_monotonic_ns": motor_command_ns,
        "gps_read_start_monotonic_ns": gps_start_ns,
        "gps_read_end_monotonic_ns": gps_end_ns,
        "gps_sample_monotonic_ns": gps_sample_ns,
        "imu_read_start_monotonic_ns": imu_start_ns,
        "imu_read_end_monotonic_ns": imu_end_ns,
        "imu_sample_monotonic_ns": imu_sample_ns,
        "vision_start_monotonic_ns": vision_start_ns,
        "vision_finish_monotonic_ns": vision_finish_ns,
        "radar_age_ms": radar_age_ms,
        "radar_has_targets": radar_has_targets,
        "radar_communication_alive": radar_communication_alive,
        "radar_target_age_fresh": radar_target_age_fresh,
        "radar_target_sync_fresh": radar_target_sync_fresh,
        "radar_delta_ms": radar_delta_ms,
        "gps_delta_ms": gps_delta_ms,
        "vision_latency_ms": inference_ms,
        "radar_to_motor_latency_ms": radar_to_motor_ms,
    }
    warning_snapshot = warning_system.snapshot() if warning_system is not None else None
    if warning_snapshot is not None:
        timestamps["risk_decision_monotonic_ns"] = int(
            warning_snapshot["risk_decision_monotonic_ns"])
        timestamps["risk_effective_updated_monotonic_ns"] = int(
            warning_snapshot["risk_effective_updated_monotonic_ns"])
        timestamps["risk_source_timing"] = warning_snapshot["risk_source_timing"]
        timestamps["raw_risk_source_timing"] = warning_snapshot["raw_risk_source_timing"]
        timestamps["risk_timestamp_alignment"] = warning_snapshot[
            "risk_timestamp_alignment"]
        latest_radar_event = warning_snapshot.get("radar_event")
        dispatch_ns = int(getattr(motor, "last_dispatch_monotonic_ns", 0) or 0)
        parsed_ns = int(getattr(latest_radar_event, "capture_monotonic_ns", 0) or 0)
        if dispatch_ns and parsed_ns and dispatch_ns >= parsed_ns:
            radar_to_motor_ms = (dispatch_ns - parsed_ns) / 1_000_000.0
            timestamps["radar_to_motor_latency_ms"] = radar_to_motor_ms
            timestamps["radar_parsed_to_motor_go_latency_ms"] = radar_to_motor_ms
    if recorder is not None and record_sample:
        recorder.write(
            frame, radar, gps, vision, fusion, inference_ms, timestamps,
            camera_frame_id=camera_frame_id,
            radar_valid=bool(radar.valid) and radar_measurement_fresh,
            gps_valid=bool(gps.valid) and gps_fresh,
            risk_decision=warning_snapshot if warning_snapshot is not None else decision,
        )

    targets = [{
        "target_id": int(t.target_id),
        "distance_m": round(float(t.distance_m), 2),
        "relative_speed_mps": round(float(t.relative_speed_mps), 2),
        "angle_deg": round(float(t.angle_deg), 1),
        "confidence": round(float(t.confidence), 3),
    } for t in radar.targets]
    objects = [] if vision is None else [{
        "class_name": o.class_name, "risk_class": o.risk_class,
        "confidence": round(float(o.confidence), 3), "bbox": list(o.bbox),
        "in_drivable_area": o.in_drivable_area,
    } for o in vision.objects]

    radar_connected = getattr(radar_reader, "_serial", None) is not None
    gps_connected = getattr(gps_reader, "_serial", None) is not None
    imu_connected = (imu_reader is not None
                     and getattr(imu_reader, "_serial", None) is not None)
    vision_valid = bool(vision is not None and vision.valid)
    radar_diagnostics = (radar_reader.get_diagnostics()
                         if hasattr(radar_reader, "get_diagnostics") else {})
    gps_diagnostics = (gps_reader.get_diagnostics()
                       if hasattr(gps_reader, "get_diagnostics") else {})
    imu_diagnostics = (
        imu_reader.get_diagnostics()
        if imu_reader is not None and hasattr(imu_reader, "get_diagnostics") else {})
    radar_state, radar_reason = _radar_diagnostic_state(
        connected=radar_connected,
        raw_valid=bool(radar.valid),
        fresh=radar_fresh,
        communication_alive=radar_communication_alive,
        target_count=len(targets),
        diagnostics=radar_diagnostics,
        age_ms=radar_age_ms,
    )
    gps_state, gps_reason = _gps_diagnostic_state(
        connected=gps_connected,
        raw_valid=bool(gps.valid),
        sync_fresh=gps_fresh,
        fix_quality=int(getattr(gps, "fix_quality", 0) or 0),
        diagnostics=gps_diagnostics,
    )
    camera_state, camera_reason = _camera_diagnostic_state(bool(camera_available))
    vision_state, vision_reason = _vision_diagnostic_state(vision_adapter, vision_valid, vision)
    motor_state, motor_reason = _motor_diagnostic_state(motor)
    hardware_status = {
        "camera": {"status": camera_state, "reason": camera_reason},
        "vision": {"status": vision_state, "reason": vision_reason},
        "radar": {"status": radar_state, "reason": radar_reason, "diagnostics": radar_diagnostics},
        "gps": {"status": gps_state, "reason": gps_reason, "diagnostics": gps_diagnostics},
        "imu": {
            "status": "active" if imu.valid else "invalid" if imu_connected else "disabled",
            "reason": ("imu is producing valid posture data" if imu.valid
                       else "imu serial is open but a complete sample is not available" if imu_connected
                       else "imu is not enabled"),
            "diagnostics": imu_diagnostics,
        },
        "motor": {"status": motor_state, "reason": motor_reason},
    }
    required_ready = bool(
        camera_state == "active"
        and vision_state == "active"
        and radar_state in {"tracking", "no_target"}
        and imu.valid
        and warning_snapshot is not None
        and motor_state == "active"
    )
    if not required_ready:
        startup_readiness = "not_ready"
    elif gps_state == "active":
        startup_readiness = "ready"
    else:
        startup_readiness = "degraded"
    fused_items = None
    fused_weights = {}
    if warning_snapshot is not None:
        fused_score = warning_snapshot["risk_score"]
        level = warning_snapshot["warning_level"]
        status = warning_snapshot["system_status"]
        label = ({0: "low", 1: "mid", 2: "high"}.get(level, "unknown"))
        fused_items = None
        fused_weights = {}
    elif risk_model is not None and classifier is not None:
        from copy import copy
        from src.fusion.data_types import FusionInput, IMUData, VisionData

        risk_radar = copy(radar)
        risk_radar.valid = bool(radar.valid) and radar_measurement_fresh
        risk_gps = copy(gps)
        risk_gps.valid = bool(gps.valid) and gps_fresh
        risk_vision = vision if vision is not None else VisionData(timestamp=time.time(), valid=False)
        risk_input = FusionInput(
            timestamp=time.time(), gps=risk_gps, imu=IMUData(timestamp=time.time(), valid=False),
            radar=risk_radar, vision=risk_vision, vision_enabled=vision_adapter is not None,
        )
        fused_items, fused_weights = risk_model.compute(risk_input)
        fused_score = max(0.0, min(1.0, float(fused_items["risk_score"])))
        level, label = classifier.classify(fused_score)
        status = "active" if any((risk_radar.valid, risk_gps.valid, vision_valid)) else "degraded"
    else:
        fused_score = None
        level = int(decision.level) if decision is not None else 0
        label = decision.label if decision is not None else "disabled"
        status = decision.status if decision is not None else "disabled"
    reason = (warning_snapshot["warning_reason"] if warning_snapshot is not None else
              decision.reason if decision is not None else
              ("adaptive_fusion" if fused_score is not None else "risk_rule_disabled"))
    display_index = (fused_score if fused_score is not None else
                     (level / 2.0 if level is not None and status != "unknown" else None))

    radar_warning_event = warning_snapshot.get("radar_event") if warning_snapshot else None
    imu_warning_event = warning_snapshot.get("imu_event") if warning_snapshot else None
    radar_warning_details = (dict(radar_warning_event.details)
                             if radar_warning_event is not None else {})
    risk_rule_state = {
        "status": status,
        "reason": reason,
        "critical_ttc_s": (radar_warning_details.get("critical_ttc_s")
                           if warning_snapshot else decision.min_path_ttc_s if decision else None),
        "urgent_ttc_s": (radar_warning_details.get("urgent_ttc_s")
                         if warning_snapshot else decision.urgent_ttc_s if decision else None),
        "critical_distance_m": (radar_warning_details.get("critical_distance_m")
                                if warning_snapshot else decision.critical_distance_m if decision else None),
        "critical_lateral_m": (radar_warning_details.get("critical_lateral_m")
                               if warning_snapshot else decision.critical_lateral_m if decision else None),
        "point_gate_half_width_m": (radar_warning_details.get("point_gate_half_width_m")
                                    if warning_snapshot else decision.point_gate_half_width_m if decision else None),
        "path_target_count": (radar_warning_details.get("path_target_count", 0)
                              if warning_snapshot else decision.path_target_count if decision else 0),
        "raw_target_count": (radar_warning_details.get("raw_target_count", 0)
                             if warning_snapshot else decision.raw_target_count if decision else 0),
        "valid_target_count": (radar_warning_details.get("valid_target_count", 0)
                               if warning_snapshot else decision.valid_target_count if decision else 0),
        "invalid_target_count": (radar_warning_details.get("invalid_target_count", 0)
                                 if warning_snapshot else decision.invalid_target_count if decision else 0),
        "imu_level": (warning_snapshot.get("imu_level")
                      if warning_snapshot else None),
        "imu_score": (warning_snapshot.get("imu_score")
                      if warning_snapshot else None),
        "imu_status": (warning_snapshot.get("imu_status")
                       if warning_snapshot else "disabled"),
        "imu_details": (dict(imu_warning_event.details)
                        if imu_warning_event is not None else {}),
    }

    return {
        "timestamp": time.time(),
        "risk_score": display_index,
        "raw_risk_score": (warning_snapshot["raw_risk_score"]
                           if warning_snapshot else display_index),
        "risk_score_state": (warning_snapshot["risk_score_state"]
                             if warning_snapshot else "legacy"),
        "risk_decision_monotonic_ns": (
            warning_snapshot["risk_decision_monotonic_ns"]
            if warning_snapshot else risk_decision_ns),
        "risk_effective_updated_monotonic_ns": (
            warning_snapshot["risk_effective_updated_monotonic_ns"]
            if warning_snapshot else risk_decision_ns),
        "risk_source_timing": (warning_snapshot["risk_source_timing"]
                               if warning_snapshot else {}),
        "raw_risk_source_timing": (warning_snapshot["raw_risk_source_timing"]
                                   if warning_snapshot else {}),
        "risk_timestamp_alignment": (
            warning_snapshot["risk_timestamp_alignment"]
            if warning_snapshot else "legacy"),
        "warning_rule_config": (warning_snapshot["warning_rule_config"]
                                if warning_snapshot else {}),
        "risk_score_semantics": ("rule_based_intervention_urgency" if warning_snapshot is not None
                                 else "adaptive_weighted_fusion" if fused_score is not None
                                 else "ordinal_display_index_not_probability"),
        "risk_level": level,
        "warning_level": level,
        "final_level": level,
        "raw_final_level": (warning_snapshot["raw_final_level"]
                            if warning_snapshot else level),
        "last_known_level": (warning_snapshot["last_known_level"]
                             if warning_snapshot else level),
        "risk_label": label,
        "risk_status": status,
        "risk_reason": reason,
        "warning_reason": reason,
        "raw_warning_reason": (warning_snapshot["raw_warning_reason"]
                               if warning_snapshot else reason),
        "system_status": status,
        "radar_level": warning_snapshot["radar_level"] if warning_snapshot else level,
        "vision_level": warning_snapshot["vision_level"] if warning_snapshot else None,
        "imu_level": warning_snapshot["imu_level"] if warning_snapshot else None,
        "radar_score": warning_snapshot["radar_score"] if warning_snapshot else None,
        "vision_score": warning_snapshot["vision_score"] if warning_snapshot else None,
        "imu_score": warning_snapshot["imu_score"] if warning_snapshot else None,
        "radar_status": radar_state,
        "radar_safety_status": (warning_snapshot["radar_status"]
                                if warning_snapshot else status),
        "vision_status": warning_snapshot["vision_status"] if warning_snapshot else "off",
        "imu_status": warning_snapshot["imu_status"] if warning_snapshot else "off",
        "radar_score_status": (warning_snapshot["radar_score_status"]
                               if warning_snapshot else "legacy"),
        "vision_score_status": (warning_snapshot["vision_score_status"]
                                if warning_snapshot else "legacy"),
        "imu_score_status": (warning_snapshot["imu_score_status"]
                             if warning_snapshot else "legacy"),
        "gps_context_status": (warning_snapshot["gps_status"]
                               if warning_snapshot else "off"),
        "gps_speed_factor": (warning_snapshot["gps_speed_factor"]
                             if warning_snapshot else 1.0),
        "vision_proximity_score": (warning_snapshot["vision_proximity_score"]
                                   if warning_snapshot else None),
        "vision_proximity_adjusted_score": (
            warning_snapshot["vision_proximity_adjusted_score"]
            if warning_snapshot else None),
        "evidence_sources": (list(warning_snapshot["evidence_sources"])
                             if warning_snapshot else (["radar"] if level else [])),
        "both_modalities_active": (warning_snapshot["both_modalities_active"]
                                   if warning_snapshot else False),
        "risk_rule": risk_rule_state,
        "risk_items": {"obs": float(fused_items["R_obs"]) if fused_items else 0.0,
                       "dist": float(fused_items["R_dist"]) if fused_items else 0.0,
                       "pose": float(fused_items["R_pose"]) if fused_items else 0.0,
                       "speed": float(fused_items["R_speed"]) if fused_items else 0.0,
                       "ttc_s": risk_rule_state["critical_ttc_s"],
                       "urgent_ttc_s": risk_rule_state["urgent_ttc_s"]},
        "weights": fused_weights,
        "sensors": {
            "camera": "real" if camera_state == "active" else "invalid",
            "vision": ("real" if vision_state == "active"
                       else "invalid" if vision_state in {"waiting", "invalid"}
                       else "off"),
            "radar": ("real" if radar_state in {"tracking", "no_target", "waiting"}
                      else "invalid" if radar_connected else "off"),
            "gps": "real" if gps_state == "active" else "invalid" if gps_connected else "off",
            "imu": "real" if imu.valid else "invalid" if imu_connected else "off",
            "motor": "mock" if motor_state == "mock" else "real" if motor_state == "active" else "off",
        },
        "hardware_status": hardware_status,
        "startup_readiness": startup_readiness,
        "mode": "real-recording" if recorder is not None else "real-sensors",
        "message": ("Parallel radar/vision/IMU warning arbitration active; no modality weights"
                    if warning_snapshot is not None
                    else "Legacy adaptive risk calculation active"),
        "radar_data": {
            "connected": radar_connected,
            "valid": bool(radar.valid) and radar_measurement_fresh,
            "fresh": radar_measurement_fresh,
            "status": radar_state,
            "reason": radar_reason,
            "diagnostics": radar_diagnostics,
            "communication_alive": radar_communication_alive,
            "target_age_fresh": radar_target_age_fresh,
            "target_sync_fresh": radar_target_sync_fresh,
            "age_ms": radar_age_ms,
            "target_count": len(targets),
            "nearest_distance_m": round(float(radar.nearest_distance_m), 2),
            "min_ttc_s": round(float(radar.min_ttc), 2),
            "targets": targets,
        },
        "gps_data": {
            "connected": gps_connected, "valid": bool(gps.valid) and gps_fresh,
            "fresh": gps_fresh,
            "status": gps_state,
            "reason": gps_reason,
            "diagnostics": gps_diagnostics,
            "speed_kmh": round(float(gps.speed_kmh), 2),
            "latitude": round(float(gps.latitude), 7),
            "longitude": round(float(gps.longitude), 7),
            "fix_quality": int(gps.fix_quality),
        },
        "imu_data": {
            "connected": imu_connected,
            "valid": bool(imu.valid),
            "roll": round(float(
                imu.body_roll if imu.body_roll is not None else imu.roll), 2),
            "pitch": round(float(
                imu.body_pitch if imu.body_pitch is not None else imu.pitch), 2),
            "raw_roll": round(float(imu.roll), 2),
            "raw_pitch": round(float(imu.pitch), 2),
            "yaw": round(float(imu.yaw), 2),
            "acc_x": round(float(imu.acc_x), 2),
            "acc_y": round(float(imu.acc_y), 2),
            "acc_z": round(float(imu.acc_z), 2),
            "gyro_x": round(float(imu.gyro_x), 2),
            "gyro_y": round(float(imu.gyro_y), 2),
            "gyro_z": round(float(imu.gyro_z), 2),
            "brake_score": round(float(imu.brake_score), 3),
            "bump_score": round(float(imu.bump_score), 3),
            "tilt_score": round(float(imu.tilt_score), 3),
            "risk_level": (warning_snapshot.get("imu_level")
                           if warning_snapshot else None),
            "risk_score": (warning_snapshot.get("imu_score")
                           if warning_snapshot else None),
            "risk_status": (warning_snapshot.get("imu_status")
                            if warning_snapshot else "disabled"),
            "turn_compensation_status": (
                imu_warning_event.details.get("turn_compensation_status")
                if imu_warning_event is not None else "unavailable"),
            "roll_error_deg": (
                imu_warning_event.details.get("roll_error_deg")
                if imu_warning_event is not None else None),
            "outward_rate_deg_s": (
                imu_warning_event.details.get("outward_rate_deg_s")
                if imu_warning_event is not None else None),
            "time_to_critical_s": (
                imu_warning_event.details.get("time_to_critical_s")
                if imu_warning_event is not None else None),
        },
        "fusion_data": {
            "valid": fusion is not None,
            "vision_radar_count": fusion.n_vision_radar if fusion else 0,
            "vision_only_count": fusion.n_vision_only if fusion else 0,
            "radar_only_count": fusion.n_radar_only if fusion else 0,
        },
        "performance": {
            "vision_infer_ms": round(inference_ms, 2),
            "pipeline_infer_ms": round(float(getattr(vision, "pipeline_inference_ms", inference_ms)), 2),
            "detection_infer_ms": round(float(getattr(vision, "detection_inference_ms", 0.0)), 2),
            "segmentation_infer_ms": round(float(getattr(vision, "segmentation_inference_ms", 0.0)), 2),
            "radar_to_motor_ms": radar_to_motor_ms,
            "radar_parsed_to_motor_go_ms": radar_to_motor_ms,
            "vision_runtime": vision_adapter.get_runtime_info() if vision_adapter is not None else {},
        },
        "timestamps": timestamps,
        "vision_details": {
            "valid": vision_valid, "object_count": len(objects),
            "person_count": vision.person_count if vision else 0,
            "vehicle_count": vision.vehicle_count if vision else 0,
            "max_confidence": float(vision.max_confidence) if vision else 0.0,
            "drivable_area_ratio": float(vision.drivable_area_ratio) if vision else 0.0,
            "max_visual_risk": float(vision.max_visual_risk) if vision else 0.0,
            "objects": objects,
            "frame_size": {"width": frame.shape[1] if frame is not None else 640,
                           "height": frame.shape[0] if frame is not None else 480},
        },
    }
