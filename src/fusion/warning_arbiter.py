"""No-veto, no-weight arbitration across usable modality events."""
from __future__ import annotations

from typing import Optional

from src.fusion.gps_risk_context import (
    DEFAULT_GPS_SPEED_MODIFIER,
    GpsSpeedModifierConfig,
    adjust_visual_proximity_score,
    gps_speed_factor,
)
from src.fusion.risk_score_contract import normalize_risk_score
from src.fusion.warning_events import ArbitrationResult, ModalityEvent


def _event_score(event: Optional[ModalityEvent], usable: bool):
    if not usable or event is None:
        return normalize_risk_score(None, None)
    return normalize_risk_score(event.level, event.risk_score)


def arbitrate_warning_events(
    radar_event: Optional[ModalityEvent],
    vision_event: Optional[ModalityEvent],
    gps_event: Optional[ModalityEvent] = None,
    imu_event: Optional[ModalityEvent] = None,
    *,
    now_ns: int,
    target_stale_ms: float = 500.0,
    vision_stale_ms: float = 500.0,
    gps_stale_ms: float = 1000.0,
    imu_stale_ms: float = 100.0,
    imu_enabled: bool = False,
    gps_modifier_config: GpsSpeedModifierConfig = DEFAULT_GPS_SPEED_MODIFIER,
    radar_communication_watchdog_ms: float = 2000.0,
) -> ArbitrationResult:
    radar_stale_limit = (radar_communication_watchdog_ms
                         if radar_event and radar_event.reason == "radar_reports_no_target"
                         else target_stale_ms)
    radar_usable = bool(radar_event and radar_event.fresh(
        now_ns, radar_stale_limit, check_capture=True))
    vision_usable = bool(vision_event and vision_event.fresh(
        now_ns, vision_stale_ms, check_capture=True))
    gps_usable = bool(gps_event and gps_event.fresh(
        now_ns, gps_stale_ms, check_capture=True))
    imu_usable = bool(imu_event and imu_event.fresh(
        now_ns, imu_stale_ms, check_capture=True))

    radar_level = radar_event.level if radar_usable else None
    vision_level = vision_event.level if vision_usable else None
    imu_level = imu_event.level if imu_usable else None
    radar_score_result = _event_score(radar_event, radar_usable)
    vision_score_result = _event_score(vision_event, vision_usable)
    imu_score_result = _event_score(imu_event, imu_usable)
    radar_score = radar_score_result.score
    vision_score = vision_score_result.score
    imu_score = imu_score_result.score
    gps_speed_kmh = (gps_event.details.get("speed_kmh")
                     if gps_usable and gps_event is not None else None)
    speed_factor = 1.0
    proximity_score = None
    adjusted_proximity_score = None
    if vision_usable and vision_event is not None:
        proximity_score = vision_event.details.get("proximity_risk_score")
        tau_score = vision_event.details.get("tau_risk_score")
        if gps_usable and gps_event is not None and proximity_score is not None:
            speed_factor = gps_speed_factor(
                gps_speed_kmh, usable=True, config=gps_modifier_config)
            adjusted_proximity_score = adjust_visual_proximity_score(
                proximity_score, gps_speed_kmh, gps_usable=True,
                config=gps_modifier_config)
            try:
                tau_value = max(0.0, min(1.0, float(tau_score or 0.0)))
                adjusted_result = normalize_risk_score(
                    vision_level, max(tau_value, adjusted_proximity_score))
                vision_score = adjusted_result.score
                vision_score_result = adjusted_result
            except (TypeError, ValueError):
                adjusted_proximity_score = None
        elif proximity_score is not None:
            try:
                proximity_score = max(0.0, min(1.0, float(proximity_score)))
            except (TypeError, ValueError):
                proximity_score = None
    levels = [v for v in (radar_level, vision_level, imu_level) if v is not None]
    final_level = max(levels) if levels else None
    scores = [value for value in (radar_score, vision_score, imu_score)
              if value is not None]
    risk_score = max(scores) if scores else None

    evidence = []
    if radar_level is not None and radar_level > 0:
        evidence.append("radar")
    if vision_level is not None and vision_level > 0:
        evidence.append("vision")
    if imu_level is not None and imu_level > 0:
        evidence.append("imu")

    imu_required = bool(imu_enabled or imu_event is not None)
    required_usable = [radar_usable, vision_usable]
    if imu_required:
        required_usable.append(imu_usable)
    if all(required_usable):
        system_status = "normal"
    elif any(required_usable):
        system_status = "degraded"
    else:
        system_status = "unknown"

    reason = "no_usable_modality"
    if radar_level == 2:
        reason = radar_event.reason
    elif vision_level == 2:
        reason = vision_event.reason
    elif imu_level == 2:
        reason = imu_event.reason
    elif radar_level == 1:
        reason = radar_event.reason
    elif vision_level == 1:
        reason = vision_event.reason
    elif imu_level == 1:
        reason = imu_event.reason
    elif final_level == 0:
        reason = "no_warning_event"

    def status(event: Optional[ModalityEvent], usable: bool, stale_ms: float,
               *, check_capture: bool) -> str:
        if usable:
            return event.status
        if event is None:
            return "unavailable"
        temporal_status = event.temporal_status(
            now_ns, stale_ms, check_capture=check_capture)
        if temporal_status != "fresh":
            return temporal_status
        return event.status

    return ArbitrationResult(
        decision_monotonic_ns=now_ns,
        final_level=final_level, risk_score=risk_score, system_status=system_status,
        warning_reason=reason, evidence_sources=tuple(evidence),
        both_modalities_active=len(evidence) >= 2,
        radar_level=radar_level, vision_level=vision_level, imu_level=imu_level,
        radar_score=radar_score, vision_score=vision_score, imu_score=imu_score,
        radar_score_status=radar_score_result.status,
        vision_score_status=vision_score_result.status,
        imu_score_status=imu_score_result.status,
        radar_status=status(radar_event, radar_usable, radar_stale_limit,
                            check_capture=True),
        vision_status=status(vision_event, vision_usable, vision_stale_ms,
                             check_capture=True),
        imu_status=status(imu_event, imu_usable, imu_stale_ms,
                          check_capture=True),
        gps_status=status(gps_event, gps_usable, gps_stale_ms,
                          check_capture=True),
        gps_speed_kmh=gps_speed_kmh,
        gps_speed_factor=speed_factor,
        vision_proximity_score=proximity_score,
        vision_proximity_adjusted_score=adjusted_proximity_score,
    )
