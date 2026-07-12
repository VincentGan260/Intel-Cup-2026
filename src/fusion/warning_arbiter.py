"""No-veto, no-weight arbitration across usable modality events."""
from __future__ import annotations

from typing import Optional

from src.fusion.warning_events import ArbitrationResult, ModalityEvent


def arbitrate_warning_events(
    radar_event: Optional[ModalityEvent],
    vision_event: Optional[ModalityEvent],
    *,
    now_ns: int,
    target_stale_ms: float = 500.0,
    vision_stale_ms: float = 500.0,
    radar_communication_watchdog_ms: float = 2000.0,
) -> ArbitrationResult:
    radar_stale_limit = (radar_communication_watchdog_ms
                         if radar_event and radar_event.reason == "radar_reports_no_target"
                         else target_stale_ms)
    radar_usable = bool(radar_event and radar_event.fresh(
        now_ns, radar_stale_limit, check_capture=False))
    vision_usable = bool(vision_event and vision_event.fresh(
        now_ns, vision_stale_ms, check_capture=True))

    radar_level = radar_event.level if radar_usable else None
    vision_level = vision_event.level if vision_usable else None
    levels = [v for v in (radar_level, vision_level) if v is not None]
    final_level = max(levels) if levels else None

    evidence = []
    if radar_level is not None and radar_level > 0:
        evidence.append("radar")
    if vision_level is not None and vision_level > 0:
        evidence.append("vision")

    if radar_usable and vision_usable:
        system_status = "normal"
    elif radar_usable or vision_usable:
        system_status = "degraded"
    else:
        system_status = "unknown"

    reason = "no_usable_modality"
    if radar_level == 2:
        reason = radar_event.reason
    elif radar_level == 1:
        reason = radar_event.reason
    elif vision_level == 1:
        reason = vision_event.reason
    elif final_level == 0:
        reason = "no_warning_event"

    def status(event: Optional[ModalityEvent], usable: bool) -> str:
        if usable:
            return event.status
        if event is None:
            return "unavailable"
        if event.usable:
            return "stale"
        return event.status

    return ArbitrationResult(
        final_level=final_level, system_status=system_status,
        warning_reason=reason, evidence_sources=tuple(evidence),
        both_modalities_active=len(evidence) == 2,
        radar_level=radar_level, vision_level=vision_level,
        radar_status=status(radar_event, radar_usable),
        vision_status=status(vision_event, vision_usable),
    )
