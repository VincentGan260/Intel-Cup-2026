"""Coordinator for asynchronous radar and visual warning events."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Optional

from src.fusion.gps_risk_context import (
    DEFAULT_GPS_SPEED_MODIFIER,
    GpsSpeedModifierConfig,
)
from src.fusion.risk_score_contract import normalize_risk_score
from src.fusion.warning_arbiter import arbitrate_warning_events
from src.fusion.warning_events import ArbitrationResult, ModalityEvent
from src.fusion.warning_state import WarningState


class MultimodalWarningSystem:
    def __init__(self, *, motor=None, target_stale_ms: float = 500.0,
                 vision_stale_ms: float = 500.0,
                 gps_stale_ms: float = 1000.0,
                 imu_stale_ms: float = 100.0,
                 imu_enabled: bool = False,
                 gps_modifier_config: GpsSpeedModifierConfig = DEFAULT_GPS_SPEED_MODIFIER,
                 release_hold_ms: float = 500.0,
                 radar_communication_watchdog_ms: float = 2000.0,
                 rule_config_metadata: Optional[dict] = None) -> None:
        self.target_stale_ms = target_stale_ms
        self.vision_stale_ms = vision_stale_ms
        self.gps_stale_ms = gps_stale_ms
        self.imu_stale_ms = imu_stale_ms
        self.imu_enabled = imu_enabled
        self.gps_modifier_config = gps_modifier_config
        self.rule_config_metadata = dict(rule_config_metadata or {})
        self.radar_communication_watchdog_ms = radar_communication_watchdog_ms
        self.state = WarningState(motor, release_hold_ms=release_hold_ms)
        self._lock = threading.Lock()
        self._radar_event: Optional[ModalityEvent] = None
        self._vision_event: Optional[ModalityEvent] = None
        self._gps_event: Optional[ModalityEvent] = None
        self._imu_event: Optional[ModalityEvent] = None
        self._effective_risk_score: Optional[float] = None
        self._effective_warning_reason = "no_usable_modality"
        self._effective_evidence_sources: tuple[str, ...] = ()
        self._effective_both_modalities_active = False
        self._risk_score_state = "unknown"
        self._effective_updated_monotonic_ns = 0
        self._effective_source_timestamps: dict[str, dict[str, int]] = {}
        self._raw_source_timestamps: dict[str, dict[str, int]] = {}
        self._arbiter_sequence = 0
        self._last_result = arbitrate_warning_events(
            None, None, now_ns=time.monotonic_ns(),
            target_stale_ms=target_stale_ms, vision_stale_ms=vision_stale_ms,
            gps_stale_ms=gps_stale_ms,
            imu_stale_ms=imu_stale_ms, imu_enabled=imu_enabled,
            gps_modifier_config=gps_modifier_config,
            radar_communication_watchdog_ms=radar_communication_watchdog_ms)

    def publish_radar(self, event: ModalityEvent, *, fast: bool = True,
                      now_ns: Optional[int] = None) -> ArbitrationResult:
        publish_ns = now_ns or time.monotonic_ns()
        stale_limit = (self.radar_communication_watchdog_ms
                       if event.reason == "radar_reports_no_target"
                       else self.target_stale_ms)
        if event.temporal_status(
                publish_ns, stale_limit, check_capture=True) != "fresh":
            return self._last_result
        with self._lock:
            if self._radar_event and event.sequence <= self._radar_event.sequence:
                return self._last_result
            self._radar_event = event
        if fast and event.usable and event.level == 2:
            self.state.request(2, reason=event.reason, source="radar-fast",
                               sequence=event.sequence,
                               now_ns=event.completed_monotonic_ns)
        return self.refresh(publish_ns)

    def publish_vision(self, event: ModalityEvent, *,
                       now_ns: Optional[int] = None) -> ArbitrationResult:
        publish_ns = now_ns or time.monotonic_ns()
        if event.temporal_status(
                publish_ns, self.vision_stale_ms,
                check_capture=True) != "fresh":
            return self._last_result
        with self._lock:
            if (self._vision_event
                    and event.capture_monotonic_ns <= self._vision_event.capture_monotonic_ns):
                return self._last_result
            self._vision_event = event
        return self.refresh(publish_ns)

    def publish_gps(self, event: ModalityEvent, *,
                    now_ns: Optional[int] = None) -> ArbitrationResult:
        publish_ns = now_ns or time.monotonic_ns()
        if event.temporal_status(
                publish_ns, self.gps_stale_ms,
                check_capture=True) != "fresh":
            return self._last_result
        with self._lock:
            if (self._gps_event
                    and event.capture_monotonic_ns <= self._gps_event.capture_monotonic_ns):
                return self._last_result
            self._gps_event = event
        return self.refresh(publish_ns)

    def publish_imu(self, event: ModalityEvent, *,
                    now_ns: Optional[int] = None) -> ArbitrationResult:
        publish_ns = now_ns or time.monotonic_ns()
        if event.temporal_status(
                publish_ns, self.imu_stale_ms,
                check_capture=True) != "fresh":
            return self._last_result
        with self._lock:
            if (self._imu_event
                    and event.capture_monotonic_ns <= self._imu_event.capture_monotonic_ns):
                return self._last_result
            self._imu_event = event
        return self.refresh(publish_ns)

    def refresh(self, now_ns: Optional[int] = None) -> ArbitrationResult:
        now_ns = now_ns or time.monotonic_ns()
        with self._lock:
            radar, vision, gps, imu = (
                self._radar_event, self._vision_event,
                self._gps_event, self._imu_event)
        result = arbitrate_warning_events(
            radar, vision, gps, imu, now_ns=now_ns,
            target_stale_ms=self.target_stale_ms,
            vision_stale_ms=self.vision_stale_ms,
            gps_stale_ms=self.gps_stale_ms,
            imu_stale_ms=self.imu_stale_ms,
            imu_enabled=self.imu_enabled,
            gps_modifier_config=self.gps_modifier_config,
            radar_communication_watchdog_ms=self.radar_communication_watchdog_ms,
        )
        with self._lock:
            self._arbiter_sequence += 1
            sequence = self._arbiter_sequence
        self.state.request(result.final_level, reason=result.warning_reason,
                           source="arbiter", sequence=sequence, now_ns=now_ns)
        state = self.state.snapshot()
        raw_source_timestamps = {}
        for source, event, level in (
            ("radar", radar, result.radar_level),
            ("vision", vision, result.vision_level),
            ("gps", gps, 0 if result.gps_status == "usable" else None),
            ("imu", imu, result.imu_level),
        ):
            if event is not None and level is not None:
                raw_source_timestamps[source] = {
                    "capture_monotonic_ns": event.capture_monotonic_ns,
                    "completed_monotonic_ns": event.completed_monotonic_ns,
                }
        with self._lock:
            self._last_result = result
            self._raw_source_timestamps = raw_source_timestamps
            if state.current_level is None:
                self._effective_risk_score = None
                self._effective_warning_reason = result.warning_reason
                self._effective_evidence_sources = ()
                self._effective_both_modalities_active = False
                self._risk_score_state = "unknown"
                self._effective_updated_monotonic_ns = now_ns
                self._effective_source_timestamps = {}
            elif state.current_level == result.final_level:
                normalized = normalize_risk_score(state.current_level, result.risk_score)
                self._effective_risk_score = normalized.score
                self._effective_warning_reason = result.warning_reason
                self._effective_evidence_sources = result.evidence_sources
                self._effective_both_modalities_active = result.both_modalities_active
                self._risk_score_state = "current"
                self._effective_updated_monotonic_ns = now_ns
                self._effective_source_timestamps = dict(raw_source_timestamps)
            else:
                normalized = normalize_risk_score(
                    state.current_level, self._effective_risk_score)
                self._effective_risk_score = normalized.score
                self._risk_score_state = "downgrade_held"
        return result

    def snapshot(self, now_ns: Optional[int] = None) -> dict:
        result = self.refresh(now_ns)
        with self._lock:
            radar, vision, gps, imu = (
                self._radar_event, self._vision_event,
                self._gps_event, self._imu_event)
            effective_score = self._effective_risk_score
            effective_reason = self._effective_warning_reason
            effective_evidence = self._effective_evidence_sources
            effective_both_active = self._effective_both_modalities_active
            risk_score_state = self._risk_score_state
            effective_updated_ns = self._effective_updated_monotonic_ns
            effective_source_timestamps = dict(self._effective_source_timestamps)
            raw_source_timestamps = dict(self._raw_source_timestamps)
        state = self.state.snapshot()

        def source_timing(source_timestamps: dict[str, dict[str, int]]) -> dict:
            timing = {}
            for source, stamps in source_timestamps.items():
                capture_ns = int(stamps["capture_monotonic_ns"])
                completed_ns = int(stamps["completed_monotonic_ns"])
                timing[source] = {
                    **stamps,
                    "capture_age_ms": (result.decision_monotonic_ns - capture_ns)
                    / 1_000_000.0,
                    "completion_age_ms": (result.decision_monotonic_ns - completed_ns)
                    / 1_000_000.0,
                }
            return timing

        return {
            **asdict(result),
            "raw_final_level": result.final_level,
            "raw_risk_score": result.risk_score,
            "raw_warning_reason": result.warning_reason,
            "raw_evidence_sources": result.evidence_sources,
            "final_level": state.current_level,
            "risk_score": effective_score,
            "warning_reason": effective_reason,
            "evidence_sources": effective_evidence,
            "both_modalities_active": effective_both_active,
            "risk_score_state": risk_score_state,
            "risk_decision_monotonic_ns": result.decision_monotonic_ns,
            "risk_effective_updated_monotonic_ns": effective_updated_ns,
            "risk_source_timing": source_timing(effective_source_timestamps),
            "raw_risk_source_timing": source_timing(raw_source_timestamps),
            "risk_timestamp_alignment": (
                "downgrade_held" if risk_score_state == "downgrade_held"
                else "as_of_latest_fresh" if risk_score_state == "current"
                else "no_usable_warning_modality"),
            "warning_rule_config": dict(self.rule_config_metadata),
            "warning_level": state.current_level,
            "last_known_level": state.last_known_level,
            "pending_lower_level": state.pending_lower_level,
            "last_motor_level": state.last_motor_level,
            "radar_event": radar,
            "vision_event": vision,
            "gps_event": gps,
            "imu_event": imu,
        }
