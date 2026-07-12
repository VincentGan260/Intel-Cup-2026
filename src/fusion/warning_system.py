"""Coordinator for asynchronous radar and visual warning events."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Optional

from src.fusion.warning_arbiter import arbitrate_warning_events
from src.fusion.warning_events import ArbitrationResult, ModalityEvent
from src.fusion.warning_state import WarningState


class MultimodalWarningSystem:
    def __init__(self, *, motor=None, target_stale_ms: float = 500.0,
                 vision_stale_ms: float = 500.0,
                 release_hold_ms: float = 500.0,
                 radar_communication_watchdog_ms: float = 2000.0) -> None:
        self.target_stale_ms = target_stale_ms
        self.vision_stale_ms = vision_stale_ms
        self.radar_communication_watchdog_ms = radar_communication_watchdog_ms
        self.state = WarningState(motor, release_hold_ms=release_hold_ms)
        self._lock = threading.Lock()
        self._radar_event: Optional[ModalityEvent] = None
        self._vision_event: Optional[ModalityEvent] = None
        self._arbiter_sequence = 0
        self._last_result = arbitrate_warning_events(
            None, None, now_ns=time.monotonic_ns(),
            target_stale_ms=target_stale_ms, vision_stale_ms=vision_stale_ms,
            radar_communication_watchdog_ms=radar_communication_watchdog_ms)

    def publish_radar(self, event: ModalityEvent, *, fast: bool = True) -> ArbitrationResult:
        with self._lock:
            if self._radar_event and event.sequence <= self._radar_event.sequence:
                return self._last_result
            self._radar_event = event
        if fast and event.usable and event.level == 2:
            self.state.request(2, reason=event.reason, source="radar-fast",
                               sequence=event.sequence,
                               now_ns=event.completed_monotonic_ns)
        return self.refresh()

    def publish_vision(self, event: ModalityEvent) -> ArbitrationResult:
        with self._lock:
            if (self._vision_event
                    and event.capture_monotonic_ns <= self._vision_event.capture_monotonic_ns):
                return self._last_result
            self._vision_event = event
        return self.refresh()

    def refresh(self, now_ns: Optional[int] = None) -> ArbitrationResult:
        now_ns = now_ns or time.monotonic_ns()
        with self._lock:
            radar, vision = self._radar_event, self._vision_event
        result = arbitrate_warning_events(
            radar, vision, now_ns=now_ns,
            target_stale_ms=self.target_stale_ms,
            vision_stale_ms=self.vision_stale_ms,
            radar_communication_watchdog_ms=self.radar_communication_watchdog_ms,
        )
        with self._lock:
            self._arbiter_sequence += 1
            sequence = self._arbiter_sequence
        self.state.request(result.final_level, reason=result.warning_reason,
                           source="arbiter", sequence=sequence, now_ns=now_ns)
        with self._lock:
            self._last_result = result
        return result

    def snapshot(self, now_ns: Optional[int] = None) -> dict:
        result = self.refresh(now_ns)
        with self._lock:
            radar, vision = self._radar_event, self._vision_event
        state = self.state.snapshot()
        return {
            **asdict(result),
            "warning_level": state.current_level,
            "last_known_level": state.last_known_level,
            "pending_lower_level": state.pending_lower_level,
            "last_motor_level": state.last_motor_level,
            "radar_event": radar,
            "vision_event": vision,
        }
