"""Thread-safe warning state machine with immediate upgrades and held downgrades."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WarningStateSnapshot:
    current_level: Optional[int]
    last_known_level: Optional[int]
    pending_lower_level: Optional[int]
    downgrade_started_ns: Optional[int]
    last_motor_level: Optional[int]
    last_reason: str


class WarningState:
    def __init__(self, motor=None, *, release_hold_ms: float = 500.0) -> None:
        self.motor = motor
        self.release_hold_ns = int(release_hold_ms * 1_000_000)
        self._state_lock = threading.Lock()
        self._motor_lock = threading.Lock()
        self.current_level: Optional[int] = None
        self.last_known_level: Optional[int] = None
        self.pending_lower_level: Optional[int] = None
        self.downgrade_started_ns: Optional[int] = None
        self.last_motor_level: Optional[int] = None
        self.last_reason = "initializing"
        self._processed_sources: dict[str, int] = {}

    def request(self, level: Optional[int], *, reason: str, source: str,
                sequence: int, now_ns: Optional[int] = None) -> bool:
        """Apply one event. Returns True only when a motor action was requested."""
        now_ns = now_ns or time.monotonic_ns()
        motor_level: Optional[int] = None
        with self._state_lock:
            if sequence > 0 and sequence <= self._processed_sources.get(source, -1):
                return False
            if sequence > 0:
                self._processed_sources[source] = sequence
            self.last_reason = reason

            if level is None:
                if self.current_level is not None:
                    self.last_known_level = self.current_level
                self.current_level = None
                self.pending_lower_level = None
                self.downgrade_started_ns = None
                return False

            if self.current_level is None:
                self.current_level = level
                self.last_known_level = level
                self.pending_lower_level = None
                self.downgrade_started_ns = None
                motor_level = level if level in (1, 2) else None
            elif level > self.current_level:
                self.current_level = level
                self.last_known_level = level
                self.pending_lower_level = None
                self.downgrade_started_ns = None
                motor_level = level
            elif level == self.current_level:
                self.pending_lower_level = None
                self.downgrade_started_ns = None
                return False
            else:
                if self.pending_lower_level != level:
                    self.pending_lower_level = level
                    self.downgrade_started_ns = now_ns
                    return False
                if (self.downgrade_started_ns is None
                        or now_ns - self.downgrade_started_ns < self.release_hold_ns):
                    return False
                self.current_level = level
                self.last_known_level = level
                self.pending_lower_level = None
                self.downgrade_started_ns = None
                # A low command stops/clears any controller-side warning state.
                motor_level = 0 if level == 0 else None

        if motor_level is not None:
            self._dispatch_motor(motor_level)
            return True
        return False

    def _dispatch_motor(self, level: int) -> None:
        if self.motor is None:
            return
        with self._motor_lock:
            if level == 0:
                self.motor.alert_low()
            elif level == 1:
                self.motor.alert_medium()
            else:
                self.motor.alert_high()
            with self._state_lock:
                self.last_motor_level = level

    def snapshot(self) -> WarningStateSnapshot:
        with self._state_lock:
            return WarningStateSnapshot(
                current_level=self.current_level,
                last_known_level=self.last_known_level,
                pending_lower_level=self.pending_lower_level,
                downgrade_started_ns=self.downgrade_started_ns,
                last_motor_level=self.last_motor_level,
                last_reason=self.last_reason,
            )

