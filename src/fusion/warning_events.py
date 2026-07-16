"""Shared event types for the competition warning channels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ModalityEvent:
    source: str
    source_id: str
    sequence: int
    capture_monotonic_ns: int
    completed_monotonic_ns: int
    usable: bool
    level: Optional[int]
    reason: str
    risk_score: Optional[float] = None
    status: str = "usable"
    details: dict[str, Any] = field(default_factory=dict)

    def temporal_status(self, now_ns: int, stale_ms: float, *,
                        check_capture: bool = True) -> str:
        limit_ns = int(stale_ms * 1_000_000)
        if now_ns <= 0 or limit_ns < 0:
            return "invalid_time_reference"
        if self.capture_monotonic_ns <= 0 or self.completed_monotonic_ns <= 0:
            return "invalid_timestamp"
        if self.completed_monotonic_ns < self.capture_monotonic_ns:
            return "invalid_timestamp_order"
        if (self.capture_monotonic_ns > now_ns
                or self.completed_monotonic_ns > now_ns):
            return "future_timestamp"
        if now_ns - self.completed_monotonic_ns > limit_ns:
            return "stale"
        if check_capture and now_ns - self.capture_monotonic_ns > limit_ns:
            return "stale"
        return "fresh"

    def fresh(self, now_ns: int, stale_ms: float, *, check_capture: bool = True) -> bool:
        if not self.usable or self.level not in (0, 1, 2):
            return False
        return self.temporal_status(
            now_ns, stale_ms, check_capture=check_capture) == "fresh"


@dataclass(frozen=True)
class ArbitrationResult:
    decision_monotonic_ns: int
    final_level: Optional[int]
    risk_score: Optional[float]
    system_status: str
    warning_reason: str
    evidence_sources: tuple[str, ...]
    both_modalities_active: bool
    radar_level: Optional[int]
    vision_level: Optional[int]
    imu_level: Optional[int]
    radar_score: Optional[float]
    vision_score: Optional[float]
    imu_score: Optional[float]
    radar_score_status: str
    vision_score_status: str
    imu_score_status: str
    radar_status: str
    vision_status: str
    imu_status: str
    gps_status: str
    gps_speed_kmh: Optional[float]
    gps_speed_factor: float
    vision_proximity_score: Optional[float]
    vision_proximity_adjusted_score: Optional[float]

