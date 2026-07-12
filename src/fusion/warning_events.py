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
    status: str = "usable"
    details: dict[str, Any] = field(default_factory=dict)

    def fresh(self, now_ns: int, stale_ms: float, *, check_capture: bool = True) -> bool:
        if not self.usable or self.level not in (0, 1, 2):
            return False
        limit_ns = int(stale_ms * 1_000_000)
        if self.completed_monotonic_ns <= 0:
            return False
        if now_ns - self.completed_monotonic_ns > limit_ns:
            return False
        return not (check_capture and (
            self.capture_monotonic_ns <= 0
            or now_ns - self.capture_monotonic_ns > limit_ns
        ))


@dataclass(frozen=True)
class ArbitrationResult:
    final_level: Optional[int]
    system_status: str
    warning_reason: str
    evidence_sources: tuple[str, ...]
    both_modalities_active: bool
    radar_level: Optional[int]
    vision_level: Optional[int]
    radar_status: str
    vision_status: str

