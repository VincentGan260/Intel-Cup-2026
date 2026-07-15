"""Thread-safe handoff of frame-aligned asynchronous vision results."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from src.fusion.data_types import VisionData
from src.vision.common.types import VisionResult


@dataclass(frozen=True)
class VisionSnapshot:
    source_frame_id: int
    source_frame: np.ndarray
    frame_capture_monotonic_ns: int
    vision_start_monotonic_ns: int
    vision_finish_monotonic_ns: int
    vision_data: VisionData
    vision_result: VisionResult | None

    @property
    def inference_ms(self) -> float:
        return (
            self.vision_finish_monotonic_ns - self.vision_start_monotonic_ns
        ) / 1_000_000.0


class VisionSnapshotStore:
    """Keep only the newest complete snapshot; consumers track its version."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: VisionSnapshot | None = None
        self._version = 0

    def publish(self, snapshot: VisionSnapshot) -> int:
        with self._lock:
            self._snapshot = snapshot
            self._version += 1
            return self._version

    def get_snapshot(self) -> tuple[VisionSnapshot | None, int]:
        with self._lock:
            return self._snapshot, self._version
