"""Minimal visual path-obstacle rule; visual evidence never emits level 2."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.fusion.warning_events import ModalityEvent


class VisionWarningRule:
    def path_related(self, bbox: Any, drivable_mask: np.ndarray) -> bool:
        if drivable_mask is None or drivable_mask.ndim < 2 or drivable_mask.size == 0:
            return False
        height, width = drivable_mask.shape[:2]
        try:
            x1, _y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in (x1, x2, y2)):
            return False
        sample_y = min(height - 1, max(0, int(y2) + 1))
        sample_xs = (
            min(width - 1, max(0, int(x1))),
            min(width - 1, max(0, int((x1 + x2) / 2.0))),
            min(width - 1, max(0, int(x2))),
        )
        return any(bool(drivable_mask[sample_y, x]) for x in sample_xs)

    def evaluate(
        self,
        result: Any,
        *,
        source_frame_id: int,
        capture_monotonic_ns: int,
        completed_monotonic_ns: int,
        sequence: int,
    ) -> ModalityEvent:
        base = dict(
            source="vision", source_id=str(source_frame_id), sequence=sequence,
            capture_monotonic_ns=capture_monotonic_ns,
            completed_monotonic_ns=completed_monotonic_ns,
        )
        mask = getattr(result, "drivable_mask", None) if result is not None else None
        detections = list(getattr(result, "detections", []) or []) if result is not None else []
        if result is None or mask is None:
            return ModalityEvent(**base, usable=False, level=None,
                                 status="invalid", reason="vision_result_invalid")

        path_count = sum(self.path_related(getattr(det, "bbox", None), mask)
                         for det in detections)
        if path_count:
            return ModalityEvent(**base, usable=True, level=1, status="usable",
                                 reason="vision_path_obstacle",
                                 details={"path_obstacle_count": path_count,
                                          "detection_count": len(detections)})
        return ModalityEvent(**base, usable=True, level=0, status="usable",
                             reason="no_visual_path_obstacle",
                             details={"path_obstacle_count": 0,
                                      "detection_count": len(detections)})

