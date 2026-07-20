"""Visual path warning rule with calibrated-later corridor and looming evidence."""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from src.fusion.risk_score_contract import (
    ATTENTION_SCORE as SHARED_ATTENTION_SCORE,
    HIGH_SCORE as SHARED_HIGH_SCORE,
)
from src.fusion.warning_events import ModalityEvent


class VisionWarningRule:
    VALID_PATH_POLICIES = {"any", "center", "two_of_three"}
    ATTENTION_SCORE = SHARED_ATTENTION_SCORE
    HIGH_SCORE = SHARED_HIGH_SCORE

    def __init__(self, *, path_policy: str = "center",
                 corridor_top_y_ratio: float = 0.40,
                 corridor_top_width_ratio: float = 0.10,
                 corridor_bottom_width_ratio: float = 0.50,
                 near_bottom_ratio: float = 0.61,
                 very_near_bottom_ratio: float = 0.82,
                 attention_tau_s: float = 4.0,
                 urgent_tau_s: float = 2.5,
                 temporal_window_s: float = 0.5,
                 min_history_s: float = 0.2,
                 min_observations: int = 3,
                 track_iou_threshold: float = 0.30) -> None:
        if path_policy not in self.VALID_PATH_POLICIES:
            raise ValueError(
                "vision path policy must be one of: "
                + ", ".join(sorted(self.VALID_PATH_POLICIES))
            )
        if not 0.0 <= corridor_top_y_ratio < 1.0:
            raise ValueError("corridor top y ratio must be within [0, 1)")
        if not 0.0 < corridor_top_width_ratio <= corridor_bottom_width_ratio <= 1.0:
            raise ValueError("corridor widths must satisfy 0 < top <= bottom <= 1")
        if not 0.0 < near_bottom_ratio < very_near_bottom_ratio <= 1.0:
            raise ValueError("visual bottom ratios must satisfy 0 < near < very_near <= 1")
        if not 0.0 < urgent_tau_s < attention_tau_s:
            raise ValueError("visual tau references must satisfy 0 < urgent < attention")
        if temporal_window_s <= 0.0 or not 0.0 < min_history_s <= temporal_window_s:
            raise ValueError("visual history span must be positive and within the window")
        if min_observations < 2 or not 0.0 <= track_iou_threshold <= 1.0:
            raise ValueError("visual tracking parameters are invalid")
        self.path_policy = path_policy
        self.corridor_top_y_ratio = corridor_top_y_ratio
        self.corridor_top_width_ratio = corridor_top_width_ratio
        self.corridor_bottom_width_ratio = corridor_bottom_width_ratio
        self.near_bottom_ratio = near_bottom_ratio
        self.very_near_bottom_ratio = very_near_bottom_ratio
        self.attention_tau_s = attention_tau_s
        self.urgent_tau_s = urgent_tau_s
        self.temporal_window_ns = int(temporal_window_s * 1_000_000_000)
        self.min_history_ns = int(min_history_s * 1_000_000_000)
        self.min_observations = min_observations
        self.track_iou_threshold = track_iou_threshold
        self._tracks: list[dict[str, Any]] = []
        self._next_track_id = 1

    def reset(self) -> None:
        """Discard temporal tracks after the vision source becomes unavailable."""
        self._tracks.clear()
        self._next_track_id = 1

    def _inside_corridor(self, center_x: float, sample_y: int,
                         width: int, height: int) -> bool:
        top_y = self.corridor_top_y_ratio * (height - 1)
        bottom_y = float(height - 1)
        if sample_y < top_y or bottom_y <= top_y:
            return False
        progress = (sample_y - top_y) / (bottom_y - top_y)
        width_ratio = (self.corridor_top_width_ratio
                       + (self.corridor_bottom_width_ratio
                          - self.corridor_top_width_ratio) * progress)
        half_width = width * width_ratio / 2.0
        image_center = width / 2.0
        return image_center - half_width <= center_x <= image_center + half_width

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
        center_x = (x1 + x2) / 2.0
        if not self._inside_corridor(center_x, sample_y, width, height):
            return False
        sample_xs = (
            min(width - 1, max(0, int(x1))),
            min(width - 1, max(0, int((x1 + x2) / 2.0))),
            min(width - 1, max(0, int(x2))),
        )
        hits = [bool(drivable_mask[sample_y, x]) for x in sample_xs]
        if self.path_policy == "center":
            return hits[1]
        if self.path_policy == "two_of_three":
            return sum(hits) >= 2
        return any(hits)

    @staticmethod
    def _bbox_iou(left: tuple[float, float, float, float],
                  right: tuple[float, float, float, float]) -> float:
        ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
        ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _valid_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
        try:
            values = tuple(float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        if (len(values) != 4 or not all(math.isfinite(value) for value in values)
                or values[2] <= values[0] or values[3] <= values[1]):
            return None
        return values

    def _update_tracks(self, detections: list[Any], now_ns: int) -> dict[int, dict[str, Any]]:
        self._tracks = [track for track in self._tracks
                        if now_ns - track["last_seen_ns"] <= self.temporal_window_ns]
        assigned_tracks: set[int] = set()
        detection_tracks: dict[int, dict[str, Any]] = {}
        for index, detection in enumerate(detections):
            bbox = self._valid_bbox(getattr(detection, "bbox", None))
            if bbox is None:
                continue
            candidates = [track for track in self._tracks
                          if track["id"] not in assigned_tracks]
            track = max(candidates,
                        key=lambda item: self._bbox_iou(bbox, item["bbox"]),
                        default=None)
            if (track is None
                    or self._bbox_iou(bbox, track["bbox"]) < self.track_iou_threshold):
                track = {
                    "id": self._next_track_id,
                    "class_name": "obstacle",
                    "bbox": bbox,
                    "last_seen_ns": now_ns,
                    "history": deque(),
                }
                self._next_track_id += 1
                self._tracks.append(track)
            assigned_tracks.add(track["id"])
            track["bbox"] = bbox
            track["last_seen_ns"] = now_ns
            scale = math.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            history = track["history"]
            history.append((now_ns, scale))
            while history and now_ns - history[0][0] > self.temporal_window_ns:
                history.popleft()
            detection_tracks[index] = track
        return detection_tracks

    def _visual_tau_s(self, track: dict[str, Any]) -> float | None:
        history = track["history"]
        if len(history) < self.min_observations:
            return None
        first_ns, first_scale = history[0]
        last_ns, last_scale = history[-1]
        elapsed_ns = last_ns - first_ns
        if elapsed_ns < self.min_history_ns or first_scale <= 0.0 or last_scale <= first_scale:
            return None
        growth_ratio = last_scale / first_scale - 1.0
        return elapsed_ns / 1_000_000_000.0 / growth_ratio

    def _risk_score(self, path_count: int, bottom_ratio: float,
                    tau_s: float | None) -> tuple[float, float, float]:
        if path_count <= 0:
            return 0.0, 0.0, 0.0

        bottom_ratio = max(0.0, min(1.0, bottom_ratio))
        if bottom_ratio < self.near_bottom_ratio:
            span = max(1e-9, self.near_bottom_ratio - self.corridor_top_y_ratio)
            progress = max(0.0, (bottom_ratio - self.corridor_top_y_ratio) / span)
            proximity_score = self.ATTENTION_SCORE * min(1.0, progress)
        else:
            span = max(1e-9, self.very_near_bottom_ratio - self.near_bottom_ratio)
            progress = min(1.0, (bottom_ratio - self.near_bottom_ratio) / span)
            proximity_score = (self.ATTENTION_SCORE
                               + (self.HIGH_SCORE - self.ATTENTION_SCORE) * progress)
            proximity_score = min(self.HIGH_SCORE - 1e-6, proximity_score)

        tau_score = 0.0
        if tau_s is not None and tau_s > 0.0:
            if tau_s <= self.urgent_tau_s:
                urgency = 1.0 - tau_s / self.urgent_tau_s
                tau_score = self.HIGH_SCORE + (1.0 - self.HIGH_SCORE) * urgency
            elif tau_s <= self.attention_tau_s:
                span = self.attention_tau_s - self.urgent_tau_s
                progress = (self.attention_tau_s - tau_s) / span
                tau_score = (self.ATTENTION_SCORE
                             + (self.HIGH_SCORE - self.ATTENTION_SCORE) * progress)
            else:
                tau_score = self.ATTENTION_SCORE * self.attention_tau_s / tau_s

        return min(1.0, max(proximity_score, tau_score)), proximity_score, tau_score

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

        now_ns = capture_monotonic_ns or completed_monotonic_ns
        detection_tracks = self._update_tracks(detections, now_ns)
        path_indexes = [index for index, det in enumerate(detections)
                        if self.path_related(getattr(det, "bbox", None), mask)]
        path_count = len(path_indexes)
        min_tau_s = None
        max_bottom_ratio = 0.0
        for index in path_indexes:
            bbox = self._valid_bbox(getattr(detections[index], "bbox", None))
            if bbox is None:
                continue
            max_bottom_ratio = max(max_bottom_ratio, bbox[3] / mask.shape[0])
            track = detection_tracks.get(index)
            tau_s = self._visual_tau_s(track) if track is not None else None
            if tau_s is not None:
                min_tau_s = tau_s if min_tau_s is None else min(min_tau_s, tau_s)

        level = 0
        reason = "no_visual_path_obstacle"
        if min_tau_s is not None and min_tau_s <= self.urgent_tau_s:
            level, reason = 2, "visual_tau_entered_urgency_reference"
        elif (max_bottom_ratio >= self.near_bottom_ratio
              or min_tau_s is not None and min_tau_s <= self.attention_tau_s):
            level, reason = 1, "visual_path_obstacle_attention"
        elif path_count:
            reason = "visual_path_candidate_observing"

        risk_score, proximity_score, tau_score = self._risk_score(
            path_count, max_bottom_ratio, min_tau_s)

        details = {"risk_score_semantics": "intervention_urgency_not_probability",
                   "path_obstacle_count": path_count,
                                          "path_policy": self.path_policy,
                                          "corridor_top_y_ratio": self.corridor_top_y_ratio,
                                          "corridor_top_width_ratio": self.corridor_top_width_ratio,
                                          "corridor_bottom_width_ratio": self.corridor_bottom_width_ratio,
                   "near_bottom_ratio": self.near_bottom_ratio,
                   "very_near_bottom_ratio": self.very_near_bottom_ratio,
                   "visual_tau_s": min_tau_s,
                   "attention_tau_s": self.attention_tau_s,
                   "urgent_tau_s": self.urgent_tau_s,
                   "max_path_bottom_ratio": max_bottom_ratio,
                   "proximity_risk_score": proximity_score,
                   "tau_risk_score": tau_score,
                   "detection_count": len(detections)}
        return ModalityEvent(**base, usable=True, level=level, status="usable",
                             reason=reason, risk_score=risk_score, details=details)
