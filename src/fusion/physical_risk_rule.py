"""Evidence-based, radar-primary warning rule used by the live demo.

No learned weights are used.  A target must be approaching and lie inside a
physical lateral corridor.  Medium means an approaching in-corridor target is
present.  High means its TTC no longer exceeds the 2.5 s rider
perception/brake-reaction allowance plus processing time already consumed by
the current sample.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PhysicalRiskDecision:
    level: int
    label: str
    reason: str
    min_path_ttc_s: Optional[float]
    high_boundary_s: float
    path_target_count: int
    corridor_half_width_m: float


class PhysicalRiskRule:
    def __init__(
        self,
        *,
        body_width_m: float,
        radar_lateral_error_m: float,
        mounting_offset_m: float = 0.0,
        mounting_uncertainty_m: float = 0.06,
        reaction_time_s: float = 2.5,
        min_closing_speed_mps: float = 0.05,
    ) -> None:
        if body_width_m <= 0:
            raise ValueError("body_width_m must be positive")
        if radar_lateral_error_m < 0 or mounting_uncertainty_m < 0:
            raise ValueError("error/uncertainty values must be non-negative")
        self.body_width_m = body_width_m
        self.radar_lateral_error_m = radar_lateral_error_m
        self.mounting_offset_m = mounting_offset_m
        self.mounting_uncertainty_m = mounting_uncertainty_m
        self.reaction_time_s = reaction_time_s
        self.min_closing_speed_mps = min_closing_speed_mps

    @property
    def corridor_half_width_m(self) -> float:
        return (
            self.body_width_m / 2.0
            + self.radar_lateral_error_m
            + self.mounting_uncertainty_m
        )

    def decide(self, radar: Any, processing_elapsed_s: float) -> PhysicalRiskDecision:
        boundary = self.reaction_time_s + max(0.0, processing_elapsed_s)
        if radar is None or not bool(getattr(radar, "valid", False)):
            return PhysicalRiskDecision(
                0, "unknown", "radar_invalid", None, boundary, 0,
                self.corridor_half_width_m,
            )

        path_ttcs: list[float] = []
        for target in getattr(radar, "targets", []):
            distance = float(getattr(target, "distance_m", -1.0))
            relative_speed = float(getattr(target, "relative_speed_mps", 0.0))
            angle_deg = float(getattr(target, "angle_deg", 0.0))
            closing_speed = -relative_speed  # project convention: negative means approaching
            if distance <= 0 or closing_speed < self.min_closing_speed_mps:
                continue
            lateral_from_radar = distance * math.sin(math.radians(angle_deg))
            # mounting_offset_m is the radar origin in the bicycle frame
            # (right positive); 5.5 cm left is represented as -0.055.
            lateral_from_bike_axis = self.mounting_offset_m + lateral_from_radar
            if abs(lateral_from_bike_axis) > self.corridor_half_width_m:
                continue
            path_ttcs.append(distance / closing_speed)

        if not path_ttcs:
            return PhysicalRiskDecision(
                0, "low", "no_approaching_path_target", None, boundary, 0,
                self.corridor_half_width_m,
            )

        min_ttc = min(path_ttcs)
        if min_ttc <= boundary:
            return PhysicalRiskDecision(
                2, "high", "ttc_within_reaction_and_processing_time",
                min_ttc, boundary, len(path_ttcs), self.corridor_half_width_m,
            )
        return PhysicalRiskDecision(
            1, "mid", "approaching_path_target",
            min_ttc, boundary, len(path_ttcs), self.corridor_half_width_m,
        )
