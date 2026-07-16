"""Competition radar TTC urgency rule (no learned weights or braking claims)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from src.fusion.risk_score_contract import (
    ATTENTION_SCORE as SHARED_ATTENTION_SCORE,
    HIGH_SCORE as SHARED_HIGH_SCORE,
)
from src.fusion.warning_events import ModalityEvent


@dataclass(frozen=True)
class PhysicalRiskDecision:
    level: int
    risk_score: Optional[float]
    label: str
    status: str
    reason: str
    min_path_ttc_s: Optional[float]
    attention_ttc_s: float
    urgent_ttc_s: float
    path_target_count: int
    point_gate_half_width_m: float
    raw_target_count: int = 0
    valid_target_count: int = 0
    invalid_target_count: int = 0
    critical_distance_m: Optional[float] = None
    critical_lateral_m: Optional[float] = None

    @property
    def high_boundary_s(self) -> float:
        return self.urgent_ttc_s

    @property
    def corridor_half_width_m(self) -> float:
        return self.point_gate_half_width_m


class PhysicalRiskRule:
    """Map valid radar targets to low / attention / urgent levels.

    `urgent_ttc_s` is a reaction-time urgency reference. It is explicitly not
    a stopping-safety threshold and contains no braking-time model.
    """

    ATTENTION_SCORE = SHARED_ATTENTION_SCORE
    HIGH_SCORE = SHARED_HIGH_SCORE

    def __init__(
        self,
        *,
        body_width_m: float,
        point_gate_lateral_margin_m: float,
        mounting_offset_m: float,
        mounting_uncertainty_m: float,
        configured_warning_range_m: float,
        radar_to_motor_p95_s: float | None = None,
        radar_parsed_to_motor_go_p95_s: float | None = None,
        attention_reference_s: float = 4.0,
        urgent_reference_s: float = 2.5,
        min_valid_distance_m: float = 0.01,
        max_valid_distance_m: float = 100.0,
        max_abs_angle_deg: float = 15.0,
        min_confidence: float | None = None,
    ) -> None:
        positive = (body_width_m, configured_warning_range_m,
                    attention_reference_s, urgent_reference_s)
        parsed_latency = (radar_parsed_to_motor_go_p95_s
                          if radar_parsed_to_motor_go_p95_s is not None
                          else radar_to_motor_p95_s)
        if parsed_latency is None:
            raise ValueError("parsed-to-motor GO P95 latency is required")
        nonnegative = (point_gate_lateral_margin_m, mounting_uncertainty_m,
                       parsed_latency, min_valid_distance_m)
        if any(value <= 0 for value in positive):
            raise ValueError("body width, configured range and TTC references must be positive")
        if attention_reference_s <= urgent_reference_s:
            raise ValueError("attention reference must be greater than urgency reference")
        if any(value < 0 for value in nonnegative):
            raise ValueError("margins, noise, latency and minimum distance must be non-negative")
        if max_valid_distance_m < configured_warning_range_m:
            raise ValueError("configured warning range cannot exceed radar valid distance")
        if max_abs_angle_deg <= 0 or max_abs_angle_deg > 15.0:
            raise ValueError("LD2451 competition gate must stay within the manual's +/-15 deg FOV")

        self.body_width_m = body_width_m
        self.point_gate_lateral_margin_m = point_gate_lateral_margin_m
        self.mounting_offset_m = mounting_offset_m
        self.mounting_uncertainty_m = mounting_uncertainty_m
        self.configured_warning_range_m = configured_warning_range_m
        self.radar_parsed_to_motor_go_p95_s = parsed_latency
        self.attention_reference_s = attention_reference_s
        self.urgent_reference_s = urgent_reference_s
        self.min_valid_distance_m = min_valid_distance_m
        self.max_valid_distance_m = max_valid_distance_m
        self.max_abs_angle_deg = max_abs_angle_deg
        self.min_confidence = min_confidence

    @property
    def urgent_ttc_s(self) -> float:
        return self.urgent_reference_s + self.radar_parsed_to_motor_go_p95_s

    @property
    def attention_ttc_s(self) -> float:
        return self.attention_reference_s + self.radar_parsed_to_motor_go_p95_s

    @property
    def radar_to_motor_p95_s(self) -> float:
        """Backward-compatible alias; the actual start is frame parse completion."""
        return self.radar_parsed_to_motor_go_p95_s

    @property
    def point_gate_half_width_m(self) -> float:
        return (self.body_width_m / 2.0 + self.point_gate_lateral_margin_m
                + self.mounting_uncertainty_m)

    @property
    def corridor_half_width_m(self) -> float:
        return self.point_gate_half_width_m

    def _candidate_risk_score(self, ttc_s: float, distance_m: float) -> float:
        """Map one path candidate to the shared intervention-urgency scale."""
        if ttc_s <= self.urgent_ttc_s:
            urgency = 1.0 - ttc_s / self.urgent_ttc_s
            return min(1.0, self.HIGH_SCORE + (1.0 - self.HIGH_SCORE) * urgency)

        if ttc_s <= self.attention_ttc_s:
            span = self.attention_ttc_s - self.urgent_ttc_s
            progress = (self.attention_ttc_s - ttc_s) / span
            return self.ATTENTION_SCORE + (self.HIGH_SCORE - self.ATTENTION_SCORE) * progress

        return self.ATTENTION_SCORE * self.attention_ttc_s / ttc_s

    def _make(self, level: int, label: str, status: str, reason: str, *,
              raw: int = 0, valid: int = 0, invalid: int = 0,
              candidates: list[tuple[float, float, float]] | None = None) -> PhysicalRiskDecision:
        candidates = candidates or []
        critical = min(candidates, key=lambda item: item[0]) if candidates else None
        usable = status not in {"unknown", "degraded"}
        risk_score = (max(self._candidate_risk_score(ttc, distance)
                          for ttc, distance, _lateral in candidates)
                      if candidates else 0.0 if usable else None)
        return PhysicalRiskDecision(
            level=level, risk_score=risk_score, label=label, status=status, reason=reason,
            min_path_ttc_s=critical[0] if critical else None,
            attention_ttc_s=self.attention_ttc_s,
            urgent_ttc_s=self.urgent_ttc_s,
            path_target_count=len(candidates),
            point_gate_half_width_m=self.point_gate_half_width_m,
            raw_target_count=raw, valid_target_count=valid,
            invalid_target_count=invalid,
            critical_distance_m=critical[1] if critical else None,
            critical_lateral_m=critical[2] if critical else None,
        )

    def decide(self, radar: Any, *, radar_fresh: bool) -> PhysicalRiskDecision:
        if radar is None or not bool(getattr(radar, "valid", False)):
            return self._make(0, "unknown", "unknown", "radar_frame_invalid")
        if not radar_fresh:
            return self._make(0, "unknown", "unknown", "radar_frame_stale")

        targets = list(getattr(radar, "targets", []) or [])
        if not targets:
            return self._make(0, "low", "normal", "radar_reports_no_target")

        valid_count = 0
        invalid_count = 0
        candidates: list[tuple[float, float, float]] = []
        for target in targets:
            try:
                distance = float(getattr(target, "distance_m"))
                relative_speed = float(getattr(target, "relative_speed_mps"))
                angle_deg = float(getattr(target, "angle_deg"))
                confidence = float(getattr(target, "confidence", 1.0))
            except (AttributeError, TypeError, ValueError):
                invalid_count += 1
                continue

            valid_measurement = (
                math.isfinite(distance) and math.isfinite(relative_speed)
                and math.isfinite(angle_deg) and math.isfinite(confidence)
                and self.min_valid_distance_m <= distance <= self.max_valid_distance_m
                and abs(angle_deg) <= self.max_abs_angle_deg
                and (self.min_confidence is None or confidence >= self.min_confidence)
            )
            if not valid_measurement:
                invalid_count += 1
                continue
            valid_count += 1

            # A valid target outside the configured competition working range
            # is not abnormal; this software configuration cannot alert on it.
            if distance > self.configured_warning_range_m:
                continue

            closing_speed = -relative_speed
            # Mathematical validity condition for TTC, not an empirical speed
            # threshold. LD2451's configured minimum detection speed owns the
            # physical filtering.
            if not math.isfinite(closing_speed) or closing_speed <= 0.0:
                continue
            lateral_m = self.mounting_offset_m + distance * math.sin(math.radians(angle_deg))
            if abs(lateral_m) > self.point_gate_half_width_m:
                continue
            candidates.append((distance / closing_speed, distance, lateral_m))

        if valid_count == 0:
            return self._make(
                0, "unknown", "degraded", "all_reported_targets_invalid",
                raw=len(targets), invalid=invalid_count,
            )
        if not candidates:
            return self._make(
                0, "low", "normal", "no_approaching_target_in_point_gate",
                raw=len(targets), valid=valid_count, invalid=invalid_count,
            )

        if min(item[0] for item in candidates) <= self.urgent_ttc_s:
            return self._make(
                2, "high", "urgent", "ttc_entered_reaction_time_urgency_reference",
                raw=len(targets), valid=valid_count, invalid=invalid_count,
                candidates=candidates,
            )
        if min(item[0] for item in candidates) <= self.attention_ttc_s:
            return self._make(
                1, "mid", "warning", "ttc_entered_attention_reference",
                raw=len(targets), valid=valid_count, invalid=invalid_count,
                candidates=candidates,
            )
        return self._make(
            0, "low", "normal", "approaching_target_beyond_attention_reference",
            raw=len(targets), valid=valid_count, invalid=invalid_count,
            candidates=candidates,
        )

    def evaluate_event(self, radar: Any, *, radar_fresh: bool, sequence: int,
                       packet_monotonic_ns: int,
                       completed_monotonic_ns: int | None = None) -> ModalityEvent:
        completed_ns = completed_monotonic_ns or __import__("time").monotonic_ns()
        decision = self.decide(radar, radar_fresh=radar_fresh)
        usable = decision.status not in {"unknown", "degraded"}
        level = decision.level if usable else None
        return ModalityEvent(
            source="radar", source_id=str(sequence), sequence=sequence,
            capture_monotonic_ns=packet_monotonic_ns,
            completed_monotonic_ns=completed_ns,
            usable=usable, level=level, reason=decision.reason,
            risk_score=decision.risk_score if usable else None,
            status=("usable" if usable else decision.status),
            details={
                "risk_score_semantics": "intervention_urgency_not_probability",
                "critical_ttc_s": decision.min_path_ttc_s,
                "attention_ttc_s": decision.attention_ttc_s,
                "urgent_ttc_s": decision.urgent_ttc_s,
                "critical_distance_m": decision.critical_distance_m,
                "critical_lateral_m": decision.critical_lateral_m,
                "point_gate_half_width_m": decision.point_gate_half_width_m,
                "path_target_count": decision.path_target_count,
                "raw_target_count": decision.raw_target_count,
                "valid_target_count": decision.valid_target_count,
                "invalid_target_count": decision.invalid_target_count,
            },
        )
