"""Deterministic fallback for one unavailable core risk sensor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.fusion.imu_warning_rule import ImuWarningRule, ImuWarningRuleConfig
from src.fusion.physical_risk_rule import PhysicalRiskRule
from src.fusion.vision_warning_rule import VisionWarningRule
from src.fusion.warning_config import load_warning_rule_config


CORE_SENSORS = ("radar", "vision", "imu")


@dataclass(frozen=True)
class DegradedRiskDecision:
    level: int | None
    risk_score: float | None
    reason: str
    evidence_sources: tuple[str, ...]
    missing_sensors: tuple[str, ...]
    modality_scores: dict[str, float | None]
    modality_levels: dict[str, int | None]
    modality_statuses: dict[str, str]

    def as_dict(self) -> dict:
        return asdict(self)


class SingleSensorDegradationController:
    """Reuse production physical rules when one core modality is unavailable."""

    def __init__(
        self,
        config_path: str | Path = "configs/warning_rules.yaml",
    ) -> None:
        config = load_warning_rule_config(config_path)
        radar = config.section("radar")
        vision = config.section("vision")
        imu = config.section("imu")
        self.config_metadata = config.metadata
        self.radar_rule = PhysicalRiskRule(
            body_width_m=float(radar["body_width_m"]),
            point_gate_lateral_margin_m=float(
                radar["point_gate_lateral_margin_m"]
            ),
            mounting_offset_m=float(radar["mounting_offset_m"]),
            mounting_uncertainty_m=float(radar["mounting_uncertainty_m"]),
            configured_warning_range_m=float(
                radar["configured_warning_range_m"]
            ),
            radar_parsed_to_motor_go_p95_s=(
                float(radar["radar_parsed_to_motor_go_p95_ms"]) / 1000.0
            ),
            attention_reference_s=float(radar["attention_reference_s"]),
            urgent_reference_s=float(radar["urgent_reference_s"]),
            max_abs_angle_deg=float(radar["max_abs_angle_deg"]),
        )
        self.vision_rule = VisionWarningRule(
            path_policy=str(vision["path_policy"]),
            corridor_top_y_ratio=float(vision["corridor_top_y_ratio"]),
            corridor_top_width_ratio=float(
                vision["corridor_top_width_ratio"]
            ),
            corridor_bottom_width_ratio=float(
                vision["corridor_bottom_width_ratio"]
            ),
            near_bottom_ratio=float(vision["near_bottom_ratio"]),
            very_near_bottom_ratio=float(vision["very_near_bottom_ratio"]),
            attention_tau_s=float(vision["attention_tau_s"]),
            urgent_tau_s=float(vision["urgent_tau_s"]),
            temporal_window_s=float(vision["temporal_window_s"]),
            min_history_s=float(vision["min_history_s"]),
            min_observations=int(vision["min_observations"]),
            track_iou_threshold=float(vision["track_iou_threshold"]),
        )
        self.imu_rule = ImuWarningRule(ImuWarningRuleConfig(
            calibration_status=str(imu["calibration_status"]),
            roll_offset_deg=float(imu["roll_offset_deg"]),
            pitch_offset_deg=float(imu["pitch_offset_deg"]),
            turn_sign=float(imu["turn_sign"]),
            gravity_mps2=float(imu["gravity_mps2"]),
            min_turn_compensation_speed_kmh=float(
                imu["min_turn_compensation_speed_kmh"]
            ),
            attention_error_deg=float(imu["attention_error_deg"]),
            critical_error_deg=float(imu["critical_error_deg"]),
            attention_outward_rate_deg_s=float(
                imu["attention_outward_rate_deg_s"]
            ),
            urgent_outward_rate_deg_s=float(
                imu["urgent_outward_rate_deg_s"]
            ),
            attention_persistence_ms=float(
                imu["attention_persistence_ms"]
            ),
            prediction_horizon_s=float(imu["prediction_horizon_s"]),
            urgent_min_error_deg=float(imu["urgent_min_error_deg"]),
            urgent_consistent_samples=int(imu["urgent_consistent_samples"]),
            max_sample_gap_ms=float(imu["max_sample_gap_ms"]),
        ))
        self._sequence = 0

    def evaluate(
        self,
        *,
        now_monotonic_ns: int,
        radar: Any,
        radar_usable: bool,
        vision_result: Any,
        vision_usable: bool,
        imu: Any,
        imu_usable: bool,
        gps: Any,
        gps_usable: bool,
    ) -> DegradedRiskDecision:
        self._sequence += 1
        sequence = self._sequence
        usable = {
            "radar": bool(radar_usable),
            "vision": bool(vision_usable),
            "imu": bool(imu_usable),
        }
        missing = tuple(name for name in CORE_SENSORS if not usable[name])
        scores: dict[str, float | None] = {
            name: None for name in CORE_SENSORS
        }
        levels: dict[str, int | None] = {
            name: None for name in CORE_SENSORS
        }
        statuses = {
            name: "unavailable" for name in CORE_SENSORS
        }
        reasons: dict[str, str] = {}

        if usable["radar"]:
            radar_decision = self.radar_rule.decide(
                radar, radar_fresh=True
            )
            if radar_decision.status not in {"unknown", "degraded"}:
                scores["radar"] = radar_decision.risk_score
                levels["radar"] = radar_decision.level
                statuses["radar"] = radar_decision.status
                reasons["radar"] = radar_decision.reason
            else:
                statuses["radar"] = radar_decision.status

        if usable["vision"] and vision_result is not None:
            vision_event = self.vision_rule.evaluate(
                vision_result,
                source_frame_id=sequence,
                capture_monotonic_ns=now_monotonic_ns,
                completed_monotonic_ns=now_monotonic_ns,
                sequence=sequence,
            )
            if vision_event.usable:
                scores["vision"] = vision_event.risk_score
                levels["vision"] = vision_event.level
                statuses["vision"] = vision_event.status
                reasons["vision"] = vision_event.reason
            else:
                statuses["vision"] = vision_event.status
        else:
            self.vision_rule.reset()

        if usable["imu"]:
            imu_event = self.imu_rule.evaluate_event(
                imu,
                capture_monotonic_ns=now_monotonic_ns,
                completed_monotonic_ns=now_monotonic_ns,
                sequence=sequence,
                gps_speed_kmh=(
                    float(getattr(gps, "speed_kmh", 0.0))
                    if gps_usable else None
                ),
                gps_usable=bool(gps_usable),
            )
            if imu_event.usable:
                scores["imu"] = imu_event.risk_score
                levels["imu"] = imu_event.level
                statuses["imu"] = imu_event.status
                reasons["imu"] = imu_event.reason
            else:
                statuses["imu"] = imu_event.status
        else:
            self.imu_rule.reset()

        candidates = [
            (levels[name], scores[name], name)
            for name in CORE_SENSORS
            if levels[name] in (0, 1, 2) and scores[name] is not None
        ]
        if not candidates:
            return DegradedRiskDecision(
                level=None,
                risk_score=None,
                reason="no_usable_fallback_modality",
                evidence_sources=(),
                missing_sensors=missing,
                modality_scores=scores,
                modality_levels=levels,
                modality_statuses=statuses,
            )

        level = max(int(item[0]) for item in candidates)
        risk_score = max(float(item[1]) for item in candidates)
        evidence = tuple(
            name for item_level, _score, name in candidates
            if int(item_level) > 0
        )
        dominant = max(
            candidates,
            key=lambda item: (int(item[0]), float(item[1])),
        )[2]
        return DegradedRiskDecision(
            level=level,
            risk_score=risk_score,
            reason=reasons.get(dominant, "single_sensor_fallback"),
            evidence_sources=evidence,
            missing_sensors=missing,
            modality_scores=scores,
            modality_levels=levels,
            modality_statuses=statuses,
        )
