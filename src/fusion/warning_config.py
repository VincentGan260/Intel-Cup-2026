"""Versioned loader for the competition warning-rule configuration."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.fusion.risk_score_contract import ATTENTION_SCORE, HIGH_SCORE


@dataclass(frozen=True)
class WarningRuleConfig:
    path: Path
    data: dict[str, Any]
    sha256: str

    @property
    def version(self) -> str:
        return str(self.data["version"])

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.data["schema_version"]),
            "version": self.version,
            "calibration_status": str(self.data["calibration_status"]),
            "sha256": self.sha256,
            "path": self.path.as_posix(),
        }

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data[name])


def _positive(section: dict, names: tuple[str, ...]) -> None:
    for name in names:
        if float(section[name]) <= 0.0:
            raise ValueError(f"warning config {name} must be positive")


def load_warning_rule_config(path: str | Path) -> WarningRuleConfig:
    import yaml

    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("warning config root must be a mapping")
    required = {"schema_version", "version", "calibration_status", "score_contract",
                "radar", "vision", "gps", "imu", "freshness", "state"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("warning config missing sections: " + ", ".join(missing))
    if int(data["schema_version"]) != 1:
        raise ValueError("unsupported warning config schema version")

    score = data["score_contract"]
    if float(score["attention_boundary"]) != ATTENTION_SCORE:
        raise ValueError("warning config attention boundary disagrees with score contract")
    if float(score["high_boundary"]) != HIGH_SCORE:
        raise ValueError("warning config high boundary disagrees with score contract")

    radar = data["radar"]
    _positive(radar, ("body_width_m", "mounting_uncertainty_m",
                      "radar_parsed_to_motor_go_p95_ms", "max_abs_angle_deg",
                      "attention_reference_s", "urgent_reference_s"))
    if float(radar["attention_reference_s"]) <= float(radar["urgent_reference_s"]):
        raise ValueError("radar attention reference must exceed urgent reference")
    if float(radar["point_gate_lateral_margin_m"]) < 0.0:
        raise ValueError("radar point-gate margin must be non-negative")
    warning_range = radar.get("configured_warning_range_m")
    if warning_range is not None and float(warning_range) <= 0.0:
        raise ValueError("configured warning range must be positive or null")

    vision = data["vision"]
    if vision["path_policy"] not in {"any", "center", "two_of_three"}:
        raise ValueError("unsupported visual path policy")
    _positive(vision, ("corridor_top_width_ratio", "corridor_bottom_width_ratio",
                       "near_bottom_ratio", "very_near_bottom_ratio",
                       "attention_tau_s", "urgent_tau_s", "temporal_window_s",
                       "min_history_s", "min_observations", "track_iou_threshold"))
    if not (0.0 <= float(vision["corridor_top_y_ratio"]) < 1.0):
        raise ValueError("vision corridor top must be within [0, 1)")

    gps = data["gps"]
    if not (0.0 <= float(gps["neutral_below_kmh"])
            < float(gps["full_effect_kmh"])):
        raise ValueError("GPS speed modifier range is invalid")
    if float(gps["max_factor"]) < 1.0:
        raise ValueError("GPS maximum factor must be at least one")

    imu = data["imu"]
    for name in ("roll_offset_deg", "pitch_offset_deg"):
        try:
            value = float(imu[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"warning config {name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"warning config {name} must be finite")
    _positive(imu, (
        "gravity_mps2", "min_turn_compensation_speed_kmh",
        "attention_error_deg", "critical_error_deg",
        "attention_outward_rate_deg_s", "urgent_outward_rate_deg_s",
        "attention_persistence_ms", "prediction_horizon_s",
        "urgent_min_error_deg", "urgent_consistent_samples",
        "max_sample_gap_ms",
    ))
    if float(imu["critical_error_deg"]) <= float(imu["attention_error_deg"]):
        raise ValueError("IMU critical residual must exceed attention residual")
    if float(imu["urgent_outward_rate_deg_s"]) < float(
            imu["attention_outward_rate_deg_s"]):
        raise ValueError("IMU urgent outward rate must not be below attention rate")
    if float(imu["urgent_min_error_deg"]) > float(imu["critical_error_deg"]):
        raise ValueError("IMU urgent minimum residual must not exceed critical residual")
    if float(imu["turn_sign"]) not in (-1.0, 1.0):
        raise ValueError("IMU turn sign must be +1 or -1")
    if int(imu["urgent_consistent_samples"]) != float(
            imu["urgent_consistent_samples"]):
        raise ValueError("IMU urgent consistent samples must be an integer")

    freshness = data["freshness"]
    _positive(freshness, ("target_stale_ms", "vision_stale_ms", "gps_stale_ms",
                          "imu_stale_ms",
                          "radar_communication_watchdog_ms"))
    _positive(data["state"], ("release_hold_ms",))
    return WarningRuleConfig(
        path=resolved, data=data, sha256=hashlib.sha256(raw).hexdigest())
