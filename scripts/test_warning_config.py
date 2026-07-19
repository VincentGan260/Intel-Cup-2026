"""Checks for the versioned warning-rule configuration."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.warning_config import load_warning_rule_config


def main() -> None:
    config = load_warning_rule_config(ROOT / "configs" / "warning_rules.yaml")
    assert config.version == "warning-rules-2026-07-18-imu-stationary-recalibration"
    assert len(config.sha256) == 64
    assert config.section("radar")["urgent_reference_s"] == 2.5
    assert config.section("radar")["configured_warning_range_m"] == 100.0
    assert config.section("vision")["path_policy"] == "center"
    assert config.section("gps")["max_factor"] == 1.25
    assert config.section("imu")["attention_error_deg"] == 10.0
    assert config.section("imu")["roll_offset_deg"] == -0.231277
    assert config.section("imu")["pitch_offset_deg"] == -0.518034
    assert config.section("imu")["critical_error_deg"] == 25.0
    assert config.section("imu")["prediction_horizon_s"] == 0.8
    assert config.section("freshness")["target_stale_ms"] == 500.0
    assert config.section("freshness")["imu_stale_ms"] == 100.0
    assert config.section("state")["score_variation"]["max_amplitude"] == 0.012
    assert config.metadata["calibration_status"] == "stationary_measured_pending_vehicle_validation"
    print("warning rule config: all tests passed")


if __name__ == "__main__":
    main()
