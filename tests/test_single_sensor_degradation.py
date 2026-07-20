from __future__ import annotations

from pathlib import Path

from src.fusion.data_types import GPSData, IMUData, RadarData, RadarTarget
from src.fusion.single_sensor_degradation import (
    SingleSensorDegradationController,
)


ROOT = Path(__file__).resolve().parents[1]


def _controller() -> SingleSensorDegradationController:
    return SingleSensorDegradationController(
        ROOT / "configs/warning_rules.yaml"
    )


def test_radar_rule_still_warns_when_vision_is_unavailable_and_gps_has_no_fix():
    controller = _controller()
    radar = RadarData(
        valid=True,
        targets=[
            RadarTarget(
                target_id=1,
                distance_m=2.0,
                relative_speed_mps=-2.0,
                angle_deg=0.0,
                confidence=1.0,
            )
        ],
    )
    decision = controller.evaluate(
        now_monotonic_ns=1_000_000_000,
        radar=radar,
        radar_usable=True,
        vision_result=None,
        vision_usable=False,
        imu=IMUData(valid=True),
        imu_usable=True,
        gps=GPSData(valid=False),
        gps_usable=False,
    )

    assert decision.missing_sensors == ("vision",)
    assert decision.level == 2
    assert decision.risk_score is not None
    assert decision.risk_score >= 0.70
    assert decision.modality_scores["vision"] is None
    assert "radar" in decision.evidence_sources


def test_unavailable_imu_is_not_reported_as_a_zero_risk_contribution():
    controller = _controller()
    decision = controller.evaluate(
        now_monotonic_ns=1_000_000_000,
        radar=RadarData(valid=True),
        radar_usable=True,
        vision_result=None,
        vision_usable=True,
        imu=IMUData(valid=False),
        imu_usable=False,
        gps=GPSData(valid=True, speed_kmh=12.0),
        gps_usable=True,
    )

    assert decision.missing_sensors == ("imu",)
    assert decision.modality_scores["imu"] is None
    assert decision.modality_levels["imu"] is None
    assert decision.level == 0
