"""Dependency-light boundary checks for the provisional IMU warning rule."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.data_types import IMUData
from src.fusion.imu_warning_rule import ImuWarningRule, ImuWarningRuleConfig
from src.fusion.warning_arbiter import arbitrate_warning_events
from src.fusion.warning_events import ModalityEvent


OFFSET = -5.296


def sample(*, body_roll: float, gyro_x: float, gyro_z: float = 0.0) -> IMUData:
    return IMUData(
        valid=True, roll=OFFSET + body_roll,
        gyro_x=gyro_x, gyro_z=gyro_z,
    )


def evaluate(rule: ImuWarningRule, index: int, *, body_roll: float,
             gyro_x: float, gyro_z: float = 0.0,
             gps_speed_kmh: float = 18.0, gps_usable: bool = True,
             step_ms: int = 50):
    capture_ns = 1_000_000_000 + index * step_ms * 1_000_000
    return rule.evaluate_event(
        sample(body_roll=body_roll, gyro_x=gyro_x, gyro_z=gyro_z),
        capture_monotonic_ns=capture_ns,
        completed_monotonic_ns=capture_ns + 1_000_000,
        sequence=index + 1,
        gps_speed_kmh=gps_speed_kmh,
        gps_usable=gps_usable,
    )


def base_event(source: str, level: int, score: float, now_ns: int) -> ModalityEvent:
    return ModalityEvent(
        source=source, source_id="1", sequence=1,
        capture_monotonic_ns=now_ns, completed_monotonic_ns=now_ns,
        usable=True, level=level, reason=f"{source}_test", risk_score=score,
    )


def main() -> None:
    config = ImuWarningRuleConfig()

    stationary = ImuWarningRule(config)
    event = evaluate(stationary, 0, body_roll=18.0, gyro_x=0.0)
    assert event.level == 0
    assert event.risk_score is not None and event.risk_score < 0.35

    # Reader-calibrated body angles take precedence and must not have the
    # installation offset subtracted a second time.
    calibrated = ImuWarningRule(config)
    calibrated_sample = sample(body_roll=0.0, gyro_x=0.0)
    calibrated_sample.body_roll = 0.0
    calibrated_event = calibrated.evaluate_event(
        calibrated_sample,
        capture_monotonic_ns=1_000_000_000,
        completed_monotonic_ns=1_001_000_000,
        sequence=1,
    )
    assert calibrated_event.details["body_roll_deg"] == 0.0

    corner = ImuWarningRule(config)
    speed_mps = 5.0
    yaw_rate_deg_s = 20.0
    equilibrium_deg = config.turn_sign * math.degrees(math.atan(
        speed_mps * math.radians(yaw_rate_deg_s) / config.gravity_mps2))
    event = evaluate(
        corner, 0, body_roll=equilibrium_deg,
        gyro_x=0.0, gyro_z=yaw_rate_deg_s)
    assert event.level == 0
    assert abs(event.details["roll_error_deg"]) < 1e-9
    assert event.details["turn_compensation_valid"] is True

    attention = ImuWarningRule(config)
    events = [evaluate(attention, index, body_roll=10.0, gyro_x=5.0)
              for index in range(4)]
    assert events[2].level == 0
    assert events[3].level == 1
    assert 0.35 <= events[3].risk_score < 0.70

    urgent = ImuWarningRule(config)
    events = [evaluate(urgent, index, body_roll=8.0, gyro_x=25.0)
              for index in range(3)]
    assert [item.level for item in events] == [0, 0, 2]
    assert events[-1].risk_score is not None and events[-1].risk_score >= 0.70
    assert events[-1].details["time_to_critical_s"] <= 0.8

    no_gps = ImuWarningRule(config)
    events = [evaluate(
        no_gps, index, body_roll=12.0, gyro_x=25.0,
        gps_usable=False) for index in range(4)]
    assert events[-1].level == 1
    assert max(item.level for item in events) < 2
    assert events[-1].details["high_risk_cap"] == 1

    now_ns = events[-1].completed_monotonic_ns
    fused = arbitrate_warning_events(
        base_event("radar", 0, 0.10, now_ns),
        base_event("vision", 0, 0.20, now_ns),
        None,
        base_event("imu", 2, 0.85, now_ns),
        now_ns=now_ns, imu_enabled=True,
    )
    assert fused.final_level == 2
    assert fused.risk_score == 0.85
    assert fused.warning_reason == "imu_test"
    assert fused.evidence_sources == ("imu",)

    degraded = arbitrate_warning_events(
        base_event("radar", 1, 0.50, now_ns),
        base_event("vision", 0, 0.10, now_ns),
        now_ns=now_ns, imu_enabled=True,
    )
    assert degraded.final_level == 1
    assert degraded.system_status == "degraded"
    assert degraded.imu_status == "unavailable"
    print("IMU warning rule: all tests passed")


if __name__ == "__main__":
    main()
