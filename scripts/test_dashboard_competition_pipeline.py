"""Test that Dashboard real mode dispatches radar warning before vision."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.real_sensor_state import build_real_sensor_state
from src.fusion.data_types import GPSData, RadarData, RadarTarget, VisionData
from src.fusion.physical_risk_rule import PhysicalRiskRule


class FakeRadarReader:
    def __init__(self, data, age_ms=1.0):
        self.data = data
        self.age_ms = age_ms
        self.last_sample_monotonic_ns = 0
        self._serial = object()

    def read_once(self):
        self.last_sample_monotonic_ns = time.monotonic_ns() - int(self.age_ms * 1_000_000)
        return self.data


class FakeGPSReader:
    _serial = object()
    last_sample_monotonic_ns = 0

    def read_once(self):
        self.last_sample_monotonic_ns = time.monotonic_ns()
        return GPSData(valid=False)


class FakeMotor:
    def __init__(self):
        self.calls = []
        self.last_command_was_dispatched = False
        self.last_dispatch_monotonic_ns = 0

    def _call(self, level):
        self.calls.append(level)
        self.last_command_was_dispatched = True
        self.last_dispatch_monotonic_ns = time.monotonic_ns()

    def alert_low(self): self._call(0)
    def alert_medium(self): self._call(1)
    def alert_high(self): self._call(2)


class SlowVision:
    vision_start_ns = 0

    def process(self, frame):
        self.vision_start_ns = time.monotonic_ns()
        time.sleep(0.05)
        return VisionData(valid=True)

    def get_latest_vision_result(self): return None
    def get_runtime_info(self): return {"test": True}


class Frame:
    shape = (480, 640, 3)


def make_rule():
    return PhysicalRiskRule(
        body_width_m=0.66,
        point_gate_lateral_margin_m=0.025,
        mounting_offset_m=-0.055,
        mounting_uncertainty_m=0.005,
        configured_warning_range_m=10.0,
        radar_to_motor_p95_s=0.2,
    )


def main() -> None:
    target = RadarTarget(distance_m=5.0, relative_speed_mps=-1.0,
                         angle_deg=0.63, confidence=1.0)
    radar_data = RadarData(valid=True, targets=[target], nearest_distance_m=5.0, min_ttc=5.0)
    motor = FakeMotor()
    vision = SlowVision()
    state = build_real_sensor_state(
        camera_available=True,
        radar_reader=FakeRadarReader(radar_data),
        gps_reader=FakeGPSReader(),
        frame=Frame(),
        frame_capture_monotonic_ns=time.monotonic_ns(),
        vision_adapter=vision,
        risk_rule=make_rule(),
        motor=motor,
        target_stale_ms=500.0,
        radar_communication_watchdog_ms=2000.0,
    )
    assert state["risk_level"] == 1
    assert motor.calls == [1]
    assert motor.last_dispatch_monotonic_ns < vision.vision_start_ns
    assert state["risk_score_semantics"] == "ordinal_display_index_not_probability"

    stale_motor = FakeMotor()
    stale = build_real_sensor_state(
        camera_available=False,
        radar_reader=FakeRadarReader(radar_data, age_ms=500),
        gps_reader=FakeGPSReader(),
        risk_rule=make_rule(),
        motor=stale_motor,
        target_stale_ms=500.0,
        radar_communication_watchdog_ms=2000.0,
    )
    assert stale["risk_status"] == "unknown"
    assert stale_motor.calls == []
    print("dashboard competition pipeline: all tests passed")


if __name__ == "__main__":
    main()
