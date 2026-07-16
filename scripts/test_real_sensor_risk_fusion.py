"""Regression test: real Dashboard mode must publish adaptive fused risk."""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.real_sensor_state import build_real_sensor_state
from src.fusion.data_types import GPSData, RadarData, VisionData
from src.fusion.risk_level import RiskLevelClassifier
from src.fusion.risk_model import RiskModel


class Reader:
    def __init__(self, value):
        self.value = value
        self.last_sample_monotonic_ns = 0
        self._serial = object()

    def read_once(self):
        return self.value


class Vision:
    def process(self, _frame):
        return VisionData(
            timestamp=time.time(), valid=True, max_visual_risk=0.8,
            drivable_area_ratio=0.6,
        )

    def get_latest_vision_result(self):
        return None

    def get_runtime_info(self):
        return {"test": True}


class Frame:
    shape = (480, 640, 3)


def main() -> int:
    ts = time.time()
    state = build_real_sensor_state(
        camera_available=True,
        radar_reader=Reader(RadarData(timestamp=ts, valid=False)),
        gps_reader=Reader(GPSData(timestamp=ts, valid=False)),
        frame=Frame(),
        vision_adapter=Vision(),
        risk_model=RiskModel(),
        classifier=RiskLevelClassifier(),
    )
    assert state["risk_score_semantics"] == "adaptive_weighted_fusion"
    assert state["risk_items"]["obs"] == 0.8
    assert state["vision_details"]["max_visual_risk"] == 0.8
    assert state["risk_score"] > 0.0
    assert state["risk_items"]["pose"] == 0.0
    assert state["radar_data"]["valid"] is False
    assert state["gps_data"]["valid"] is False
    print("real sensor adaptive risk fusion test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
