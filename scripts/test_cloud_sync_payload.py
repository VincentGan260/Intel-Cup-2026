import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.cloud_sync import build_ride_payload


def main() -> int:
    state = {
        "timestamp": 1_700_000_000.0,
        "risk_score": 0.68,
        "risk_level": 1,
        "gps_data": {"valid": True, "latitude": 39.9, "longitude": 116.4,
                     "speed_kmh": 18.6},
        "radar_data": {"valid": True, "target_count": 2,
                       "nearest_distance_m": 4.25, "min_ttc_s": 3.1},
        "vision_details": {"valid": True, "object_count": 3,
                           "drivable_area_ratio": 0.63},
        "imu_data": {"valid": True, "roll": -8.0, "pitch": 6.0},
    }
    payload = build_ride_payload(state, "bike-001")
    assert payload["device_id"] == "bike-001"
    assert payload["target_count"] == 2
    assert payload["obstacle_count"] == 3
    assert payload["imu_posture"] == "向左倾斜并向前倾斜"
    assert payload["drivable_area_ratio"] == 0.63
    assert payload["risk_level"] == 1
    invalid = build_ride_payload({"timestamp": 1_700_000_000.0}, "bike-001")
    assert invalid["latitude"] is None
    assert invalid["radar_valid"] is True
    assert invalid["target_count"] == 0
    assert invalid["nearest_distance_m"] == 0.0
    assert invalid["min_ttc_s"] == 0.0
    assert invalid["radar_level"] == 0
    assert invalid["drivable_area_ratio"] is None
    assert invalid["risk_score"] is None
    assert invalid["risk_level"] is None
    invalid_numbers = build_ride_payload({
        "timestamp": 1_700_000_000.0,
        "radar_data": {"valid": True, "nearest_distance_m": math.inf,
                       "min_ttc_s": math.nan},
    }, "bike-001")
    assert invalid_numbers["nearest_distance_m"] == 0.0
    assert invalid_numbers["min_ttc_s"] == 0.0
    no_target = build_ride_payload({
        "timestamp": 1_700_000_000.0,
        "radar_data": {"valid": True, "target_count": 0,
                       "nearest_distance_m": -1.0, "min_ttc_s": -1.0},
    }, "bike-001")
    assert no_target["radar_valid"] is True
    assert no_target["nearest_distance_m"] == 0.0
    assert no_target["min_ttc_s"] == 0.0
    print("cloud sync payload test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
