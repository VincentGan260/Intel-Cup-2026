#!/usr/bin/env python3
"""Regression test for the XGBoost-to-existing-Dashboard/cloud adapter."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.dashboard import server
    from src.dashboard.cloud_sync import build_ride_payload
    from src.dashboard.state_store import DashboardStateStore
    from src.risk_ml.dashboard_compat import build_dashboard_state

    contributions = {
        name: {
            "importance_pct": pct,
            "direction": "raises" if name in {"radar", "vision"} else "lowers",
        }
        for name, pct in {
            "radar": 40.0, "vision": 30.0, "imu": 20.0, "gps": 10.0,
        }.items()
    }
    prediction = {
        "level": 2,
        "label": "高风险",
        "risk_score": 0.91,
        "confidence": 0.97,
        "module_contributions": contributions,
    }
    sensors = {
        name: {"status": "active", "valid": True, "age_ms": 10.0, "error": ""}
        for name in ("radar", "gps", "imu", "vision")
    }
    radar = SimpleNamespace(
        valid=True,
        targets=[SimpleNamespace()],
        nearest_distance_m=2.5,
        min_ttc=1.2,
    )
    gps = SimpleNamespace(
        valid=True, latitude=39.9, longitude=116.4, speed_kmh=18.0,
        fix_quality=1, satellites=8,
    )
    imu = SimpleNamespace(
        valid=True, body_roll=2.0, body_pitch=-1.0, roll=2.0, pitch=-1.0,
        yaw=0.0, acc_x=0.0, acc_y=0.0, acc_z=9.8,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        brake_score=0.0, bump_score=0.0, tilt_score=0.0,
    )
    vision = SimpleNamespace(
        valid=True, objects=[], person_count=0, vehicle_count=0,
        drivable_area_ratio=0.62, pipeline_inference_ms=12.3,
    )
    state = build_dashboard_state(
        timestamp=1_700_000_000.0,
        status="active",
        prediction=prediction,
        prediction_error="",
        features={f"feature_{index}": float(index) for index in range(31)},
        modules={name: {"contribution": value}
                 for name, value in contributions.items()},
        feature_window={"warm": True},
        sensor_states=sensors,
        runtime={"state_hz": 10.0, "feature_count": 31},
        motor_control=True,
        motor={
            "mode": "real", "connected": True, "faulted": False,
            "gate_open": True, "gate_reason": "xgboost_prediction_accepted",
        },
        radar=radar,
        gps=gps,
        imu=imu,
        vision=vision,
        camera_available=True,
        frame_width=640,
        frame_height=480,
        started_at=1_699_999_900.0,
    )
    assert state["decision_engine"] == "xgboost-only"
    assert state["old_rules_loaded"] is False
    assert state["radar_level"] == 2
    assert state["vision_level"] == 2
    assert state["radar_score"] == 0.4
    assert state["gps_score"] == 0.1
    assert len(state["features"]) == 31

    payload = build_ride_payload(state, "bike-001")
    assert tuple(payload) == (
        "device_id", "collected_at", "gps_valid", "latitude", "longitude",
        "speed_kmh", "radar_valid", "target_count", "nearest_distance_m",
        "min_ttc_s", "vision_valid", "obstacle_count", "drivable_area_ratio",
        "imu_posture", "risk_score", "risk_level", "system_status",
        "warning_reason", "radar_level", "vision_level",
    )

    store = DashboardStateStore()
    store.set_state(state)
    server.inject_state_store(store)
    server.inject_camera(SimpleNamespace(is_available=True))
    api_state = json.loads(server.api_state().body)
    health = json.loads(asyncio.run(server.api_health()).body)
    html = asyncio.run(server.index()).body.decode()
    assert api_state["decision_engine"] == "xgboost-only"
    assert health["decision_engine"] == "xgboost-only"
    assert health["motor_control"] is True
    assert all(
        text in html
        for text in ("XGBoost 模态贡献", "GPS 贡献", "/video_annotated_feed")
    )
    print("xgboost dashboard/cloud compatibility test passed")


if __name__ == "__main__":
    main()
