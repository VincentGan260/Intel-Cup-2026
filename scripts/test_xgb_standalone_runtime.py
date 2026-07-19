#!/usr/bin/env python3
"""Unit checks for the isolated XGBoost runtime."""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_feature_contract_and_live_signals() -> None:
    from src.fusion.data_types import (
        GPSData,
        IMUData,
        RadarData,
        RadarTarget,
        VisionData,
        VisionObject,
    )
    from src.risk_ml.feature_window import FEATURE_NAMES, XGBoostFeatureWindow

    window = XGBoostFeatureWindow()
    gps = GPSData(timestamp=1.0, valid=True, speed_kmh=12.0)
    imu = IMUData(
        timestamp=1.0, valid=True, body_roll=2.0, body_pitch=-1.0,
        acc_x=0.0, acc_y=0.0, acc_z=9.8,
    )
    radar = RadarData(
        timestamp=1.0,
        valid=True,
        targets=[RadarTarget(
            target_id=1, distance_m=6.0, relative_speed_mps=-2.0,
            angle_deg=0.0, confidence=0.9,
        )],
    )
    vision_1 = VisionData(
        timestamp=1.0,
        valid=True,
        objects=[VisionObject(
            class_name="person", confidence=0.9,
            bbox=(270.0, 170.0, 370.0, 350.0), in_drivable_area=True,
        )],
    )
    first = window.update(
        now_monotonic=10.0, gps=gps, imu=imu, radar=radar, vision=vision_1
    )
    assert tuple(first.values) == FEATURE_NAMES
    assert first.values["radar_closing_speed_mps"] == 2.0
    assert first.values["radar_ttc_s"] == 3.0
    assert first.values["radar_person_matched"] == 1
    assert first.values["path_object_count"] == 1

    imu_2 = IMUData(
        timestamp=1.25, valid=True, body_roll=12.0, body_pitch=-1.0,
        acc_x=0.0, acc_y=0.0, acc_z=15.0, gyro_x=40.0,
    )
    vision_2 = VisionData(
        timestamp=1.25,
        valid=True,
        objects=[VisionObject(
            class_name="person", confidence=0.95,
            bbox=(250.0, 120.0, 390.0, 410.0), in_drivable_area=True,
        )],
    )
    second = window.update(
        now_monotonic=10.25, gps=gps, imu=imu_2, radar=radar, vision=vision_2
    )
    assert second.values["roll_error_deg"] == 12.0
    assert second.values["outward_rate_deg_s"] > 10.0
    assert second.values["acc_change_abs_mps2"] > 5.0
    assert second.values["jerk_abs_mps3"] > 10.0
    assert second.values["box_growth_rate_per_s"] > 0.0
    assert second.values["visual_tau_s"] is not None


def test_model_matches_held_out_rows() -> None:
    from src.risk_ml.predictor import XGBoostRiskPredictor

    predictor = XGBoostRiskPredictor(
        ROOT / "models/xgboost_risk/risk_classifier.json",
        ROOT / "models/xgboost_risk/metadata.json",
    )
    checked = 0
    correct = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    with (ROOT / "data/xgb/test.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["risk_label"])
            if class_counts[label] >= 30:
                continue
            features = {
                name: None if row[name] == "" else float(row[name])
                for name in predictor.feature_names
            }
            prediction = predictor.predict(features)
            correct += int(prediction.level == label)
            checked += 1
            class_counts[label] += 1
            if all(count >= 30 for count in class_counts.values()):
                break
    assert checked == 90
    assert class_counts == {0: 30, 1: 30, 2: 30}
    assert correct / checked >= 0.95


def test_runtime_has_no_rule_or_motor_imports() -> None:
    forbidden = {
        "src.fusion.physical_risk_rule",
        "src.fusion.imu_warning_rule",
        "src.fusion.vision_warning_rule",
        "src.fusion.warning_system",
        "src.actuator",
    }
    paths = [ROOT / "run_xgb_dashboard.py", *sorted((ROOT / "src/risk_ml").glob("*.py"))]
    imported = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    violations = sorted(
        name for name in imported
        if any(name == item or name.startswith(item + ".") for item in forbidden)
    )
    assert not violations, violations
    service = (ROOT / "deploy/edge/rider-xgb.service").read_text(encoding="utf-8")
    assert "--motor-mode" not in service
    assert "--enable-risk-rule" not in service


def main() -> None:
    test_feature_contract_and_live_signals()
    test_model_matches_held_out_rows()
    test_runtime_has_no_rule_or_motor_imports()
    print("standalone XGBoost runtime: all tests passed")


if __name__ == "__main__":
    main()
