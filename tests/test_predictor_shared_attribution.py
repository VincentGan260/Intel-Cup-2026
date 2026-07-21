from __future__ import annotations

import numpy as np
import pytest

from src.risk_ml.predictor import (
    GPS_IMU_SHARED_FEATURES,
    MODULE_FEATURES,
    XGBoostRiskPredictor,
)


class _FakeBooster:
    def __init__(self, contributions: np.ndarray) -> None:
        self._contributions = contributions

    def predict(self, *args, **kwargs) -> np.ndarray:
        return self._contributions


def _predictor_with_contributions() -> XGBoostRiskPredictor:
    predictor = XGBoostRiskPredictor.__new__(XGBoostRiskPredictor)
    predictor.feature_names = tuple(
        feature
        for module_features in MODULE_FEATURES.values()
        for feature in module_features
    )
    predictor.labels = {0: "low", 1: "medium", 2: "high"}

    values = np.zeros((1, 3, len(predictor.feature_names) + 1), dtype=np.float32)
    values[0, :, 0] = 1.0
    values[0, :, 1] = 1.0
    values[0, :, 2] = 4.0
    for name in GPS_IMU_SHARED_FEATURES:
        values[0, :, predictor.feature_names.index(name)] = (1.0, 2.0, 3.0)
    predictor._booster = _FakeBooster(values)
    return predictor


def test_gps_imu_shared_contributions_are_split_without_double_counting() -> None:
    predictor = _predictor_with_contributions()

    modules, explanation = predictor._module_explanation(
        object(), {}, gps_usable=True
    )

    assert modules["gps"]["importance_pct"] == pytest.approx(31.429, abs=0.001)
    assert modules["imu"]["importance_pct"] == pytest.approx(68.571, abs=0.001)
    assert sum(module["importance_pct"] for module in modules.values()) == pytest.approx(100.0)
    assert modules["gps"]["class_margin_contributions"] == {
        "low": 3.2,
        "medium": 4.4,
        "high": 5.6,
    }
    assert modules["imu"]["class_margin_contributions"] == {
        "low": 6.8,
        "medium": 9.6,
        "high": 12.4,
    }
    assert explanation["gps_imu_shared_attribution"]["active"] is True


def test_gps_imu_shared_contributions_stay_with_imu_when_gps_is_invalid() -> None:
    predictor = _predictor_with_contributions()

    modules, explanation = predictor._module_explanation(
        object(), {}, gps_usable=False
    )

    assert modules["gps"]["importance_pct"] == pytest.approx(14.286, abs=0.001)
    assert modules["imu"]["importance_pct"] == pytest.approx(85.714, abs=0.001)
    assert explanation["gps_imu_shared_attribution"]["active"] is False
