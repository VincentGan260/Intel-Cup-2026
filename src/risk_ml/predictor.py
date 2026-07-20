"""Small, dependency-light XGBoost inference adapter."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


MODULE_FEATURES = {
    "gps": (
        "gps_valid",
        "gps_speed_kmh",
    ),
    "imu": (
        "imu_valid",
        "pitch_abs_deg",
        "roll_abs_deg",
        "roll_error_deg",
        "outward_rate_deg_s",
        "imu_attention_duration_ms",
        "imu_urgent_consistent_samples",
        "acc_norm_mean_mps2",
        "acc_delta_signed_mps2",
        "acc_change_abs_mps2",
        "jerk_abs_mps3",
    ),
    "radar": (
        "radar_valid",
        "radar_target_count",
        "radar_path_target_count",
        "radar_min_distance_m",
        "radar_relative_speed_mps",
        "radar_closing_speed_mps",
        "radar_ttc_s",
    ),
    "vision": (
        "vision_valid",
        "object_count",
        "path_object_count",
        "max_path_bottom_ratio",
        "box_growth_rate_per_s",
        "growth_duration_s",
        "visual_tau_s",
        "vision_confidence",
    ),
}


@dataclass(frozen=True)
class RiskPrediction:
    """One three-class prediction produced by the standalone model."""

    level: int
    label: str
    confidence: float
    risk_score: float
    probabilities: dict[str, float]
    module_contributions: dict[str, dict]
    explanation: dict
    inference_ms: float

    def as_dict(self) -> dict:
        result = asdict(self)
        result["risk_score_100"] = round(self.risk_score * 100.0, 2)
        return result


class XGBoostRiskPredictor:
    """Load the portable JSON model and enforce its feature contract."""

    SCORE_WEIGHTS = (0.15, 0.55, 0.90)

    def __init__(self, model_path: str | Path, metadata_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"XGBoost model not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"XGBoost metadata not found: {self.metadata_path}")

        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.feature_names = tuple(self.metadata["feature_columns"])
        expected_count = int(self.metadata.get("feature_count", len(self.feature_names)))
        if expected_count != len(self.feature_names):
            raise ValueError(
                f"metadata feature_count={expected_count}, names={len(self.feature_names)}"
            )

        import xgboost as xgb

        self._xgb = xgb
        self._booster = xgb.Booster()
        self._booster.load_model(str(self.model_path))
        if self._booster.num_features() != expected_count:
            raise ValueError(
                f"model expects {self._booster.num_features()} features, "
                f"metadata declares {expected_count}"
            )
        model_names = tuple(self._booster.feature_names or ())
        if model_names and model_names != self.feature_names:
            raise ValueError("model feature order does not match metadata feature_columns")

        self.best_iteration = int(self.metadata.get("best_iteration", -1))
        raw_labels = self.metadata.get("labels", {"0": "低风险", "1": "中风险", "2": "高风险"})
        self.labels = {int(key): str(value) for key, value in raw_labels.items()}
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        assigned_features = {
            feature
            for module_features in MODULE_FEATURES.values()
            for feature in module_features
        }
        if assigned_features != set(self.feature_names):
            raise ValueError(
                "module feature groups do not match the model feature contract"
            )

    def _module_explanation(self, matrix, kwargs: dict) -> tuple[dict, dict]:
        import numpy as np

        contributions = self._booster.predict(
            matrix,
            pred_contribs=True,
            validate_features=True,
            **kwargs,
        )[0]
        expected_shape = (3, len(self.feature_names) + 1)
        if contributions.shape != expected_shape:
            raise RuntimeError(
                f"expected SHAP contributions {expected_shape}, "
                f"got {contributions.shape}"
            )

        feature_values = contributions[:, :-1]
        total_absolute = float(np.abs(feature_values).sum())
        feature_indexes = {
            name: index for index, name in enumerate(self.feature_names)
        }
        modules = {}
        for module, names in MODULE_FEATURES.items():
            indexes = [feature_indexes[name] for name in names]
            grouped = feature_values[:, indexes]
            class_contributions = grouped.sum(axis=1)
            module_absolute = float(np.abs(grouped).sum())
            high_risk_contribution = float(class_contributions[2])
            if high_risk_contribution > 1e-6:
                direction = "raises"
            elif high_risk_contribution < -1e-6:
                direction = "lowers"
            else:
                direction = "neutral"

            ranked = sorted(
                indexes,
                key=lambda index: float(
                    np.max(np.abs(feature_values[:, index]))
                ),
                reverse=True,
            )
            modules[module] = {
                "importance_pct": round(
                    100.0 * module_absolute / total_absolute, 3
                ) if total_absolute > 0.0 else 0.0,
                "high_risk_margin": round(high_risk_contribution, 6),
                "direction": direction,
                "class_margin_contributions": {
                    self.labels.get(class_index, str(class_index)): round(
                        float(class_contributions[class_index]), 6
                    )
                    for class_index in range(3)
                },
                "top_features": [
                    {
                        "name": self.feature_names[index],
                        "importance": round(
                            float(np.max(np.abs(feature_values[:, index]))), 6
                        ),
                        "high_risk_margin": round(
                            float(feature_values[2, index]), 6
                        ),
                    }
                    for index in ranked[:3]
                ],
            }

        explanation = {
            "method": "tree_shap_margin",
            "importance_semantics": (
                "share of absolute TreeSHAP margin contributions"
            ),
            "direction_semantics": (
                "signed contribution to the high-risk class margin"
            ),
            "class_bias": {
                self.labels.get(class_index, str(class_index)): round(
                    float(contributions[class_index, -1]), 6
                )
                for class_index in range(3)
            },
        }
        return modules, explanation

    def predict(self, features: Mapping[str, float | int | None]) -> RiskPrediction:
        """Predict one window after validating all names and their exact order."""

        missing = [name for name in self.feature_names if name not in features]
        extra = sorted(set(features) - set(self.feature_names))
        if missing or extra:
            raise ValueError(f"feature contract mismatch; missing={missing}, extra={extra}")

        import numpy as np

        values = []
        for name in self.feature_names:
            raw = features[name]
            value = float("nan") if raw is None else float(raw)
            if not math.isfinite(value):
                value = float("nan")
            values.append(value)
        matrix = self._xgb.DMatrix(
            np.asarray([values], dtype=np.float32),
            feature_names=list(self.feature_names),
            missing=np.nan,
        )

        started = time.perf_counter()
        kwargs = {}
        if self.best_iteration >= 0:
            kwargs["iteration_range"] = (0, self.best_iteration + 1)
        raw_probabilities = self._booster.predict(
            matrix, validate_features=True, **kwargs
        )
        module_contributions, explanation = self._module_explanation(
            matrix, kwargs
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        probabilities = [float(value) for value in raw_probabilities[0]]
        if len(probabilities) != 3:
            raise RuntimeError(f"expected 3 probabilities, got {len(probabilities)}")

        level = max(range(3), key=probabilities.__getitem__)
        score = sum(weight * probability for weight, probability in zip(
            self.SCORE_WEIGHTS, probabilities
        ))
        named = {
            self.labels.get(index, str(index)): round(probability, 8)
            for index, probability in enumerate(probabilities)
        }
        return RiskPrediction(
            level=level,
            label=self.labels.get(level, str(level)),
            confidence=max(probabilities),
            risk_score=max(0.0, min(1.0, score)),
            probabilities=named,
            module_contributions=module_contributions,
            explanation=explanation,
            inference_ms=inference_ms,
        )

    def runtime_info(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "model_sha256": self.model_sha256,
            "feature_count": len(self.feature_names),
            "best_iteration": self.best_iteration,
            "model_type": self.metadata.get("model_type", "XGBoost"),
            "synthetic_data_warning": self.metadata.get("synthetic_data_warning", ""),
        }
