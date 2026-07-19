"""Small, dependency-light XGBoost inference adapter."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RiskPrediction:
    """One three-class prediction produced by the standalone model."""

    level: int
    label: str
    confidence: float
    risk_score: float
    probabilities: dict[str, float]
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
