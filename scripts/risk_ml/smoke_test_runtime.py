#!/usr/bin/env python3
"""Dependency and model-contract smoke test for DK-2500 deployment."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_rows(path: Path, feature_names: tuple[str, ...], per_class: int = 50):
    rows = []
    counts = {0: 0, 1: 0, 2: 0}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["risk_label"])
            if counts[label] >= per_class:
                continue
            features = {
                name: None if row[name] == "" else float(row[name])
                for name in feature_names
            }
            rows.append((features, label))
            counts[label] += 1
            if all(count >= per_class for count in counts.values()):
                break
    if any(count < per_class for count in counts.values()):
        raise RuntimeError(f"insufficient class coverage: {counts}")
    return rows


def main() -> None:
    from src.risk_ml.feature_window import FEATURE_NAMES
    from src.risk_ml.predictor import XGBoostRiskPredictor

    predictor = XGBoostRiskPredictor(
        ROOT / "models/xgboost_risk/risk_classifier.json",
        ROOT / "models/xgboost_risk/metadata.json",
    )
    if predictor.feature_names != FEATURE_NAMES:
        raise SystemExit("feature order mismatch")

    rows = load_rows(ROOT / "data/xgb/test.csv", FEATURE_NAMES)
    correct = 0
    latency = []
    for features, expected in rows:
        prediction = predictor.predict(features)
        correct += int(prediction.level == expected)
        latency.append(prediction.inference_ms)
    accuracy = correct / max(1, len(rows))
    if accuracy < 0.95:
        raise SystemExit(f"smoke accuracy too low: {accuracy:.4f}")

    result = {
        "status": "ok",
        "samples": len(rows),
        "accuracy": round(accuracy, 6),
        "feature_count": len(FEATURE_NAMES),
        "model_sha256": predictor.model_sha256,
        "max_inference_ms": round(max(latency), 4),
        "mean_inference_ms": round(sum(latency) / len(latency), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
