from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FeatureSchema:
    version: int
    window_size: int
    class_names: tuple[str, ...]
    modalities: tuple[str, ...]
    features: dict[str, tuple[str, ...]]
    score_weights: tuple[float, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            f"{modality}.{name}"
            for modality in self.modalities
            for name in (*self.features[modality], "valid")
        )

    @property
    def modality_dims(self) -> dict[str, int]:
        return {m: len(self.features[m]) + 1 for m in self.modalities}

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


def load_feature_schema(path: str | Path = "configs/gt_mrfn_features.yaml") -> FeatureSchema:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    modalities = tuple(raw["modalities"])
    return FeatureSchema(
        version=int(raw["schema_version"]),
        window_size=int(raw["window_size"]),
        class_names=tuple(raw["class_names"]),
        modalities=modalities,
        features={m: tuple(raw["modalities"][m]["features"]) for m in modalities},
        score_weights=tuple(float(x) for x in raw["score_weights"]),
    )


def _value(data: dict[str, Any], name: str) -> float:
    if name == "target_count":
        return float(data.get(name, len(data.get("targets", []))))
    if name == "object_count":
        return float(data.get(name, len(data.get("objects", []))))
    if name == "obstacle_count":
        return float(data.get(name, data.get("object_count", len(data.get("objects", [])))))
    if name in {"relative_speed_mps", "target_confidence"}:
        targets = data.get("targets", [])
        nearest = min(targets, key=lambda t: float(t.get("distance_m", float("inf"))), default={})
        source_name = "confidence" if name == "target_confidence" else name
        return float(data.get(name, nearest.get(source_name, 0.0)))
    return float(data.get(name, 0.0) or 0.0)


def vectorize_fusion_row(row: dict[str, Any], schema: FeatureSchema) -> np.ndarray:
    """Convert one fusion.jsonl row to the frozen feature order."""
    values: list[float] = []
    for modality in schema.modalities:
        block = row.get(modality) or {}
        data = block.get("data") if isinstance(block, dict) and "data" in block else block
        data = data if isinstance(data, dict) else {}
        valid = bool(block.get("valid", data.get("valid", False)))
        values.extend(_value(data, name) if valid else 0.0 for name in schema.features[modality])
        values.append(float(valid))
    return np.asarray(values, dtype=np.float32)


def modality_slices(schema: FeatureSchema) -> dict[str, slice]:
    result: dict[str, slice] = {}
    start = 0
    for name, dim in schema.modality_dims.items():
        result[name] = slice(start, start + dim)
        start += dim
    return result
