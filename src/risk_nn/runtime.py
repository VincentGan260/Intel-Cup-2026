from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .features import FeatureSchema, load_feature_schema, modality_slices, vectorize_fusion_row
from .model import GTMRFN, RiskPrediction

def save_model_package(path: str | Path, model: GTMRFN, schema: FeatureSchema, mu: np.ndarray, sigma: np.ndarray, **metadata: Any) -> None:
    mu, sigma = np.asarray(mu, dtype=np.float32), np.asarray(sigma, dtype=np.float32)
    if mu.shape != (schema.feature_dim,) or sigma.shape != (schema.feature_dim,): raise ValueError("mu/sigma shape must match feature schema")
    torch.save({"format_version": 1, "model_state": model.state_dict(), "modality_dims": schema.modality_dims, "hidden_dim": model.head[0].in_features, "feature_names": schema.feature_names, "window_size": schema.window_size, "class_names": schema.class_names, "score_weights": schema.score_weights, "mu": mu, "sigma": np.maximum(sigma, 1e-6), "metadata": metadata}, Path(path))

class GTMRFNRuntime:
    def __init__(self, package_path: str | Path, schema_path: str | Path = "configs/gt_mrfn_features.yaml") -> None:
        self.schema = load_feature_schema(schema_path); package = torch.load(Path(package_path), map_location="cpu", weights_only=False)
        if tuple(package["feature_names"]) != self.schema.feature_names or package["window_size"] != self.schema.window_size: raise ValueError("model package and feature schema are incompatible")
        self.model = GTMRFN(package["modality_dims"], package["hidden_dim"]); self.model.load_state_dict(package["model_state"]); self.model.eval()
        self.mu, self.sigma = np.asarray(package["mu"]), np.asarray(package["sigma"]); self.buffer: deque[np.ndarray] = deque(maxlen=self.schema.window_size); self.slices = modality_slices(self.schema)
    def reset(self) -> None: self.buffer.clear()
    def predict_row(self, row: dict[str, Any]) -> RiskPrediction:
        vector = vectorize_fusion_row(row, self.schema); self.buffer.append(vector); frames = list(self.buffer); frames = [frames[0]] * (self.schema.window_size - len(frames)) + frames
        raw = np.stack(frames); x = (raw - self.mu) / self.sigma
        for sl in self.slices.values():
            valid = raw[:, sl.stop - 1:sl.stop]
            x[:, sl.start:sl.stop - 1] *= valid
            x[:, sl.stop - 1] = raw[:, sl.stop - 1]
        tensors = {m: torch.from_numpy(x[:, sl].astype(np.float32)).unsqueeze(0) for m, sl in self.slices.items()}
        with torch.no_grad(): probabilities = torch.softmax(self.model(tensors), 1)[0].numpy()
        return RiskPrediction(tuple(float(v) for v in probabilities), float(probabilities @ np.asarray(self.schema.score_weights)), int(probabilities.argmax()))
