from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .features import FeatureSchema, modality_slices

def split_groups(groups: np.ndarray, y: np.ndarray, seed: int = 42) -> dict[str, list[str]]:
    unique = np.unique(groups.astype(str))
    if len(unique) < 3:
        raise ValueError("至少需要3个独立group/session才能进行7:2:1无泄漏划分")
    for cls in np.unique(y):
        carrying = np.unique(groups[y == cls])
        if len(carrying) < 3:
            raise ValueError(f"类别{int(cls)}至少要分布在3个独立group中，当前只有{len(carrying)}个")
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(y))
    n_test = max(n_classes, round(len(unique) * .1)); n_val = max(n_classes, round(len(unique) * .2))
    if n_test + n_val >= len(unique):
        raise ValueError("独立group数量不足以让train/val/test都覆盖全部类别")
    # Small datasets often contain one label-dominant session. Try many deterministic
    # group permutations and retain one that covers every class in every split.
    for _ in range(10000):
        order = unique[rng.permutation(len(unique))]
        result = {"train": order[n_test+n_val:].tolist(), "val": order[n_test:n_test+n_val].tolist(), "test": order[:n_test].tolist()}
        if all(set(np.unique(y[np.isin(groups, ids)])) == set(np.unique(y)) for ids in result.values()):
            return result
    raise ValueError("无法得到每个集合都覆盖全部类别的group划分；请增加独立采集片段")

def fit_normalizer(x: np.ndarray, schema: FeatureSchema) -> tuple[np.ndarray, np.ndarray]:
    mu, sigma = np.zeros(schema.feature_dim, np.float32), np.ones(schema.feature_dim, np.float32)
    for sl in modality_slices(schema).values():
        valid = x[:, :, sl.stop - 1] > .5
        for index in range(sl.start, sl.stop - 1):
            values = x[:, :, index][valid]
            if values.size:
                mu[index] = values.mean(); sigma[index] = max(float(values.std()), 1e-6)
    return mu, sigma

def normalize(x: np.ndarray, schema: FeatureSchema, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    out = ((x - mu) / sigma).astype(np.float32)
    for sl in modality_slices(schema).values():
        valid = x[:, :, sl.stop - 1:sl.stop]
        out[:, :, sl.start:sl.stop - 1] *= valid
        out[:, :, sl.stop - 1] = x[:, :, sl.stop - 1]
    return out

class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, schema: FeatureSchema, modality_dropout: float = 0.0) -> None:
        self.x, self.y, self.schema, self.dropout = x, y, schema, modality_dropout
        self.slices = modality_slices(schema)
    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, index: int):
        x = self.x[index].copy()
        if self.dropout > 0:
            dropped = np.random.random(len(self.slices)) < self.dropout
            if dropped.all(): dropped[np.random.randint(len(dropped))] = False
            for drop, sl in zip(dropped, self.slices.values()):
                if drop: x[:, sl] = 0.0
        return {m: torch.from_numpy(x[:, sl]) for m, sl in self.slices.items()}, torch.tensor(self.y[index], dtype=torch.long)

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]) -> dict:
    n = len(class_names); cm = np.zeros((n, n), dtype=int)
    for truth, pred in zip(y_true, y_pred): cm[int(truth), int(pred)] += 1
    per_class = {}
    for i, name in enumerate(class_names):
        tp, fp, fn = cm[i, i], cm[:, i].sum()-cm[i, i], cm[i, :].sum()-cm[i, i]
        precision = tp / max(1, tp+fp); recall = tp / max(1, tp+fn)
        per_class[name] = {"precision": precision, "recall": recall, "f1": 2*precision*recall/max(1e-12, precision+recall), "support": int(cm[i].sum())}
    return {"accuracy": float(np.trace(cm)/max(1, cm.sum())), "confusion_matrix": cm.tolist(), "per_class": per_class}
