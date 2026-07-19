#!/usr/bin/env python3
"""Train and evaluate the rule-supervised XGBoost risk classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = ROOT / ".codex_tmp" / "risk_ml_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


DEFAULT_DATA_DIR = ROOT / "data" / "xgb"
DEFAULT_MODEL_DIR = ROOT / "models" / "xgboost_risk"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "xgboost_risk_training_20260720"
LABEL_NAMES = ["低风险", "中风险", "高风险"]
PLOT_LABEL_NAMES = ["Low", "Medium", "High"]
SEED = 20260719


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(data_dir: Path) -> tuple[
    dict[str, pd.DataFrame], list[str], str, dict[str, Any]
]:
    config = json.loads((data_dir / "feature_config.json").read_text(
        encoding="utf-8"
    ))
    feature_columns = list(config["feature_columns"])
    target = str(config["target"])
    if target in feature_columns:
        raise ValueError("target column appears in the feature whitelist")
    forbidden_fragments = ("rule_score", "risk_label", "trigger_reason",
                           "hard_rule", "scenario", "sample_id")
    leaked = [
        column for column in feature_columns
        if any(fragment in column for fragment in forbidden_fragments)
    ]
    if leaked:
        raise ValueError(f"label leakage fields in feature whitelist: {leaked}")

    frames = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing = [column for column in feature_columns + [target]
                   if column not in frame.columns]
        if missing:
            raise ValueError(f"{split} is missing columns: {missing}")
        if not set(frame[target].unique()).issubset({0, 1, 2}):
            raise ValueError(f"{split} contains unexpected labels")
        frames[split] = frame

    sample_sets = {
        split: set(frame["sample_id"].astype(str))
        for split, frame in frames.items()
    }
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        overlap = sample_sets[left] & sample_sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} sample ID overlap: {len(overlap)}")
    return frames, feature_columns, target, config


def metrics_for(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    high_total = int((y_true == 2).sum())
    high_missed = int(((y_true == 2) & (y_pred < 2)).sum())
    high_to_low = int(((y_true == 2) & (y_pred == 0)).sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "high_risk_recall": float(recall_score(
            y_true, y_pred, labels=[2], average="macro", zero_division=0
        )),
        "high_risk_missed_count": high_missed,
        "high_risk_to_low_count": high_to_low,
        "high_risk_miss_rate": (
            float(high_missed / high_total) if high_total else 0.0
        ),
        "multiclass_log_loss": float(log_loss(
            y_true, probabilities, labels=[0, 1, 2]
        )),
        "confusion_matrix": matrix.tolist(),
        "support": {
            str(label): int((y_true == label).sum()) for label in (0, 1, 2)
        },
    }


def save_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    result = frame[[
        "sample_id", "scenario_type", "boundary_case",
        "risk_label", "risk_label_name", "rule_score",
    ]].copy()
    result["predicted_label"] = predicted
    result["predicted_label_name"] = [LABEL_NAMES[int(value)]
                                      for value in predicted]
    result["prob_low"] = probabilities[:, 0]
    result["prob_medium"] = probabilities[:, 1]
    result["prob_high"] = probabilities[:, 2]
    result["xgb_risk_score"] = (
        0.15 * probabilities[:, 0]
        + 0.55 * probabilities[:, 1]
        + 0.90 * probabilities[:, 2]
    )
    result["xgb_risk_score_100"] = 100.0 * result["xgb_risk_score"]
    result["correct"] = (result["risk_label"] == result["predicted_label"]).astype(int)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_confusion(matrix: list[list[int]], title: str, output_path: Path) -> None:
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(6.2, 5.2), dpi=160)
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=PLOT_LABEL_NAMES,
        yticklabels=PLOT_LABEL_NAMES,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    threshold = values.max() / 2.0 if values.size else 0.0
    for row in range(3):
        for column in range(3):
            axis.text(
                column, row, f"{values[row, column]:,}",
                ha="center", va="center",
                color="white" if values[row, column] > threshold else "#243746",
                fontsize=11, fontweight="bold",
            )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def feature_importance(
    model: XGBClassifier, feature_columns: list[str]
) -> pd.DataFrame:
    gain_by_name = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame({
        "feature": feature_columns,
        "gain": [float(gain_by_name.get(name, 0.0))
                 for name in feature_columns],
    })
    total = float(importance["gain"].sum())
    importance["gain_normalized"] = (
        importance["gain"] / total if total > 0 else 0.0
    )
    return importance.sort_values(
        ["gain_normalized", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def plot_importance(importance: pd.DataFrame, output_path: Path) -> None:
    top = importance.head(15).sort_values("gain_normalized")
    figure, axis = plt.subplots(figsize=(8.2, 6.2), dpi=160)
    axis.barh(top["feature"], top["gain_normalized"], color="#147D82")
    axis.set_xlabel("Normalized gain")
    axis.set_title("XGBoost feature importance (top 15)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames, feature_columns, target, config = load_data(args.data_dir)
    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]
    x_train = train[feature_columns]
    y_train = train[target].astype(int).to_numpy()
    x_validation = validation[feature_columns]
    y_validation = validation[target].astype(int).to_numpy()
    x_test = test[feature_columns]
    y_test = test[target].astype(int).to_numpy()

    parameters = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "objective": "multi:softprob",
        "num_class": 3,
        # XGBoost early stopping watches the last metric. Keep probability
        # quality (mlogloss) last because the risk score uses predict_proba.
        "eval_metric": ["merror", "mlogloss"],
        "tree_method": "hist",
        "early_stopping_rounds": 30,
        "random_state": SEED,
        "n_jobs": 8,
    }
    model = XGBClassifier(**parameters)
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    training_seconds = time.perf_counter() - started

    results: dict[str, Any] = {}
    reports: dict[str, str] = {}
    for split, x_values, y_values, frame in (
        ("validation", x_validation, y_validation, validation),
        ("test", x_test, y_test, test),
    ):
        probabilities = model.predict_proba(x_values)
        predicted = probabilities.argmax(axis=1)
        results[split] = metrics_for(y_values, predicted, probabilities)
        reports[split] = classification_report(
            y_values,
            predicted,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
        save_predictions(
            frame, probabilities, predicted,
            args.output_dir / f"{split}_predictions.csv",
        )
        plot_confusion(
            results[split]["confusion_matrix"],
            f"{split.title()} confusion matrix",
            args.output_dir / f"{split}_confusion_matrix.png",
        )

    importance = feature_importance(model, feature_columns)
    importance.to_csv(
        args.output_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_importance(
        importance, args.output_dir / "feature_importance.png"
    )

    model_path = args.model_dir / "risk_classifier.json"
    model.save_model(model_path)
    input_files = {
        name: {
            "path": str(
                (args.data_dir / f"{name}.csv").resolve().relative_to(ROOT.resolve())
            ),
            "sha256": sha256(args.data_dir / f"{name}.csv"),
            "rows": len(frames[name]),
        }
        for name in ("train", "validation", "test")
    }
    metadata = {
        "model_type": "XGBClassifier",
        "task": "three_class_rule_supervised_risk_classification",
        "labels": {"0": "低风险", "1": "中风险", "2": "高风险"},
        "risk_score_formula": (
            "0.15*P(low)+0.55*P(medium)+0.90*P(high)"
        ),
        "synthetic_data_warning": (
            "All labels are generated from deterministic rules; metrics measure "
            "rule imitation on synthetic scenarios, not real-road generalization."
        ),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "target": target,
        "parameters": parameters,
        "best_iteration": int(model.best_iteration),
        "best_score": float(model.best_score),
        "training_seconds": training_seconds,
        "random_seed": SEED,
        "input_files": input_files,
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "metrics": results,
        "feature_config": config,
    }
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_text = "\n\n".join(
        f"[{split}]\n{report}" for split, report in reports.items()
    )
    (args.output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    summary = f"""# XGBoost风险模型训练结果

- 训练样本：{len(train)}
- 验证样本：{len(validation)}
- 测试样本：{len(test)}
- 训练特征：{len(feature_columns)}
- 最佳迭代：{model.best_iteration}
- 训练耗时：{training_seconds:.4f} 秒
- 验证集宏平均F1：{results['validation']['macro_f1']:.4f}
- 测试集宏平均F1：{results['test']['macro_f1']:.4f}
- 测试集高风险召回率：{results['test']['high_risk_recall']:.4f}
- 测试集高风险漏报：{results['test']['high_risk_missed_count']}

模型使用规则生成的合成标签训练。上述指标只能证明模型能够拟合当前规则与合成场景，
不能证明真实道路安全性或泛化能力。部署时仍应保留硬规则兜底，并取更高风险等级。
"""
    (args.output_dir / "training_summary.md").write_text(
        summary, encoding="utf-8"
    )
    print(json.dumps({
        "training_seconds": training_seconds,
        "best_iteration": model.best_iteration,
        "validation": results["validation"],
        "test": results["test"],
        "model_path": str(model_path),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
