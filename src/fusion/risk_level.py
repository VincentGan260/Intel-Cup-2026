"""风险等级判断。

根据综合风险分数和配置阈值确定风险等级：
  - level=0: 低风险  (risk_score < low_threshold)
  - level=1: 中风险  (low_threshold <= risk_score < high_threshold)
  - level=2: 高风险  (risk_score >= high_threshold)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(path: str) -> str:
    """将相对路径解析为基于项目根的绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p)


def _load_thresholds(config_path: str = "configs/risk_params.yaml") -> Dict[str, float]:
    """从配置文件加载风险阈值。"""
    with open(_resolve_path(config_path), "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params["risk_thresholds"]


def determine_risk_level(
    risk_score: float,
    low_threshold: float = 0.30,
    high_threshold: float = 0.70,
) -> Tuple[int, str]:
    """根据风险分数确定等级。

    Args:
        risk_score: 综合风险分数 [0, 1]
        low_threshold: 低风险上限（不含）
        high_threshold: 高风险下限（含）

    Returns:
        (level, label)
        level 为 0/1/2，label 为 "低风险"/"中风险"/"高风险"
    """
    if risk_score >= high_threshold:
        return 2, "高风险"
    elif risk_score >= low_threshold:
        return 1, "中风险"
    else:
        return 0, "低风险"


class RiskLevelClassifier:
    """风险等级分类器，支持从配置文件加载阈值。"""

    def __init__(self, config_path: str = "configs/risk_params.yaml") -> None:
        thresholds = _load_thresholds(config_path)
        self.low_threshold = thresholds.get("low", 0.30)
        self.high_threshold = thresholds.get("high", 0.70)

    def classify(self, risk_score: float) -> Tuple[int, str]:
        """对风险分数进行等级分类。

        Args:
            risk_score: 综合风险分数 [0, 1]

        Returns:
            (level, label) 例如 (0, "低风险") / (1, "中风险") / (2, "高风险")
        """
        return determine_risk_level(
            risk_score,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )
