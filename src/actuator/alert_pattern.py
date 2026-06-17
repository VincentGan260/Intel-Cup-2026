"""风险等级至震动模式的映射定义。

将融合模块输出的风险等级转换为具体的 DRV2605 震动参数。
"""
from __future__ import annotations

from typing import Dict, List

from src.fusion.data_types import MotorCommand


# 各等级震动描述
ALERT_LABELS: Dict[int, str] = {
    0: "静默",
    1: "中风险 - 短促轻震",
    2: "高风险 - 持续强震",
}


class AlertPattern:
    """单次震动模式。"""

    def __init__(
        self,
        risk_level: int,
        effect_ids: List[int],
        pulse_duration_sec: float,
        repeat_count: int = 1,
        pattern_name: str = "silent",
    ) -> None:
        self.risk_level = risk_level
        self.effect_ids = effect_ids
        self.pulse_duration_sec = pulse_duration_sec
        self.repeat_count = repeat_count
        self.pattern_name = pattern_name

    def to_command(self, risk_score: float = 0.0) -> MotorCommand:
        return MotorCommand(
            risk_level=self.risk_level,
            risk_score=risk_score,
            pattern=self.pattern_name,
            duration_ms=int(self.pulse_duration_sec * 1000 * self.repeat_count),
            intensity=1.0 if self.risk_level == 2 else 0.5,
        )


# — 预定义模式 —


def pattern_for_level(
    level: int,
    risk_score: float = 0.0,
    medium_effect_ids: List[int] | None = None,
    medium_duration: float = 0.05,
    medium_repeat: int = 3,
    high_effect_ids: List[int] | None = None,
    high_duration: float = 0.5,
) -> MotorCommand:
    """根据风险等级获取对应的马达控制指令。

    Args:
        level: 风险等级 0/1/2
        risk_score: 综合风险分数
        medium_effect_ids: 中风险效果编号列表
        medium_duration: 中风险单次脉冲时长（秒）
        medium_repeat: 中风险重复次数
        high_effect_ids: 高风险效果编号列表
        high_duration: 高风险震动时长（秒）

    Returns:
        对应的 MotorCommand
    """
    if level == 0:
        return MotorCommand(risk_level=0, risk_score=risk_score, pattern="silent")

    elif level == 1:
        pattern = AlertPattern(
            risk_level=1,
            effect_ids=medium_effect_ids or [1],
            pulse_duration_sec=medium_duration,
            repeat_count=medium_repeat,
            pattern_name="short_pulse",
        )
        return pattern.to_command(risk_score)

    elif level == 2:
        pattern = AlertPattern(
            risk_level=2,
            effect_ids=high_effect_ids or [64],
            pulse_duration_sec=high_duration,
            repeat_count=1,
            pattern_name="continuous_strong",
        )
        return pattern.to_command(risk_score)

    return MotorCommand(risk_level=0, pattern="silent")
