"""风险融合模型：四类风险项计算 + 自适应加权。

公式：
  R_obs   = vision 视觉障碍物风险    [0, 1]
  R_dist  = radar  距离 / TTC 风险    [0, 1]
  R_pose  = imu    姿态稳定性风险      [0, 1]
  R_speed = gps    速度风险            [0, 1]

  综合风险 R = w1 * R_obs + w2 * R_dist + w3 * R_pose + w4 * R_speed
  所有权重从 configs/risk_params.yaml 读取。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import yaml

from src.fusion.data_types import FusionInput


def _load_risk_params(path: str = "configs/risk_params.yaml") -> dict:
    """加载风险融合参数。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
#  单项风险计算
# ============================================================


def calculate_risk_obs(fusion: FusionInput) -> float:
    """视觉风险 R_obs [0, 1]。

    视觉未接入或视觉无效时返回 0。
    视觉接入后从 VisionData.max_visual_risk 获取。
    """
    if not fusion.vision_enabled:
        return 0.0
    if not fusion.vision.valid:
        return 0.0
    return max(0.0, min(1.0, fusion.vision.max_visual_risk))


def calculate_risk_dist(
    fusion: FusionInput,
    ttc_safe_sec: float = 5.0,
) -> float:
    """距离/TTC 风险 R_dist [0, 1]。

    策略：
      - 雷达异常（valid=False）或无目标 → R_dist=0
      - TTC > 0（有目标正在接近）:
        R_dist = 1 - min(TTC, ttc_safe) / ttc_safe
        即 TTC 越小风险越高，TTC >= ttc_safe 时风险 ≈ 0
      - 只有远离目标（TTC=-1）→ R_dist=0
    """
    radar = fusion.radar
    if not radar.valid or len(radar.targets) == 0:
        return 0.0

    # 有目标正在接近
    if radar.min_ttc > 0:
        r_dist = 1.0 - min(radar.min_ttc, ttc_safe_sec) / ttc_safe_sec
        return max(0.0, min(1.0, r_dist))

    # 有目标但都是远离的，再考虑距离
    if radar.nearest_distance_m > 0:
        # 非常近的目标（< 2m）即使远离也略有风险
        if radar.nearest_distance_m < 2.0:
            return max(0.0, 1.0 - radar.nearest_distance_m / 2.0)

    return 0.0


def calculate_risk_pose(fusion: FusionInput) -> float:
    """姿态风险 R_pose [0, 1]。

    R_pose = 0.4 * brake_score + 0.3 * bump_score + 0.3 * tilt_score

    当 IMU 无效时返回 0。
    """
    if not fusion.imu.valid:
        return 0.0

    imu = fusion.imu
    r_pose = (
        0.4 * imu.brake_score
        + 0.3 * imu.bump_score
        + 0.3 * imu.tilt_score
    )
    return max(0.0, min(1.0, r_pose))


def calculate_risk_speed(
    fusion: FusionInput,
    max_speed_kmh: float = 25.0,
) -> float:
    """速度风险 R_speed [0, 1]。

    R_speed = min(speed_kmh / max_speed_kmh, 1.0)

    GPS 无效时返回 0。
    """
    if not fusion.gps.valid:
        return 0.0

    speed_kmh = max(0.0, fusion.gps.speed_kmh)
    r_speed = min(speed_kmh / max_speed_kmh, 1.0)
    return max(0.0, min(1.0, r_speed))


# ============================================================
#  自适应权重
# ============================================================


def _compute_adaptive_weights(
    base_weights: Dict[str, float],
    modules_valid: Dict[str, bool],
    invalid_factor: float = 0.3,
) -> Dict[str, float]:
    """根据模块有效性计算自适应权重。

    - 有效的模块：保留完整权重
    - 无效的模块：权重衰减为 base_weight * invalid_factor
    - 最后所有权重重归一化，保证总和 = 1
    """
    adjusted = {}
    for key in base_weights:
        if modules_valid.get(key, True):
            adjusted[key] = base_weights[key]
        else:
            adjusted[key] = base_weights[key] * invalid_factor

    total = sum(adjusted.values())
    if total > 0:
        for key in adjusted:
            adjusted[key] /= total

    return adjusted


# ============================================================
#  综合风险计算
# ============================================================


class RiskModel:
    """综合风险融合模型。

    使用方式：
      1. 加载配置（或使用默认值）
      2. 调用 compute(fusion_input) 获得各项风险和综合评分
    """

    def __init__(self, config_path: str = "configs/risk_params.yaml") -> None:
        params = _load_risk_params(config_path)

        # 基础权重
        self.base_weights = params["risk_weights"]
        # 风险阈值
        self.thresholds = params["risk_thresholds"]
        # 自适应权重
        adaptive = params.get("adaptive_weights", {})
        self.adaptive_enabled = adaptive.get("enabled", True)
        self.invalid_weight_factor = adaptive.get("invalid_weight_factor", 0.3)
        # TTC 安全阈值
        dist = params.get("distance", {})
        self.ttc_safe_sec = dist.get("ttc_safe_sec", 5.0)
        # 速度上限
        speed = params.get("speed", {})
        self.max_speed_kmh = speed.get("max_speed_kmh", 25.0)

    def compute(self, fusion: FusionInput) -> Tuple[Dict[str, float], Dict[str, float]]:
        """执行一帧风险融合。

        Args:
            fusion: 同步后的 FusionInput

        Returns:
            (risk_items, weights_used)
            risk_items = {
                "R_obs":   float,   # 视觉风险
                "R_dist":  float,   # 距离风险
                "R_pose":  float,   # 姿态风险
                "R_speed": float,   # 速度风险
                "risk_score": float # 综合风险
            }
            weights_used = {
                "obs":   float,  # 实际使用的 obs 权重
                "dist":  float,  # 实际使用的 dist 权重
                "pose":  float,  # 实际使用的 pose 权重
                "speed": float,  # 实际使用的 speed 权重
            }
        """
        # 1. 计算各项风险
        r_obs = calculate_risk_obs(fusion)
        r_dist = calculate_risk_dist(fusion, self.ttc_safe_sec)
        r_pose = calculate_risk_pose(fusion)
        r_speed = calculate_risk_speed(fusion, self.max_speed_kmh)

        # 2. 判断各模块有效性（用于自适应加权）
        modules_valid = {
            "obs": fusion.vision_enabled and fusion.vision.valid,
            "dist": fusion.radar.valid and len(fusion.radar.targets) > 0,
            "pose": fusion.imu.valid,
            "speed": fusion.gps.valid,
        }

        # 3. 如果视觉未启用，手动标记 obs 模块为无效
        if not fusion.vision_enabled:
            modules_valid["obs"] = False

        # 4. 计算权重
        if self.adaptive_enabled:
            weights = _compute_adaptive_weights(
                self.base_weights, modules_valid, self.invalid_weight_factor
            )
        else:
            weights = dict(self.base_weights)
            # 非自适应模式下视觉未启用时，手动将 obs 权重均分给其他模块
            if not fusion.vision_enabled:
                obs_w = weights.pop("obs", 0.0)
                remaining_keys = [k for k in weights]
                if remaining_keys:
                    share = obs_w / len(remaining_keys)
                    for k in remaining_keys:
                        weights[k] += share

        # 5. 计算综合风险
        risk_score = (
            weights["obs"] * r_obs
            + weights["dist"] * r_dist
            + weights["pose"] * r_pose
            + weights["speed"] * r_speed
        )
        risk_score = max(0.0, min(1.0, risk_score))

        risk_items = {
            "R_obs": r_obs,
            "R_dist": r_dist,
            "R_pose": r_pose,
            "R_speed": r_speed,
            "risk_score": risk_score,
        }

        return risk_items, weights
