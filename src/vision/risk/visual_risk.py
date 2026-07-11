"""视觉风险初判（仅视觉因素，不含雷达 / IMU / GPS / TTC）。

面向「前向骑行单帧图像」场景重做的打分模型：
    visual_risk = (w_class·class + w_prox·proximity + w_lat·lateral
                   + w_driv·drivable + w_size·size) · conf_factor

各因子均归一化到 [0, 1]，权重默认和为 1（内部会再归一化，方便后续标定时
随意改动而不必手动凑成 1）。这一项对应方案书风险公式里的视觉障碍物项 R_obs。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.vision.common.types import BBox, DetectionResult
from src.vision.detection.class_mapping import CLASS_BASE_RISK


@dataclass(frozen=True)
class VisualRiskConfig:
    """视觉风险打分的可调参数（后续用真实骑行数据标定）。"""

    # —— 因子权重（和会在内部归一化，无需手动凑成 1）——
    w_class: float = 0.25
    w_proximity: float = 0.30
    w_lateral: float = 0.25
    w_drivable: float = 0.12
    w_size: float = 0.08

    # —— 纵向接近度：bbox 底边归一化高度 y2/H 低于该值视为地平线附近（远处），风险≈0 ——
    horizon_ratio: float = 0.40

    # —— 横向在途：中央行进走廊宽度占图像宽度的比例 ——
    corridor_ratio: float = 0.50

    # —— 尺寸风险：bbox 面积占比达到该值即饱和为最大尺寸风险 ——
    size_saturation: float = 0.20

    # —— 可行驶区域风险取值 ——
    drivable_in: float = 1.00
    drivable_out: float = 0.30
    drivable_unknown: float = 0.50

    # —— 置信度调制：conf<=floor 衰减到 min_factor，conf>=full 不衰减，之间线性 ——
    use_confidence: bool = True
    conf_floor: float = 0.25
    conf_full: float = 0.60
    conf_min_factor: float = 0.50

    # 未在 CLASS_BASE_RISK 中命中的类别的兜底基础风险
    default_class_risk: float = 0.30


DEFAULT_VISUAL_RISK_CONFIG = VisualRiskConfig()


def _clamp01(value: float) -> float:
    return float(max(0.0, min(value, 1.0)))


def calculate_class_risk(risk_class: str, cfg: VisualRiskConfig) -> float:
    """类别基础危险度。最终输出类别无关：所有 obstacle 取同一较高基础危险度
    （见 class_mapping.CLASS_BASE_RISK）。差异化威胁由接近度 / 在途 / 尺寸等因子体现。"""
    return float(CLASS_BASE_RISK.get(risk_class, cfg.default_class_risk))


def calculate_proximity_risk(
    bbox: BBox,
    image_height: int,
    cfg: VisualRiskConfig,
) -> float:
    """纵向接近度：bbox 底边越靠近画面底部，目标越近、风险越高。

    单目前向相机里，目标「落地点」的纵向位置是比 bbox 面积更稳的距离线索
    （不受类别体型差异影响）。底边在地平线 horizon_ratio 以上的目标视为远处。
    """
    if image_height <= 0:
        return 0.0

    y2 = bbox[3]
    bottom_ratio = y2 / image_height  # 0=画面顶部, 1=画面底部

    span = 1.0 - cfg.horizon_ratio
    if span <= 0:
        return _clamp01(bottom_ratio)

    return _clamp01((bottom_ratio - cfg.horizon_ratio) / span)


def calculate_lateral_risk(
    bbox: BBox,
    image_width: int,
    cfg: VisualRiskConfig,
) -> float:
    """横向在途度：bbox 与中央行进走廊的重叠程度。

    相比「只看中心点离画面中心的距离」，这里用区间重叠，能正确反映
    「整车横在前方挡道」这类大目标——只要把走廊填满即视为完全在途。
    """
    if image_width <= 0:
        return 0.0

    x1, _, x2, _ = bbox
    bbox_w = max(0.0, x2 - x1)
    if bbox_w <= 0:
        return 0.0

    center = image_width / 2.0
    corridor_w = max(1e-6, cfg.corridor_ratio * image_width)
    c1 = center - corridor_w / 2.0
    c2 = center + corridor_w / 2.0

    overlap = max(0.0, min(x2, c2) - max(x1, c1))
    # 分母取「目标宽」与「走廊宽」的较小者：
    #   目标比走廊窄 -> 目标完全进入走廊即 1；
    #   目标比走廊宽 -> 填满走廊即 1。两种情况「完全挡道」都得满分。
    denom = min(bbox_w, corridor_w)
    return _clamp01(overlap / denom)


def calculate_size_risk(
    bbox: BBox,
    image_width: int,
    image_height: int,
    cfg: VisualRiskConfig,
) -> float:
    """视觉占比（粗略威胁体量）。仅作降权辅助项，精确距离应交由雷达/深度相机。"""
    x1, y1, x2, y2 = bbox

    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = image_width * image_height
    if image_area <= 0 or cfg.size_saturation <= 0:
        return 0.0

    ratio = bbox_area / image_area
    return _clamp01(ratio / cfg.size_saturation)


def calculate_drivable_risk(
    in_drivable_area: Optional[bool],
    cfg: VisualRiskConfig,
) -> float:
    """落在可行驶区域内的目标更危险；未知（未开分割）取中间值。"""
    if in_drivable_area is True:
        return cfg.drivable_in
    if in_drivable_area is False:
        return cfg.drivable_out
    return cfg.drivable_unknown


def calculate_confidence_factor(confidence: float, cfg: VisualRiskConfig) -> float:
    """置信度调制系数 ∈ [conf_min_factor, 1]：低置信度目标风险打折，抑制误报。"""
    if not cfg.use_confidence:
        return 1.0

    span = cfg.conf_full - cfg.conf_floor
    if span <= 0:
        ramp = 1.0 if confidence >= cfg.conf_full else 0.0
    else:
        ramp = _clamp01((confidence - cfg.conf_floor) / span)

    return cfg.conf_min_factor + (1.0 - cfg.conf_min_factor) * ramp


def calculate_detection_visual_risk(
    det: DetectionResult,
    image_width: int,
    image_height: int,
    cfg: VisualRiskConfig = DEFAULT_VISUAL_RISK_CONFIG,
) -> float:
    """计算单个目标的视觉风险（仅视觉初判，不含雷达距离 / TTC / IMU / GPS）。"""
    weights = (cfg.w_class, cfg.w_proximity, cfg.w_lateral, cfg.w_drivable, cfg.w_size)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return 0.0

    factors = (
        calculate_class_risk(det.risk_class, cfg),
        calculate_proximity_risk(det.bbox, image_height, cfg),
        calculate_lateral_risk(det.bbox, image_width, cfg),
        calculate_drivable_risk(det.in_drivable_area, cfg),
        calculate_size_risk(det.bbox, image_width, image_height, cfg),
    )

    base = sum(w * f for w, f in zip(weights, factors)) / weight_sum
    risk = base * calculate_confidence_factor(det.confidence, cfg)

    return _clamp01(risk)


def attach_visual_risk(
    detections: List[DetectionResult],
    image_width: int,
    image_height: int,
    cfg: VisualRiskConfig = DEFAULT_VISUAL_RISK_CONFIG,
) -> List[DetectionResult]:
    """给每个检测目标写入 visual_risk 字段。"""
    for det in detections:
        det.visual_risk = calculate_detection_visual_risk(
            det,
            image_width,
            image_height,
            cfg,
        )

    return detections
