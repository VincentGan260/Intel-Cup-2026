"""视觉模块统一数据结构（与具体模型框架解耦）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class DetectionResult:
    """单目标检测结果。"""

    class_name: str
    risk_class: str
    confidence: float
    bbox: BBox
    in_drivable_area: Optional[bool] = None
    visual_risk: Optional[float] = None


@dataclass
class SegmentationResult:
    """语义分割结果：可行驶区域二值 mask 与原图同高宽。"""

    drivable_mask: np.ndarray
    raw_mask: Optional[np.ndarray] = None
    drivable_ratio: Optional[float] = None


@dataclass
class VisionResult:
    """视觉管线输出，供系统级 `src/fusion/` 等模块消费。"""

    detections: List[DetectionResult] = field(default_factory=list)
    segmentation: Optional[SegmentationResult] = None
    # 便于 fusion 直接读取，与 segmentation.drivable_mask 在分割开启时一致
    drivable_mask: Optional[np.ndarray] = None
    max_visual_risk: float = 0.0
