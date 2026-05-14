"""检测与分割融合：为检测框附加是否在可行驶区域等信息。"""

from __future__ import annotations

from typing import List

import numpy as np

from src.vision.common.types import DetectionResult
from src.vision.segmentation.mask_utils import is_bbox_on_drivable_area


def attach_drivable_area_flag(
    detections: List[DetectionResult],
    drivable_mask: np.ndarray,
) -> List[DetectionResult]:
    """
    给每个检测目标添加 in_drivable_area 字段。
    """
    for det in detections:
        det.in_drivable_area = is_bbox_on_drivable_area(
            drivable_mask,
            det.bbox,
        )

    return detections
