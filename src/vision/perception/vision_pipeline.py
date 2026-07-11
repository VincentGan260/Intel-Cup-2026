"""视觉主流程：检测 + 分割 + 可行驶区域标记 + 视觉风险初判。"""

from __future__ import annotations

import time
import numpy as np

from src.vision.common.interfaces import BaseDetector, BaseSegmenter
from src.vision.common.types import VisionResult
from src.vision.perception.target_on_road import attach_drivable_area_flag
from src.vision.risk.visual_risk import attach_visual_risk


class VisionPipeline:
    """
    视觉主流程：
    1. 目标检测
    2. 语义分割
    3. 判断目标是否在可行驶区域
    4. 计算视觉风险
    """

    def __init__(
        self,
        detector: BaseDetector,
        segmenter: BaseSegmenter | None = None,
        enable_segmentation: bool = True,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.enable_segmentation = enable_segmentation

    def process(self, frame: np.ndarray) -> VisionResult:
        image_height, image_width = frame.shape[:2]

        detection_start = time.perf_counter()
        detections = self.detector.infer(frame)
        detection_ms = (time.perf_counter() - detection_start) * 1000.0

        segmentation = None
        drivable_mask = None
        segmentation_ms = 0.0

        if self.enable_segmentation and self.segmenter is not None:
            segmentation_start = time.perf_counter()
            segmentation = self.segmenter.infer(frame)
            segmentation_ms = (time.perf_counter() - segmentation_start) * 1000.0
            drivable_mask = segmentation.drivable_mask

            detections = attach_drivable_area_flag(
                detections,
                drivable_mask,
            )

        detections = attach_visual_risk(
            detections,
            image_width,
            image_height,
        )

        max_visual_risk = max(
            [det.visual_risk or 0.0 for det in detections],
            default=0.0,
        )

        return VisionResult(
            detections=detections,
            segmentation=segmentation,
            drivable_mask=drivable_mask,
            max_visual_risk=max_visual_risk,
            detection_inference_ms=detection_ms,
            segmentation_inference_ms=segmentation_ms,
        )
