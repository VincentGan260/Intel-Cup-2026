"""视觉主流程：检测 + 分割 + 可行驶区域标记 + 视觉风险初判。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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
        enable_parallel_inference: bool = True,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.enable_segmentation = enable_segmentation
        self.enable_parallel_inference = enable_parallel_inference
        self._executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision-pipeline")
            if enable_parallel_inference else None
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _infer_detection(self, frame: np.ndarray):
        start = time.perf_counter()
        result = self.detector.infer(frame)
        return result, (time.perf_counter() - start) * 1000.0

    def _infer_segmentation(self, frame: np.ndarray):
        start = time.perf_counter()
        result = self.segmenter.infer(frame)  # type: ignore[union-attr]
        return result, (time.perf_counter() - start) * 1000.0

    def process(self, frame: np.ndarray) -> VisionResult:
        pipeline_start = time.perf_counter()
        image_height, image_width = frame.shape[:2]

        segmentation = None
        drivable_mask = None
        segmentation_ms = 0.0
        should_segment = self.enable_segmentation and self.segmenter is not None

        if should_segment and self._executor is not None:
            det_future = self._executor.submit(self._infer_detection, frame)
            seg_future = self._executor.submit(self._infer_segmentation, frame)
            detections, detection_ms = det_future.result()
            segmentation, segmentation_ms = seg_future.result()
        else:
            detections, detection_ms = self._infer_detection(frame)
            if should_segment:
                segmentation, segmentation_ms = self._infer_segmentation(frame)

        if segmentation is not None:
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
            pipeline_inference_ms=(time.perf_counter() - pipeline_start) * 1000.0,
        )
