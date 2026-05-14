"""YOLO OpenVINO IR 推理（占位，后续实现 `BaseDetector`）。"""

from __future__ import annotations

from typing import List

import numpy as np

from src.vision.common.interfaces import BaseDetector
from src.vision.common.types import DetectionResult


class YoloOpenVinoDetector(BaseDetector):
    """使用 OpenVINO 运行 YOLO IR；与 Ultralytics 解耦，后续补全。"""

    def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        raise NotImplementedError(
            "YoloOpenVinoDetector 尚未实现。请暂时使用 YoloUltralyticsDetector，"
            "或在此处接入 YOLO OpenVINO IR 推理与后处理。"
        )
