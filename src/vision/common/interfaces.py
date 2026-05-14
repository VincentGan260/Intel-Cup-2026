"""视觉模型抽象接口：主流程只依赖 BaseDetector / BaseSegmenter，不依赖 YOLO、PIDNet 等具体实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.vision.common.types import DetectionResult, SegmentationResult


class BaseDetector(ABC):
    """目标检测抽象基类。"""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        """输入 BGR 图像，输出目标检测结果。"""
        raise NotImplementedError


class BaseSegmenter(ABC):
    """语义分割抽象基类。"""

    @abstractmethod
    def infer(self, frame: np.ndarray) -> SegmentationResult:
        """输入 BGR 图像，输出可行驶区域 mask。"""
        raise NotImplementedError
