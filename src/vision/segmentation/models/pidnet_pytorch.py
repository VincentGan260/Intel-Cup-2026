"""PIDNet-S PyTorch 推理（占位，后续实现 `BaseSegmenter`）。"""

from __future__ import annotations

import numpy as np

from src.vision.common.interfaces import BaseSegmenter
from src.vision.common.types import SegmentationResult


class PidnetPytorchSegmenter(BaseSegmenter):
    def infer(self, frame: np.ndarray) -> SegmentationResult:
        raise NotImplementedError("PidnetPytorchSegmenter 尚未实现，请使用 RoadAdasOpenVinoSegmenter。")
