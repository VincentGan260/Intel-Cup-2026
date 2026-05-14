"""PIDNet-S OpenVINO 推理（占位，后续实现 `BaseSegmenter`）。"""

from __future__ import annotations

import numpy as np

from src.vision.common.interfaces import BaseSegmenter
from src.vision.common.types import SegmentationResult


class PidnetOpenVinoSegmenter(BaseSegmenter):
    def infer(self, frame: np.ndarray) -> SegmentationResult:
        raise NotImplementedError("PidnetOpenVinoSegmenter 尚未实现，请使用 RoadAdasOpenVinoSegmenter。")
