"""骑手前向安全预警 — 视觉子包。"""

from src.vision.common.interfaces import BaseDetector, BaseSegmenter
from src.vision.common.types import BBox, DetectionResult, SegmentationResult, VisionResult


def __getattr__(name: str):
    if name == "VisionPipeline":
        from src.vision.perception.vision_pipeline import VisionPipeline
        return VisionPipeline
    raise AttributeError(name)

__all__ = [
    "BBox",
    "BaseDetector",
    "BaseSegmenter",
    "DetectionResult",
    "SegmentationResult",
    "VisionResult",
    "VisionPipeline",
]
