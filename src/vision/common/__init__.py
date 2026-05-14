"""视觉公共模块：类型、接口、预处理、可视化。"""

from src.vision.common.interfaces import BaseDetector, BaseSegmenter
from src.vision.common.preprocess import (
    ensure_bgr_uint8,
    image_size_wh,
    load_image_bgr_from_source,
    read_image_bgr,
)
from src.vision.common.types import BBox, DetectionResult, SegmentationResult, VisionResult

__all__ = [
    "BBox",
    "BaseDetector",
    "BaseSegmenter",
    "DetectionResult",
    "SegmentationResult",
    "VisionResult",
    "ensure_bgr_uint8",
    "read_image_bgr",
    "load_image_bgr_from_source",
    "image_size_wh",
]
