"""分割子模块内的具体模型实现。"""

from src.vision.segmentation.models.pidnet_openvino import PidnetOpenVinoSegmenter
from src.vision.segmentation.models.pidnet_pytorch import PidnetPytorchSegmenter
from src.vision.segmentation.models.road_adas_openvino import RoadAdasOpenVinoSegmenter

__all__ = [
    "RoadAdasOpenVinoSegmenter",
    "PidnetPytorchSegmenter",
    "PidnetOpenVinoSegmenter",
]
