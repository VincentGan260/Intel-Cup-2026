"""检测子模块内的具体模型实现。"""

from src.vision.detection.models.yolo_openvino import YoloOpenVinoDetector
from src.vision.detection.models.yolo_ultralytics import YoloUltralyticsDetector

__all__ = ["YoloUltralyticsDetector", "YoloOpenVinoDetector"]
