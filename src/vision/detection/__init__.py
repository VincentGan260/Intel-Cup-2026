"""目标检测子模块。"""

from src.vision.detection.class_mapping import CLASS_BASE_RISK, TARGET_CLASS_MAPPING, TARGET_CLASSES
from src.vision.detection.detector import build_detector_from_config
from src.vision.detection.models import YoloOpenVinoDetector, YoloUltralyticsDetector
from src.vision.detection.postprocess import yolo_result_to_detection_results

__all__ = [
    "CLASS_BASE_RISK",
    "TARGET_CLASS_MAPPING",
    "TARGET_CLASSES",
    "build_detector_from_config",
    "YoloUltralyticsDetector",
    "YoloOpenVinoDetector",
    "yolo_result_to_detection_results",
]
