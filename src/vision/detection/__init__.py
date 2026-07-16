"""目标检测子模块。"""

from src.vision.detection.class_mapping import CLASS_BASE_RISK, TARGET_CLASS_MAPPING, TARGET_CLASSES
from src.vision.detection.postprocess import yolo_result_to_detection_results


def __getattr__(name: str):
    if name == "build_detector_from_config":
        from src.vision.detection.detector import build_detector_from_config
        return build_detector_from_config
    raise AttributeError(name)

__all__ = [
    "CLASS_BASE_RISK",
    "TARGET_CLASS_MAPPING",
    "TARGET_CLASSES",
    "build_detector_from_config",
    "yolo_result_to_detection_results",
]
