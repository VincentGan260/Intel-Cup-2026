"""目标检测后处理：Ultralytics 输出 -> `DetectionResult` 列表。"""

from __future__ import annotations

from typing import Any, List

from src.vision.common.types import DetectionResult
from src.vision.detection.class_mapping import TARGET_CLASS_MAPPING, TARGET_CLASSES


def yolo_result_to_detection_results(yolo_result: Any) -> List[DetectionResult]:
    """将单帧 YOLO `Results` 转为项目 `DetectionResult`，仅保留 `TARGET_CLASSES`。"""
    names = yolo_result.names
    out: list[DetectionResult] = []

    boxes = getattr(yolo_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out

    class_ids = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    bboxes = boxes.xyxy.cpu().tolist()

    for class_id, confidence, bbox in zip(class_ids, confidences, bboxes):
        class_name = names[int(class_id)]
        if class_name not in TARGET_CLASSES:
            continue
        risk_class = TARGET_CLASS_MAPPING[class_name]
        out.append(
            DetectionResult(
                class_name=class_name,
                risk_class=risk_class,
                confidence=float(confidence),
                bbox=tuple(float(x) for x in bbox),
            )
        )
    return out
