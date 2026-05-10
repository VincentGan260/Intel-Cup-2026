"""YOLO 检测结果后处理。

这个文件负责把 Ultralytics 原始输出转换成项目内部更容易使用的数据结构。
类别映射规则来自 `configs/vision/detection.yaml`，不要在这里写死具体类别。
"""

from typing import Any

from src.vision.common.types import Detection, ImageDetections


def build_model_to_project_class(
    project_to_model_classes: dict[str, list[str]],
) -> dict[str, str]:
    """把配置中的“项目类别 -> 模型类别列表”转换成“模型类别 -> 项目类别”。

    配置文件里更适合人阅读的形式是：
    pedestrian: [person]

    推理后处理时更方便查找的形式是：
    person: pedestrian
    """

    model_to_project_class: dict[str, str] = {}
    for project_class, model_classes in project_to_model_classes.items():
        for model_class in model_classes:
            model_to_project_class[model_class] = project_class

    return model_to_project_class


def yolo_result_to_detections(
    yolo_result: Any,
    model_to_project_class: dict[str, str],
    keep_target_classes_only: bool,
    other_project_class: str,
) -> ImageDetections:
    """把单张图片的 YOLO 输出转换成项目统一检测结果。"""

    names = yolo_result.names
    detections: list[Detection] = []

    boxes = getattr(yolo_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ImageDetections(source=str(yolo_result.path), detections=detections)

    # Ultralytics 输出是 tensor，这里转成 Python list，方便后续整理和打印。
    class_ids = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    bboxes = boxes.xyxy.cpu().tolist()

    for class_id, confidence, bbox in zip(class_ids, confidences, bboxes):
        class_name = names[int(class_id)]
        project_class = model_to_project_class.get(class_name, other_project_class)

        # 如果配置要求只保留目标类别，则过滤掉项目暂时不关心的物体。
        if keep_target_classes_only and class_name not in model_to_project_class:
            continue

        detections.append(
            Detection(
                class_name=class_name,
                project_class=project_class,
                confidence=float(confidence),
                bbox_xyxy=tuple(float(value) for value in bbox),
            )
        )

    return ImageDetections(source=str(yolo_result.path), detections=detections)
