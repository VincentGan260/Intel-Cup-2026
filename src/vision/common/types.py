"""视觉模块通用数据结构。

这些结构用于隔离第三方库输出，避免后续代码直接依赖 Ultralytics 的内部格式。
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Detection:
    """单个目标检测结果。

    class_name: 模型原始类别名，例如 person、car。
    project_class: 项目内部类别名，例如 pedestrian、motor_vehicle。
    confidence: 模型置信度。
    bbox_xyxy: 检测框坐标，格式为左上角和右下角 (x1, y1, x2, y2)。
    """

    class_name: str
    project_class: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageDetections:
    """单张图片或单帧画面的检测结果。"""

    source: str
    detections: list[Detection]
