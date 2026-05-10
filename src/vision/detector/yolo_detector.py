"""YOLO 目标检测封装。

这个类只负责调用 Ultralytics YOLO 完成模型加载、推理、验证、训练和导出。
模型路径、置信度、输出目录、类别映射等可变参数都由配置文件传入。
"""

from pathlib import Path
from typing import Any

from ultralytics import YOLO, settings

from src.vision.common.types import ImageDetections
from src.vision.detector.postprocess import (
    build_model_to_project_class,
    yolo_result_to_detections,
)


class YoloDetector:
    """项目内部使用的 YOLO 检测器。"""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        image_size: int,
        device: str | None,
        runs_dir: str | None,
        project_to_model_classes: dict[str, list[str]],
        keep_target_classes_only: bool,
        other_project_class: str,
    ) -> None:
        """根据配置初始化 YOLO 检测器。

        参数全部由上层脚本从 `configs/vision/detection.yaml` 读取后传入，
        因此这里不设置模型路径、置信度、图片尺寸等业务默认值。
        """

        if runs_dir:
            # runs_dir 是 Ultralytics 的全局输出目录设置。
            settings.update({"runs_dir": str(Path(runs_dir))})

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.model_to_project_class = build_model_to_project_class(
            project_to_model_classes
        )
        self.keep_target_classes_only = keep_target_classes_only
        self.other_project_class = other_project_class

    def predict(
        self,
        source: str | int,
        save: bool,
        project: str,
        name: str,
        exist_ok: bool,
    ) -> list[ImageDetections]:
        """运行目标检测，并返回项目内部统一格式。"""

        raw_results = self.model.predict(
            source=source,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            save=save,
            project=project,
            name=name,
            exist_ok=exist_ok,
        )

        return [
            yolo_result_to_detections(
                result,
                model_to_project_class=self.model_to_project_class,
                keep_target_classes_only=self.keep_target_classes_only,
                other_project_class=self.other_project_class,
            )
            for result in raw_results
        ]

    def val(self, data: str) -> Any:
        """使用指定数据集验证模型效果。"""

        return self.model.val(data=data)

    def train(self, data: str, epochs: int) -> Any:
        """使用指定数据集训练或微调模型。"""

        return self.model.train(data=data, epochs=epochs)

    def export(self, export_format: str) -> str:
        """把模型导出为 ONNX、OpenVINO 等格式。"""

        return str(self.model.export(format=export_format))
