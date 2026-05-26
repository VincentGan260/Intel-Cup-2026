from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from src.vision.common.interfaces import BaseDetector
from src.vision.common.types import DetectionResult


class YoloOpenVinoDetector(BaseDetector):
    """使用 Ultralytics 加载 OpenVINO IR 模型的检测器。"""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        image_size: int = 640,
        device: str = "CPU",
    ):
        # Ultralytics 自动识别 .xml 或模型目录
        self.model = YOLO(model_path, task="detect")
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def infer(self, frame: np.ndarray) -> list[DetectionResult]:
        """
        输入 BGR 图像，返回 DetectionResult 列表。
        """
        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names.get(cls_id, f"cls_{cls_id}")

                detections.append(DetectionResult(
                    class_name=class_name,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    # 如果还有其他字段（如 risk_class），请根据你的实际 DetectionResult 补充
                ))
        return detections