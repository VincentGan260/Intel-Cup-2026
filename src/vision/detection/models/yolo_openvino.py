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
        save: bool = False,
        project: str = "runs/vision/detect",
        name: str = "pred",
        exist_ok: bool = True,
        risk_mapping: dict | None = None,
    ):
        self.model = YOLO(model_path, task="detect")
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.save = save
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.risk_mapping = risk_mapping if risk_mapping else {}

    def infer(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            save=self.save,
            project=self.project,
            name=self.name,
            exist_ok=self.exist_ok,
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
                risk = self.risk_mapping.get(class_name, "unknown")
                detections.append(DetectionResult(
                    class_name=class_name,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    risk_class=risk,
                ))
        return detections