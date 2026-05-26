from src.vision.common.interfaces import BaseDetector
from src.vision.common.types import DetectionResult
from ultralytics import YOLO
import numpy as np

class YoloOpenVinoDetector(BaseDetector):
    def __init__(self, model_path, confidence=0.25, image_size=640, device="CPU", risk_mapping=None):
        self.model = YOLO(model_path, task="detect")
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.risk_mapping = risk_mapping if risk_mapping else {}

    def infer(self, frame: np.ndarray) -> list[DetectionResult]:   # 确保有这一行
        results = self.model(frame, conf=self.confidence, imgsz=self.image_size, device=self.device, verbose=False)
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