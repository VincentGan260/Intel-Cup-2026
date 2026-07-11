from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from src.vision.common.interfaces import BaseDetector
from src.vision.common.types import DetectionResult
from src.vision.detection.class_mapping import OBSTACLE_CLASS, TARGET_CLASS_MAPPING


class YoloOpenVinoDetector(BaseDetector):
    """Native OpenVINO YOLO detector.

    Do not route this through Ultralytics `predict()`: `device=NPU` there is
    interpreted as a PyTorch/Ascend device and raises a misleading torch_npu
    error. Here `NPU` is passed directly to OpenVINO `compile_model()`.
    """

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
        from openvino.runtime import Core

        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = str(device or "CPU").upper()
        self.save = bool(save)
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.risk_mapping = risk_mapping if risk_mapping else {}
        self.names = self._load_names()

        xml_path = self._resolve_xml_path(self.model_path)
        self.core = Core()
        model = self.core.read_model(str(xml_path))
        self.input = model.inputs[0]
        self.output = model.outputs[0]
        self.compiled = self.core.compile_model(model, self.device)
        print(f"[YoloOpenVinoDetector] loaded {xml_path} on OpenVINO {self.device}")

    def _resolve_xml_path(self, path: Path) -> Path:
        if path.is_file() and path.suffix.lower() == ".xml":
            return path
        if path.is_dir():
            xml_files = sorted(path.glob("*.xml"))
            if xml_files:
                return xml_files[0]
        raise FileNotFoundError(f"OpenVINO XML not found: {path}")

    def _load_names(self) -> dict[int, str]:
        meta = self.model_path / "metadata.yaml" if self.model_path.is_dir() else self.model_path.with_name("metadata.yaml")
        if meta.is_file():
            try:
                data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
                names = data.get("names") or {}
                return {int(k): str(v) for k, v in names.items()}
            except Exception:
                pass
        return {0: OBSTACLE_CLASS}

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        h, w = frame.shape[:2]
        scale = min(self.image_size / max(1, w), self.image_size / max(1, h))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.image_size - new_w
        pad_h = self.image_size - new_h
        left = pad_w / 2.0
        top = pad_h / 2.0
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        x0, y0 = int(round(left - 0.1)), int(round(top - 0.1))
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas, scale, (float(x0), float(y0))

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        image, scale, pad = self._letterbox(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(image, (2, 0, 1))[None, ...]
        return blob, scale, pad

    def _unpad_xyxy(self, boxes: np.ndarray, scale: float, pad: tuple[float, float],
                    frame_shape: tuple[int, int]) -> np.ndarray:
        boxes = boxes.astype(np.float32).copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / max(scale, 1e-9)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / max(scale, 1e-9)
        h, w = frame_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        return boxes

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
            order = order[1:][iou <= iou_threshold]
        return keep

    def _decode(self, raw_output: np.ndarray, scale: float, pad: tuple[float, float],
                frame_shape: tuple[int, int]) -> list[DetectionResult]:
        pred = np.asarray(raw_output)
        pred = np.squeeze(pred)
        if pred.ndim == 1:
            pred = pred[None, :]
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1] and pred.shape[0] in (5, 6, 7, 84, 85):
            pred = pred.T
        if pred.ndim != 2 or pred.shape[1] < 5:
            return []

        if pred.shape[1] == 7:
            boxes = pred[:, 3:7]
            scores = pred[:, 2]
            class_ids = pred[:, 1].astype(int)
        elif pred.shape[1] == 6:
            boxes = pred[:, :4]
            scores = pred[:, 4]
            class_ids = pred[:, 5].astype(int)
        else:
            boxes = pred[:, :4]
            class_scores = pred[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(len(class_scores)), class_ids]
            cx, cy, bw, bh = boxes.T
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        mask = scores >= self.confidence
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if len(boxes) == 0:
            return []

        if float(np.nanmax(boxes)) <= 1.5:
            boxes = boxes * float(self.image_size)
        boxes = self._unpad_xyxy(boxes, scale, pad, frame_shape)
        keep = self._nms(boxes, scores)
        detections: list[DetectionResult] = []
        for index in keep:
            class_name = self.names.get(int(class_ids[index]), f"cls_{int(class_ids[index])}")
            risk = self.risk_mapping.get(class_name)
            if risk is None:
                risk = OBSTACLE_CLASS if class_name == OBSTACLE_CLASS else TARGET_CLASS_MAPPING.get(class_name, "unknown")
            detections.append(DetectionResult(
                class_name=class_name,
                confidence=float(scores[index]),
                bbox=tuple(float(x) for x in boxes[index]),
                risk_class=risk,
            ))
        return detections

    def infer(self, frame: np.ndarray) -> list[DetectionResult]:
        blob, scale, pad = self._preprocess(frame)
        result = self.compiled([blob])[self.output]
        return self._decode(result, scale, pad, frame.shape[:2])
