"""Ultralytics YOLO（.pt）检测实现，实现 `BaseDetector`。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO, settings

from src.vision.common.interfaces import BaseDetector
from src.vision.common.types import DetectionResult
from src.vision.detection.postprocess import yolo_result_to_detection_results


class YoloUltralyticsDetector(BaseDetector):
    """YOLO PyTorch 权重推理；后续可换为 `YoloOpenVinoDetector` 而不改 `VisionPipeline`。"""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        image_size: int,
        device: Optional[str],
        runs_dir: str,
        *,
        predict_save: bool = False,
        predict_project: str = "yolo26n_demo",
        predict_name: str = "pred",
        predict_exist_ok: bool = True,
    ) -> None:
        # 同步到 Ultralytics 用户设置文件（可选）；真正决定保存路径的是下面「绝对 project」，
        # 因为 ultralytics 在 import 时会把 RUNS_DIR 缓存在内存里，仅 settings.update 无法更新该缓存。
        settings.update({"runs_dir": str(Path(runs_dir).resolve())})
        self._runs_dir = str(Path(runs_dir).resolve())
        self._model = YOLO(model_path)
        self._predict_task = str(getattr(self._model, "task", None) or "detect")
        self._confidence = float(confidence)
        self._image_size = int(image_size)
        self._device = device
        self._predict_save = predict_save
        self._predict_project = predict_project
        self._predict_name = predict_name
        self._predict_exist_ok = predict_exist_ok

    def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        """对单帧 BGR 图像推理。"""
        # 使用绝对路径 project，绕过 ultralytics.cfg.get_save_dir 对模块级 RUNS_DIR 的依赖
        #（RUNS_DIR 在首次 import ultralytics 时已固定，可能与 settings 不一致）。
        project_dir = str(
            (Path(self._runs_dir) / self._predict_task / self._predict_project).resolve()
        )
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            save=self._predict_save,
            project=project_dir,
            name=self._predict_name,
            exist_ok=self._predict_exist_ok,
            verbose=False,
        )
        if not results:
            return []
        return yolo_result_to_detection_results(results[0])
