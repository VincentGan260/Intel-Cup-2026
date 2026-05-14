"""目标检测统一入口：从配置构建 `BaseDetector` 实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.vision.common.interfaces import BaseDetector
from src.vision.detection.models.yolo_ultralytics import YoloUltralyticsDetector


def build_detector_from_config(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    backend: str = "yolo_ultralytics",
) -> BaseDetector:
    """根据 YAML 配置字典构建检测器。

    backend:
      - yolo_ultralytics: 当前默认，使用 .pt 与 Ultralytics。
      - yolo_openvino: 后续实现，对应 `YoloOpenVinoDetector`。
    """
    detector_cfg = dict(config["detector"])
    output_cfg = dict(config.get("output", {}))

    mp = Path(str(detector_cfg["model_path"]))
    model_path = str(mp) if mp.is_absolute() else str(project_root / mp)

    runs_dir_raw = output_cfg.get("runs_dir")
    if runs_dir_raw in (None, ""):
        # 与分割等视觉输出统一在 runs/vision/ 下（见各 vision 配置）。
        runs_dir_resolved = str((project_root / "runs" / "vision").resolve())
    else:
        rp = Path(str(runs_dir_raw))
        runs_dir_resolved = str(
            (rp if rp.is_absolute() else (project_root / rp)).resolve()
        )

    if backend == "yolo_ultralytics":
        return YoloUltralyticsDetector(
            model_path=model_path,
            confidence=float(detector_cfg["confidence"]),
            image_size=int(detector_cfg["image_size"]),
            device=detector_cfg.get("device"),
            runs_dir=runs_dir_resolved,
            predict_save=bool(output_cfg.get("save", False)),
            predict_project=str(output_cfg.get("project", "yolo26n_demo")),
            predict_name=str(output_cfg.get("name", "pred")),
            predict_exist_ok=bool(output_cfg.get("exist_ok", True)),
        )

    if backend == "yolo_openvino":
        from src.vision.detection.models.yolo_openvino import YoloOpenVinoDetector

        return YoloOpenVinoDetector()

    raise ValueError(f"不支持的 detector backend: {backend}")
