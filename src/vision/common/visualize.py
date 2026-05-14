"""检测框、分割 mask、简单文字等可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from src.vision.common.types import DetectionResult


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[DetectionResult],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """在图像上绘制检测框与 risk_class 标签。"""
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.risk_class}:{det.confidence:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def blend_binary_mask(
    image_bgr: np.ndarray,
    mask_hw: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """将单通道二值 mask（>0 为前景）以半透明颜色叠加到 BGR 图像上。"""
    if mask_hw.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("mask 与 image 高宽不一致，请先使用 segmentation.mask_utils.resize_mask_to_image")
    overlay = image_bgr.astype(np.float32)
    m = (mask_hw > 0).astype(np.float32)[..., None]
    c = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    overlay = overlay * (1.0 - alpha * m) + c * (alpha * m)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_bgr(path: str | Path, image_bgr: np.ndarray) -> None:
    """保存 BGR 图像。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image_bgr):
        raise OSError(f"无法写入图像：{out.resolve()}")


def colors_from_config(colors_bgr: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    """将配置中的嵌套列表转为 BGR 元组列表。"""
    return [tuple(int(c) for c in row) for row in colors_bgr]
