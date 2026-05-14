"""通用图像预处理（与具体模型解耦的轻量工具）。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def ensure_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    """确保为 HxWx3 的 uint8 BGR。"""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame 必须是 HxWx3 的 BGR 图像")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def read_image_bgr(path: str | Path) -> np.ndarray:
    """从本地路径读取 BGR 图像。"""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{Path(path).resolve()}")
    return ensure_bgr_uint8(image)


def load_image_bgr_from_source(source: str, project_root: Path) -> tuple[np.ndarray, str]:
    """从 URL 或相对/绝对路径加载 BGR 图像，返回 (图像, 来源描述)。"""
    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request

        raw = urllib.request.urlopen(source).read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法从 URL 解码图像：{source}")
        return ensure_bgr_uint8(image), source

    path = Path(source)
    if not path.is_absolute():
        path = project_root / path
    return read_image_bgr(path), str(path)


def image_size_wh(image: np.ndarray) -> Tuple[int, int]:
    """返回 (width, height)。"""
    h, w = image.shape[:2]
    return w, h
