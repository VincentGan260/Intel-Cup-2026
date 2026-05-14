"""分割 mask 几何与可行驶区域判断。"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from src.vision.common.types import BBox


def resize_mask_to_image(
    mask: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    将分割 mask resize 到原图大小。

    image_size: (width, height)
    """
    width, height = image_size

    return cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def bbox_bottom_center(bbox: BBox) -> Tuple[int, int]:
    """
    获取检测框底部中心点。
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    return cx, cy


def is_point_in_drivable_area(
    mask: np.ndarray,
    x: int,
    y: int,
) -> bool:
    """
    判断点 (x, y) 是否在可行驶区域内。
    mask 中大于 0 的区域表示可行驶区域。
    """
    h, w = mask.shape[:2]

    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    return bool(mask[y, x] > 0)


def is_bbox_on_drivable_area(
    mask: np.ndarray,
    bbox: BBox,
) -> bool:
    """
    判断检测框是否落在可行驶区域内。
    默认使用 bbox 底部中心点判断。
    """
    cx, cy = bbox_bottom_center(bbox)
    return is_point_in_drivable_area(mask, cx, cy)


def calculate_drivable_ratio(mask: np.ndarray) -> float:
    """
    计算画面中可行驶区域占比。
    """
    if mask.size == 0:
        return 0.0

    return float(np.count_nonzero(mask)) / float(mask.size)
