"""语义分割后处理：模型输出 -> `SegmentationResult`（可行驶二值 mask）。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.vision.common.types import SegmentationResult
from src.vision.segmentation.mask_utils import calculate_drivable_ratio, resize_mask_to_image


def logits_chw_to_label_map(logits_chw: np.ndarray) -> np.ndarray:
    """CHW logits -> HW uint8 类别索引。"""
    return np.argmax(logits_chw, axis=0).astype(np.uint8)


def label_map_to_drivable_mask(
    label_map_hw: np.ndarray,
    road_class_index: int,
) -> np.ndarray:
    """将多类 label 转为可行驶区域二值图（road 类为 1，其余为 0）。"""
    return (label_map_hw == int(road_class_index)).astype(np.uint8)


def build_segmentation_result_from_label_map(
    label_map_model_hw: np.ndarray,
    image_size_wh: tuple[int, int],
    road_class_index: int,
) -> SegmentationResult:
    """将模型分辨率下的 label 图放大到原图，并生成 `SegmentationResult`。"""
    label_full = resize_mask_to_image(label_map_model_hw, image_size_wh)
    drivable = label_map_to_drivable_mask(label_full, road_class_index)
    ratio = calculate_drivable_ratio(drivable)
    return SegmentationResult(
        drivable_mask=drivable,
        raw_mask=label_full,
        drivable_ratio=ratio,
    )


def run_openvino_adas_forward(
    compiled_model: Any,
    input_name: str,
    image_bgr: np.ndarray,
    input_height: int,
    input_width: int,
) -> np.ndarray:
    """执行一次 OpenVINO 前向，返回 CHW logits（float）。"""
    resized = cv2.resize(
        image_bgr,
        (int(input_width), int(input_height)),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = resized.transpose(2, 0, 1)[None].astype(np.float32)
    outputs = compiled_model({input_name: tensor})
    out_tensor = next(iter(outputs.values()))
    logits = np.array(out_tensor)
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f"期望输出 1xCxHxW，实际 {logits.shape}")
    return logits[0]
