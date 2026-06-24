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


def refine_drivable_mask(
    mask: np.ndarray,
    *,
    morph_kernel: int = 5,
    do_open: bool = True,
    do_close: bool = True,
    keep_largest: bool = False,
) -> np.ndarray:
    """对 road 二值掩码做轻量形态学清理（开运算去飞点 + 闭运算填空洞）。

    动机：argmax 原始掩码会有孤立误判点（反光/远处小斑块）和内部小空洞。
    下游「雷达路面门控」按某一列方位带的 road 像素占比判定，单片噪声/空洞
    就可能翻转判定 → 误报或漏报。轻量形态学让门控更稳。

    纯 CV、512×896 上约 1~2ms，不增加任何模型推理（端侧零额外推理预算内）。
      - do_open/do_close：开/闭运算开关，默认都开。
      - keep_largest：仅保留最大连通域。默认关——可能误删合法的分叉可行驶区，
        需要时再开（如只关心正前主路面）。
    """
    if mask is None or mask.size == 0:
        return mask
    if morph_kernel < 3 or not (do_open or do_close or keep_largest):
        return mask

    m = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if do_open:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if do_close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    if keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num > 1:  # 0=背景，取前景中面积最大的连通域
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (labels == largest).astype(np.uint8)
    return m.astype(np.uint8)


def build_segmentation_result_from_label_map(
    label_map_model_hw: np.ndarray,
    image_size_wh: tuple[int, int],
    road_class_index: int,
    *,
    refine: bool = True,
    morph_kernel: int = 5,
    keep_largest: bool = False,
) -> SegmentationResult:
    """将模型分辨率下的 label 图放大到原图，并生成 `SegmentationResult`。

    refine=True（默认）对 road 掩码做形态学开+闭清理（见 refine_drivable_mask），
    主要为下游雷达路面门控提供更稳的掩码；传 refine=False 可一键关闭退回原始 argmax。
    """
    label_full = resize_mask_to_image(label_map_model_hw, image_size_wh)
    drivable = label_map_to_drivable_mask(label_full, road_class_index)
    if refine:
        drivable = refine_drivable_mask(
            drivable, morph_kernel=morph_kernel, keep_largest=keep_largest,
        )
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


# ImageNet 归一化常数（PIDNet 等 Cityscapes 预训练模型的标准预处理）。
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def run_openvino_pidnet_forward(
    compiled_model: Any,
    input_name: str,
    image_bgr: np.ndarray,
    input_height: int,
    input_width: int,
) -> np.ndarray:
    """PIDNet 前向：BGR→RGB、缩放、/255、ImageNet 归一化，返回 CHW logits。

    预处理已用真实图像实测确认（见 scripts/vision/03_test_pidnet_openvino 流程）。
    """
    resized = cv2.resize(
        image_bgr,
        (int(input_width), int(input_height)),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = normalized.transpose(2, 0, 1)[None].astype(np.float32)
    outputs = compiled_model({input_name: tensor})
    out_tensor = next(iter(outputs.values()))
    logits = np.array(out_tensor)
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f"期望输出 1xCxHxW，实际 {logits.shape}")
    return logits[0]
