"""视觉风险初判（仅视觉因素，不含雷达 / IMU / GPS / TTC）。"""

from __future__ import annotations

from typing import List

from src.vision.common.types import DetectionResult
from src.vision.detection.class_mapping import CLASS_BASE_RISK


def calculate_center_risk(
    bbox,
    image_width: int,
) -> float:
    """
    目标越靠近画面中心，风险越高。
    """
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2

    image_center = image_width / 2
    distance = abs(cx - image_center)

    max_distance = image_width / 2
    risk = 1.0 - min(distance / max_distance, 1.0)

    return float(risk)


def calculate_size_risk(
    bbox,
    image_width: int,
    image_height: int,
) -> float:
    """
    bbox 面积越大，粗略认为目标越近，风险越高。
    注意：这只是视觉估计，最终距离风险应由雷达/深度相机提供。
    """
    x1, y1, x2, y2 = bbox

    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = image_width * image_height

    if image_area <= 0:
        return 0.0

    ratio = bbox_area / image_area

    # 简单归一化，避免大框直接超过 1
    return float(min(ratio / 0.25, 1.0))


def calculate_area_risk(in_drivable_area: bool | None) -> float:
    """
    目标在可行驶区域内，风险更高。
    """
    if in_drivable_area is True:
        return 1.0
    if in_drivable_area is False:
        return 0.35
    return 0.5


def calculate_detection_visual_risk(
    det: DetectionResult,
    image_width: int,
    image_height: int,
) -> float:
    """
    计算单个目标的视觉风险。
    这里只做视觉初判，不考虑雷达距离、TTC、IMU、GPS。
    """
    class_risk = CLASS_BASE_RISK.get(det.risk_class, 0.3)
    center_risk = calculate_center_risk(det.bbox, image_width)
    size_risk = calculate_size_risk(det.bbox, image_width, image_height)
    area_risk = calculate_area_risk(det.in_drivable_area)

    visual_risk = (
        0.35 * class_risk
        + 0.25 * center_risk
        + 0.20 * size_risk
        + 0.20 * area_risk
    )

    return float(max(0.0, min(visual_risk, 1.0)))


def attach_visual_risk(
    detections: List[DetectionResult],
    image_width: int,
    image_height: int,
) -> List[DetectionResult]:
    """
    给每个检测目标添加 visual_risk 字段。
    """
    for det in detections:
        det.visual_risk = calculate_detection_visual_risk(
            det,
            image_width,
            image_height,
        )

    return detections
