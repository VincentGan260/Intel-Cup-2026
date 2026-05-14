"""语义分割统一入口：从配置构建 `BaseSegmenter`。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.vision.common.interfaces import BaseSegmenter
from src.vision.segmentation.models.road_adas_openvino import RoadAdasOpenVinoSegmenter


def build_segmenter_from_config(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    backend: str = "road_adas_openvino",
) -> BaseSegmenter:
    """根据 YAML 配置构建分割器。

    backend:
      - road_adas_openvino: 当前默认，OpenVINO road-segmentation-adas-0001。
      - pidnet_pytorch / pidnet_openvino: 占位，后续接入。
    """
    if backend == "road_adas_openvino":
        return RoadAdasOpenVinoSegmenter.from_config(config, project_root=project_root)
    if backend == "pidnet_pytorch":
        from src.vision.segmentation.models.pidnet_pytorch import PidnetPytorchSegmenter

        return PidnetPytorchSegmenter()
    if backend == "pidnet_openvino":
        from src.vision.segmentation.models.pidnet_openvino import PidnetOpenVinoSegmenter

        return PidnetOpenVinoSegmenter()
    raise ValueError(f"不支持的 segmenter backend: {backend}")
