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
    backend: str | None = None,
) -> BaseSegmenter:
    """根据 YAML 配置构建分割器。

    backend 优先级：显式参数 > 配置里的 backend 字段 > 默认 road_adas_openvino。
      - road_adas_openvino: OpenVINO road-segmentation-adas-0001（4 类，road=1）。
      - pidnet_openvino: PIDNet-S（Cityscapes 19 类，road=0），实时分割。
      - pidnet_pytorch: 占位，后续接入。
    """
    if backend is None:
        backend = str(config.get("backend", "road_adas_openvino"))
    if backend == "road_adas_openvino":
        return RoadAdasOpenVinoSegmenter.from_config(config, project_root=project_root)
    if backend == "pidnet_pytorch":
        from src.vision.segmentation.models.pidnet_pytorch import PidnetPytorchSegmenter

        return PidnetPytorchSegmenter()
    if backend == "pidnet_openvino":
        from src.vision.segmentation.models.pidnet_openvino import PidnetOpenVinoSegmenter

        return PidnetOpenVinoSegmenter.from_config(config, project_root=project_root)
    raise ValueError(f"不支持的 segmenter backend: {backend}")
