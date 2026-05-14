"""OpenVINO 官方 road-segmentation-adas-0001，实现 `BaseSegmenter`。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import openvino as ov

from src.vision.common.interfaces import BaseSegmenter
from src.vision.common.preprocess import image_size_wh
from src.vision.common.types import SegmentationResult
from src.vision.segmentation.postprocess import (
    build_segmentation_result_from_label_map,
    logits_chw_to_label_map,
    run_openvino_adas_forward,
)


class RoadAdasOpenVinoSegmenter(BaseSegmenter):
    """OMZ `road-segmentation-adas-0001`：输出可行驶区域二值 mask（road 类）。"""

    def __init__(
        self,
        xml_path: str | Path,
        device: str,
        input_height: int,
        input_width: int,
        road_class_index: int = 1,
    ) -> None:
        self._xml_path = Path(xml_path)
        self._device = str(device)
        self._input_height = int(input_height)
        self._input_width = int(input_width)
        self._road_class_index = int(road_class_index)

        if not self._xml_path.is_file():
            raise FileNotFoundError(
                "未找到 OpenVINO IR："
                f"{self._xml_path.resolve()}\n"
                "请先执行：omz_downloader --name road-segmentation-adas-0001 "
                "--output_dir models/openvino\n"
                "再将 intel/.../FP32/ 下的 .xml 与 .bin 复制到配置里 xml_path 所在目录。"
            )

        core = ov.Core()
        model = core.read_model(str(self._xml_path))
        self._compiled = core.compile_model(model, self._device)
        self._input_name = self._compiled.input(0).get_any_name()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, project_root: Path) -> RoadAdasOpenVinoSegmenter:
        """从 `segmentation_openvino.yaml` 结构构建。"""
        model_cfg = dict(config["model"])
        ov_cfg = dict(config.get("openvino", {}))
        xml = Path(str(model_cfg["xml_path"]))
        if not xml.is_absolute():
            xml = project_root / xml
        road_idx = int(model_cfg.get("road_class_index", 1))
        return cls(
            xml_path=xml,
            device=str(ov_cfg.get("device", "CPU")),
            input_height=int(model_cfg["input_height"]),
            input_width=int(model_cfg["input_width"]),
            road_class_index=road_idx,
        )

    def infer(self, frame: np.ndarray) -> SegmentationResult:
        logits_chw = run_openvino_adas_forward(
            self._compiled,
            self._input_name,
            frame,
            self._input_height,
            self._input_width,
        )
        label_small = logits_chw_to_label_map(logits_chw)
        wh = image_size_wh(frame)
        return build_segmentation_result_from_label_map(
            label_small,
            wh,
            self._road_class_index,
        )
