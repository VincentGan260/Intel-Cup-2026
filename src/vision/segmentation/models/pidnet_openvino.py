"""PIDNet-S OpenVINO 推理，实现 `BaseSegmenter`。

PIDNet-S 在 Cityscapes（19 类）上预训练，实时分割模型。输出为 1/8 分辨率的
19 通道 logits，argmax 取类别后放大到原图。Cityscapes 中 road=0 即可行驶区域。

预处理：BGR→RGB、缩放到固定输入、/255、ImageNet 均值方差归一化
（已用真实街景图实测确认，见 scripts/vision/03_test_pidnet_openvino.py）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import openvino as ov

from src.vision.common.interfaces import BaseSegmenter
from src.vision.common.preprocess import image_size_wh
from src.vision.common.types import SegmentationResult
from src.vision.segmentation.postprocess import (
    build_segmentation_result_from_label_map,
    logits_chw_to_label_map,
    run_openvino_pidnet_forward,
)


class PidnetOpenVinoSegmenter(BaseSegmenter):
    """PIDNet-S（Cityscapes 19 类）OpenVINO 后端，输出可行驶区域 mask。"""

    def __init__(
        self,
        xml_path: str | Path,
        device: str,
        input_height: int,
        input_width: int,
        road_class_index: int = 0,
    ) -> None:
        self._xml_path = Path(xml_path)
        self._device = str(device)
        self._input_height = int(input_height)
        self._input_width = int(input_width)
        self._road_class_index = int(road_class_index)

        if not self._xml_path.is_file():
            raise FileNotFoundError(
                "未找到 PIDNet OpenVINO IR："
                f"{self._xml_path.resolve()}\n"
                "请先下载 ONNX 并转换为 IR，见 scripts/vision/03_test_pidnet_openvino.py 顶部说明。"
            )

        core = ov.Core()
        model = core.read_model(str(self._xml_path))
        self._compiled = core.compile_model(model, self._device)
        self._input_name = self._compiled.input(0).get_any_name()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, project_root: Path) -> "PidnetOpenVinoSegmenter":
        model_cfg = dict(config["model"])
        ov_cfg = dict(config.get("openvino", {}))
        xml = Path(str(model_cfg["xml_path"]))
        if not xml.is_absolute():
            xml = project_root / xml
        return cls(
            xml_path=xml,
            device=str(ov_cfg.get("device", "CPU")),
            input_height=int(model_cfg["input_height"]),
            input_width=int(model_cfg["input_width"]),
            road_class_index=int(model_cfg.get("road_class_index", 0)),
        )

    def infer(self, frame: np.ndarray) -> SegmentationResult:
        logits_chw = run_openvino_pidnet_forward(
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
