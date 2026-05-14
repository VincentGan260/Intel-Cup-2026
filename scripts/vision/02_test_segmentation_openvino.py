"""OpenVINO 道路分割 demo（通过 `BaseSegmenter` 接口）。

配置：`configs/vision/segmentation_openvino.yaml`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.common.preprocess import load_image_bgr_from_source
from src.vision.common.visualize import blend_binary_mask, colors_from_config, save_bgr
from src.vision.segmentation.segmenter import build_segmenter_from_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)

    source = require_value(config, "source")
    image_bgr, _ = load_image_bgr_from_source(str(source), PROJECT_ROOT)

    seg = segmenter.infer(image_bgr)

    viz_cfg = require_mapping(config, "visualization")
    out_cfg = require_mapping(config, "output")

    # 用多类上色时，用 raw_mask；仅看可行驶区域时用 drivable_mask 单色叠加
    raw = seg.raw_mask
    if raw is None:
        overlay = blend_binary_mask(
            image_bgr,
            seg.drivable_mask,
            color_bgr=(0, 200, 0),
            alpha=float(require_value(viz_cfg, "alpha")),
        )
    else:
        colors = colors_from_config(require_value(viz_cfg, "colors_bgr"))
        color_img = np.zeros_like(image_bgr)
        for idx, color in enumerate(colors):
            m = raw == idx
            color_img[m] = color
        alpha = float(require_value(viz_cfg, "alpha"))
        overlay = (image_bgr.astype(float) * (1 - alpha) + color_img.astype(float) * alpha).astype(
            np.uint8
        )

    overlay_path = PROJECT_ROOT / str(require_value(out_cfg, "overlay_path"))
    save_bgr(overlay_path, overlay)

    print(f"分割完成，叠加图已保存：{overlay_path}")
    if seg.drivable_ratio is not None:
        print(f"可行驶区域占比 drivable_ratio = {seg.drivable_ratio:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenVINO 道路分割 demo。")
    parser.add_argument(
        "--config",
        default="configs/vision/segmentation_openvino.yaml",
        help="分割配置文件路径（相对项目根目录）。",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML 映射：{path}")
    return data


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    v = require_value(data, key)
    if not isinstance(v, dict):
        raise ValueError(f"配置项 {key} 必须是字典")
    return v


def require_value(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"缺少配置项：{key}")
    return data[key]


if __name__ == "__main__":
    main()
