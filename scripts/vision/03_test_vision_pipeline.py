"""视觉最小闭环：检测 + 分割 + 可行驶区域 + 视觉风险初判。

读取 `configs/vision/vision_pipeline.yaml`，内部再加载检测与分割配置。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.common.preprocess import load_image_bgr_from_source
from src.vision.common.visualize import draw_detections, save_bgr
from src.vision.detection.detector import build_detector_from_config
from src.vision.perception.vision_pipeline import VisionPipeline
from src.vision.segmentation.segmenter import build_segmenter_from_config


def main() -> None:
    args = parse_args()
    pipe_cfg = load_config(args.config)

    enable_seg = bool(pipe_cfg.get("enable_segmentation", True))

    det_cfg_path = str(require_value(pipe_cfg, "detection_config"))
    seg_cfg_path = str(require_value(pipe_cfg, "segmentation_config"))

    det_cfg = load_config(det_cfg_path)
    seg_cfg = load_config(seg_cfg_path)

    detector = build_detector_from_config(det_cfg, project_root=PROJECT_ROOT)
    segmenter = (
        build_segmenter_from_config(seg_cfg, project_root=PROJECT_ROOT)
        if enable_seg
        else None
    )

    pipeline = VisionPipeline(
        detector=detector,
        segmenter=segmenter,
        enable_segmentation=enable_seg,
    )

    source = require_value(pipe_cfg, "source")
    image_bgr, source_tag = load_image_bgr_from_source(str(source), PROJECT_ROOT)

    result = pipeline.process(image_bgr)

    print(f"\n来源: {source_tag}")
    print(f"max_visual_risk = {result.max_visual_risk:.4f}")
    for det in result.detections:
        print(
            f"- {det.class_name} | {det.risk_class} | conf={det.confidence:.2f} "
            f"| in_drivable={det.in_drivable_area} | visual_risk={det.visual_risk:.4f}"
        )

    out_cfg = pipe_cfg.get("output") or {}
    if bool(out_cfg.get("save_debug_image", False)):
        dbg = draw_detections(image_bgr, result.detections)
        path = PROJECT_ROOT / str(out_cfg["debug_image_path"])
        save_bgr(path, dbg)
        print(f"调试图已保存：{path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionPipeline 最小闭环。")
    p.add_argument(
        "--config",
        default="configs/vision/vision_pipeline.yaml",
        help="管线配置文件（相对项目根目录）。",
    )
    return p.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 YAML 映射：{path}")
    return data


def require_value(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"缺少配置项：{key}")
    return data[key]


if __name__ == "__main__":
    main()
