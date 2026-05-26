"""单张图片 YOLO 检测脚本（通过 BaseDetector 接口）"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[0]   # 假设 main.py 在项目根目录
# 如果 main.py 在 your_project/ 下，则 project root 就是当前目录
# 原代码用 parents[2] 是因为它在 src 子目录下，这里调整为根目录
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.common.preprocess import load_image_bgr_from_source
from src.vision.detection.detector import build_detector_from_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    detector = build_detector_from_config(config, PROJECT_ROOT)

    source = require_value(config, "source")
    image_bgr, source_tag = load_image_bgr_from_source(str(source), PROJECT_ROOT)

    detections = detector.infer(image_bgr)

    print(f"\n来源: {source_tag}")
    if not detections:
        print("未检测到项目关注类别。")
        return

    for det in detections:
        bbox = ", ".join(f"{v:.1f}" for v in det.bbox)
        print(
            f"- {det.class_name} -> {det.risk_class} "
            f"conf={det.confidence:.2f} bbox=[{bbox}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 单图检测（OpenVINO）")
    parser.add_argument(
        "--config",
        default="configs/vision/detection.yaml",
        help="检测配置文件路径（相对项目根目录）。",
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


def require_value(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"缺少配置项：{key}")
    return data[key]


if __name__ == "__main__":
    main()