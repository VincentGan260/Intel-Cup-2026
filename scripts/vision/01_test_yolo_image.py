"""单张图片 YOLO26n/YOLOv8n 检测脚本。

这个脚本只负责“读取配置 -> 创建检测器 -> 跑一张图片 -> 打印结果”。
如果要修改模型、图片、置信度、输出目录等参数，请改 `configs/vision/detection.yaml`。
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.detector.yolo_detector import YoloDetector


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    config = load_config(args.config)

    # 三个配置块分别控制模型推理、输出目录和后处理规则。
    detector_config = require_mapping(config, "detector")
    output_config = require_mapping(config, "output")
    postprocess_config = require_mapping(config, "postprocess")

    # YoloDetector 不写业务默认值，所有可调参数都从配置文件传入。
    detector = YoloDetector(
        model_path=str(require_value(detector_config, "model_path")),
        confidence=float(require_value(detector_config, "confidence")),
        image_size=int(require_value(detector_config, "image_size")),
        device=detector_config.get("device"),
        runs_dir=output_config.get("runs_dir"),
        project_to_model_classes=require_mapping(postprocess_config, "target_classes"),
        keep_target_classes_only=bool(
            require_value(postprocess_config, "keep_target_classes_only")
        ),
        other_project_class=str(require_value(postprocess_config, "other_project_class")),
    )

    # source 可以是图片路径、图片 URL，也可以扩展为摄像头编号。
    results = detector.predict(
        source=parse_source(require_value(config, "source")),
        save=bool(require_value(output_config, "save")),
        project=str(require_value(output_config, "project")),
        name=str(require_value(output_config, "name")),
        exist_ok=bool(require_value(output_config, "exist_ok")),
    )

    # 先用终端打印检测结果，方便第一阶段快速确认模型是否跑通。
    for image_result in results:
        print(f"\nSource: {image_result.source}")
        if not image_result.detections:
            print("No target objects detected.")
            continue

        for detection in image_result.detections:
            bbox = ", ".join(f"{value:.1f}" for value in detection.bbox_xyxy)
            print(
                f"- {detection.class_name} -> {detection.project_class} "
                f"conf={detection.confidence:.2f} bbox=[{bbox}]"
            )


def parse_args() -> argparse.Namespace:
    """读取命令行参数。

    这里只保留 `--config`，其它运行参数统一放进配置文件，避免代码和命令行混乱。
    """

    parser = argparse.ArgumentParser(description="Run YOLO image detection.")
    parser.add_argument(
        "--config",
        default="configs/vision/detection.yaml",
        help="Path to detection config file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping: {path}")

    return data


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必须存在的字典配置块。"""

    value = require_value(data, key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field must be a mapping: {key}")

    return value


def require_value(data: dict[str, Any], key: str) -> Any:
    """读取必须存在的配置字段，缺失时直接报错。"""

    if key not in data:
        raise KeyError(f"Missing required config field: {key}")

    return data[key]


def parse_source(source: Any) -> str | int:
    """把配置中的输入源转换为 Ultralytics 可接受的格式。"""

    source_text = str(source).strip()
    if source_text.isdigit():
        return int(source_text)

    return source_text


if __name__ == "__main__":
    main()
