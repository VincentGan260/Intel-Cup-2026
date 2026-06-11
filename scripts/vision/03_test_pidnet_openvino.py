"""PIDNet-S OpenVINO 分割 demo + 延迟测试（通过 `BaseSegmenter` 接口）。

配置：configs/vision/segmentation_pidnet.yaml

【模型获取（仅需一次，PIDNet 不在 OpenVINO 模型库）】
  1. 下载 MIT 许可的 PIDNet-S ONNX：
       https://huggingface.co/oenpu/PIDNet_S_enlight_friendly_onnx
     把 PIDNet_S_enlight_friendly.onnx 放到 models/openvino/pidnet-s/PIDNet_S.onnx
  2. 转 OpenVINO IR：
       python -c "import openvino as ov; ov.save_model(ov.Core().read_model('models/openvino/pidnet-s/PIDNet_S.onnx'),'models/openvino/pidnet-s/pidnet-s.xml',compress_to_fp16=False)"
  3. 运行（需 intel 环境）：
       conda run -n intel python scripts/vision/03_test_pidnet_openvino.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.common.preprocess import load_image_bgr_from_source
from src.vision.common.visualize import blend_binary_mask, save_bgr
from src.vision.segmentation.segmenter import build_segmenter_from_config

# Cityscapes 19 类调色板（RGB），用于多类彩色叠加。
CITYSCAPES_PALETTE_RGB = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]


def color_overlay(image_bgr: np.ndarray, label_map: np.ndarray, alpha: float) -> np.ndarray:
    """用 Cityscapes 调色板给全部类别上色并叠加。"""
    color = np.zeros_like(image_bgr)
    for idx, rgb in enumerate(CITYSCAPES_PALETTE_RGB):
        color[label_map == idx] = rgb[::-1]  # RGB -> BGR
    return (image_bgr.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    source = config["source"]
    image_bgr, src_desc = load_image_bgr_from_source(str(source), PROJECT_ROOT)
    print(f"输入图像：{src_desc}  尺寸 {image_bgr.shape[1]}x{image_bgr.shape[0]}")

    # 预热
    for _ in range(2):
        segmenter.infer(image_bgr)

    # 计时（取多次平均）
    runs = max(1, int(args.runs))
    t0 = time.perf_counter()
    for _ in range(runs):
        seg = segmenter.infer(image_bgr)
    latency_ms = (time.perf_counter() - t0) / runs * 1000.0

    viz = config["visualization"]
    out_cfg = config["output"]
    alpha = float(viz.get("alpha", 0.45))
    drive_color = tuple(int(c) for c in viz.get("drivable_color_bgr", [0, 200, 0]))

    # 可行驶区域单色叠加
    drivable_overlay = blend_binary_mask(image_bgr, seg.drivable_mask, drive_color, alpha)
    overlay_path = PROJECT_ROOT / str(out_cfg["overlay_path"])
    save_bgr(overlay_path, drivable_overlay)

    # 多类彩色叠加（同目录）
    full_path = overlay_path.parent / (overlay_path.stem + "_full19.jpg")
    save_bgr(full_path, color_overlay(image_bgr, seg.raw_mask, alpha))

    print("-" * 56)
    print(f"可行驶区域占比 drivable_ratio = {seg.drivable_ratio:.4f}")
    print(f"端侧延迟 latency = {latency_ms:.1f} ms  ({1000.0 / latency_ms:.1f} FPS, "
          f"device={config.get('openvino', {}).get('device', 'CPU')})")
    print(f"可行驶叠加图：{overlay_path}")
    print(f"多类彩色图  ：{full_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIDNet-S OpenVINO 分割 demo + 延迟测试。")
    parser.add_argument("--config", default="configs/vision/segmentation_pidnet.yaml",
                        help="配置文件路径（相对项目根目录）。")
    parser.add_argument("--runs", type=int, default=20, help="计时重复次数，取平均。")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
