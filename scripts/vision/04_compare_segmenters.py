"""分割模型对比：road-segmentation-adas-0001 vs PIDNet-S（同图、同口径）。

在同一批图上分别跑两个模型，输出：
  - 并排叠加图（原图 | road-adas 可行驶 | PIDNet-S 可行驶），存到 runs/segmentation_compare/
  - 终端表格：每个模型的 可行驶占比 / 平均延迟 / FPS

没有真值标注时算不了 IoU，这里用「可行驶占比 + 目检 + 延迟」做对比。
有标注后可在此基础上加 road IoU。

运行（需 intel 环境，且 PIDNet IR 已就绪，见 03 脚本说明）：
    conda run -n intel python scripts/vision/04_compare_segmenters.py
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

from src.vision.common.preprocess import read_image_bgr
from src.vision.common.visualize import blend_binary_mask, save_bgr
from src.vision.segmentation.segmenter import build_segmenter_from_config

# 要对比的模型：(显示名, 配置文件)
MODELS = [
    ("road-adas", "configs/vision/segmentation_openvino.yaml"),
    ("PIDNet-S", "configs/vision/segmentation_pidnet.yaml"),
]
# 默认测试图（可用 --image 覆盖，可多张）
DEFAULT_IMAGES = [
    "data/sample/segmentation_test_data/frankfurt_street.png",
    "data/sample/bus.jpg",
]
OUTPUT_DIR = PROJECT_ROOT / "runs" / "segmentation_compare"


def load_yaml(rel: str) -> dict[str, Any]:
    with (PROJECT_ROOT / rel).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def label_strip(img: np.ndarray, text: str) -> np.ndarray:
    """在图顶部画一条黑底标签。"""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def main() -> None:
    args = parse_args()
    images = args.image or DEFAULT_IMAGES
    runs = max(1, int(args.runs))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 构建各模型一次（复用）
    segmenters = []
    for name, cfg_rel in MODELS:
        cfg = load_yaml(cfg_rel)
        seg = build_segmenter_from_config(cfg, project_root=PROJECT_ROOT)
        device = cfg.get("openvino", {}).get("device", "CPU")
        segmenters.append((name, seg, device))

    rows: list[tuple] = []
    for img_rel in images:
        img_path = PROJECT_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
        if not img_path.exists():
            print(f"跳过（找不到）：{img_path}")
            continue
        image = read_image_bgr(img_path)
        panels = [label_strip(image, f"original {image.shape[1]}x{image.shape[0]}")]

        for name, seg, device in segmenters:
            for _ in range(2):  # 预热
                seg.infer(image)
            t0 = time.perf_counter()
            for _ in range(runs):
                result = seg.infer(image)
            latency = (time.perf_counter() - t0) / runs * 1000.0
            fps = 1000.0 / latency

            overlay = blend_binary_mask(image, result.drivable_mask, (0, 200, 0), 0.45)
            panels.append(label_strip(
                overlay, f"{name}  drv={result.drivable_ratio*100:.0f}%  {latency:.0f}ms/{fps:.1f}fps[{device}]"))
            rows.append((img_path.name, name, device, result.drivable_ratio, latency, fps))

        # 统一高度后并排
        h = min(p.shape[0] for p in panels)
        panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
        combo = np.hstack(panels)
        out_path = OUTPUT_DIR / f"compare_{img_path.stem}.jpg"
        save_bgr(out_path, combo)
        print(f"并排图：{out_path}")

    # 汇总表
    print("\n" + "=" * 74)
    print(f"{'image':<26}{'model':<12}{'device':<7}{'drivable%':>10}{'ms':>8}{'fps':>8}")
    print("-" * 74)
    for img_name, name, device, ratio, latency, fps in rows:
        print(f"{img_name[:25]:<26}{name:<12}{device:<7}{ratio*100:>9.1f}{latency:>8.0f}{fps:>8.1f}")
    print("=" * 74)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="road-adas vs PIDNet-S 分割对比。")
    parser.add_argument("--image", action="append", default=None,
                        help="测试图路径（可多次指定）；默认用 data/sample 下的样例。")
    parser.add_argument("--runs", type=int, default=20, help="每个模型计时重复次数，取平均。")
    return parser.parse_args()


if __name__ == "__main__":
    main()
