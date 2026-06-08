"""
语义分割可视化脚本（road-segmentation-adas-0001）

修复要点（相对旧版）：
  1. 旧版用 output.squeeze()[0] 取的是「背景通道的 logit」，不是类别图，导致天空/建筑
     被误标为可行驶区域。新版复用 src.vision.segmentation 的正规管线：对 4 个通道
     做 argmax 得到每像素类别，BGR 直接输入（该模型要求 BGR，旧版错误地转了 RGB），
     label 图用最近邻 resize。
  2. 类别名与模型输出顺序对齐：0=背景 1=road 2=curb(路缘) 3=mark(标线)。
  3. 报告的是各类「面积占比」，不是准确率。要算准确率（road IoU）必须有标注 mask，
     见 compute_road_iou / --gt-dir 说明。

用法：
    conda run -n intel python scripts/visualize_segmentation.py
    conda run -n intel python scripts/visualize_segmentation.py --num 30 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_paths import load_path_config, resolve
from src.vision.segmentation.segmenter import build_segmenter_from_config

# 分割模型的 OpenVINO 配置（构建 segmenter 用，与路径配置分开）。
SEG_CONFIG_PATH = PROJECT_ROOT / "configs" / "vision" / "segmentation_openvino.yaml"

# 类别 → (中文名, BGR 颜色)。索引与模型输出通道一一对应。
CLASSES = {
    0: ("背景", (0, 0, 0)),
    1: ("road", (0, 255, 0)),       # 可行驶路面 - 绿
    2: ("curb", (0, 128, 255)),     # 路缘 - 橙
    3: ("mark", (255, 255, 255)),   # 标线 - 白
}
ROAD_CLASS = 1


def load_seg_config() -> dict:
    with SEG_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def colorize(label_map: np.ndarray) -> np.ndarray:
    """每像素类别 → BGR 彩色图。"""
    colored = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for idx, (_, color) in CLASSES.items():
        colored[label_map == idx] = color
    return colored


def visualize(original_bgr: np.ndarray, label_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把类别图半透明叠加到原图，并对可行驶区域勾黄色边界。"""
    colored = colorize(label_map)
    fg = label_map > 0  # 非背景才叠色，背景保留原图
    overlay = original_bgr.copy()
    overlay[fg] = cv2.addWeighted(original_bgr, 1 - alpha, colored, alpha, 0)[fg]

    road_mask = (label_map == ROAD_CLASS).astype(np.uint8) * 255
    edges = cv2.Canny(road_mask, 50, 150)
    overlay[edges > 0] = (0, 255, 255)
    return overlay


def class_ratios(label_map: np.ndarray) -> dict:
    """各类面积占比（注意：占比 ≠ 准确率）。"""
    total = label_map.size
    return {name: float(np.sum(label_map == idx)) / total for idx, (name, _) in CLASSES.items()}


def compute_road_iou(pred_label: np.ndarray, gt_road_mask: np.ndarray) -> float:
    """road 类的 IoU = |预测∩真值| / |预测∪真值|。

    pred_label: 模型预测的每像素类别图（与 gt 同高宽）
    gt_road_mask: 真值里 road 为 True/非0 的二值 mask
    """
    pred = pred_label == ROAD_CLASS
    gt = gt_road_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def find_gt_road_mask(img_path: Path, gt_dir: Optional[Path]) -> Optional[np.ndarray]:
    """尝试加载与图片对应的 road 真值 mask（有标注才能算 IoU）。

    约定：在 gt_dir 下找同名 *_road.png（值>0 表示 road）。各数据集标注格式不同，
    需按实际情况改这里的匹配规则；找不到则返回 None，跳过 IoU。
    """
    if gt_dir is None:
        return None
    candidate = gt_dir / f"{img_path.stem}_road.png"
    if not candidate.exists():
        return None
    m = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    return m if m is not None else None


def gather_images(image_dirs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for d in image_dirs:
        if d.exists():
            images += list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="语义分割可视化（road-segmentation-adas-0001）。"
                    "路径默认读 configs/vision/datasets.yaml（可被 datasets.local.yaml 覆盖），"
                    "下面的参数可临时覆盖。"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="额外的路径配置文件，覆盖 datasets.yaml / datasets.local.yaml")
    parser.add_argument("--image-dir", action="append", default=None,
                        help="图片目录（可多次指定）；给了就替换配置里的 image_dirs")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录，覆盖 vis_output_dir")
    parser.add_argument("--num", type=int, default=None, help="可视化样本数量，覆盖 num_samples")
    parser.add_argument("--seed", type=int, default=None, help="随机采样种子，覆盖 seed")
    parser.add_argument("--alpha", type=float, default=0.45, help="叠加透明度 0~1")
    parser.add_argument("--gt-dir", type=str, default=None,
                        help="road 真值 mask 目录，覆盖 gt_dir；提供后会计算 road IoU")
    args = parser.parse_args()

    # 路径来源优先级：命令行 > datasets.local.yaml > datasets.yaml
    paths = load_path_config(args.config)
    image_dirs = [resolve(p) for p in (args.image_dir or paths.get("image_dirs", []))]
    out_dir = resolve(args.out_dir or paths.get("vis_output_dir", "runs/segmentation_openvino/demo"))
    num_samples = args.num if args.num is not None else int(paths.get("num_samples", 20))
    seed = args.seed if args.seed is not None else int(paths.get("seed", 0))
    gt_cfg = args.gt_dir if args.gt_dir is not None else paths.get("gt_dir")
    gt_dir = resolve(gt_cfg) if gt_cfg else None

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("语义分割可视化")
    print("=" * 60)

    config = load_seg_config()
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    print(f"模型已加载：{config['model']['xml_path']}")

    all_images = gather_images(image_dirs)
    if not all_images:
        print("错误：未在以下目录找到测试图片（改 configs/vision/datasets.yaml 或 --image-dir）：")
        for d in image_dirs:
            print(f"  - {d}")
        return
    print(f"共找到 {len(all_images)} 张图片")

    rng = random.Random(seed)
    num = min(num_samples, len(all_images))
    selected = rng.sample(all_images, num)
    print(f"采样 {num} 张生成可视化...\n")

    summary: list[dict] = []
    ious: list[float] = []

    for idx, img_path in enumerate(selected, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[{idx}/{num}] 跳过（无法读取）：{img_path.name}")
            continue

        seg = segmenter.infer(img_bgr)
        label_map = seg.raw_mask  # 全分辨率类别图（已 argmax + 最近邻 resize）

        vis = visualize(img_bgr, label_map, alpha=args.alpha)
        ratios = class_ratios(label_map)

        text_lines = [
            f"Image: {img_path.name}",
            f"road: {ratios['road'] * 100:.1f}%  curb: {ratios['curb'] * 100:.1f}%  mark: {ratios['mark'] * 100:.1f}%",
            f"Size: {img_bgr.shape[1]}x{img_bgr.shape[0]}",
        ]

        # 有真值则算 road IoU
        gt = find_gt_road_mask(img_path, gt_dir)
        iou_val: Optional[float] = None
        if gt is not None:
            if gt.shape[:2] != label_map.shape[:2]:
                gt = cv2.resize(gt, (label_map.shape[1], label_map.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            iou_val = compute_road_iou(label_map, gt)
            if not np.isnan(iou_val):
                ious.append(iou_val)
            text_lines.append(f"road IoU: {iou_val:.3f}")

        for i, line in enumerate(text_lines):
            pos = (10, 30 + i * 28)
            cv2.putText(vis, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(vis, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        out_path = out_dir / f"seg_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)

        msg = f"[{idx}/{num}] road={ratios['road'] * 100:.1f}%"
        if iou_val is not None:
            msg += f"  IoU={iou_val:.3f}"
        print(f"{msg}  -> {out_path.name}")

        summary.append({
            "image": img_path.name,
            "road_ratio": ratios["road"],
            "curb_ratio": ratios["curb"],
            "mark_ratio": ratios["mark"],
            "road_iou": "" if iou_val is None else round(iou_val, 4),
            "output": out_path.name,
        })

    # 前 5 张额外存「原图 | 叠加图」对比
    print("\n生成对比图...")
    for idx, img_path in enumerate(selected[:5], 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        seg = segmenter.infer(img_bgr)
        vis = visualize(img_bgr, seg.raw_mask, alpha=args.alpha)
        comparison = np.hstack([img_bgr, vis])
        cmp_path = out_dir / f"comparison_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(cmp_path), comparison)
        print(f"  {cmp_path.name}")

    # 汇总 CSV
    summary_path = out_dir / "segmentation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "road_ratio", "curb_ratio", "mark_ratio", "road_iou", "output"]
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)
    print(f"输出目录：{out_dir}")
    if summary:
        avg_road = sum(s["road_ratio"] for s in summary) / len(summary)
        print(f"平均 road 面积占比：{avg_road * 100:.2f}%（占比，非准确率）")
    if ious:
        print(f"平均 road IoU：{sum(ious) / len(ious):.3f}（基于 {len(ious)} 张有标注的图）")
    else:
        print("未提供 --gt-dir 标注，未计算 road IoU。")


if __name__ == "__main__":
    main()
