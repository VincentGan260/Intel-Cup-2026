"""分割精度评测（manifest 驱动，road IoU）。

按 测试说明.md：读 test_plan/seg_manifest.csv，逐图算 road IoU，
**按数据集分别报 + ACDC 按天气(fog/night/rain/snow)报**，macro 均值仅作参考、不池化。
对比 road-adas vs PIDNet-S（OpenVINO FP32）。

GT 统一用数据集自带的 tools/unify_seg_labels.to_road_mask(dataset, label) → {road=1,bg=0,ignore=255}。

运行（数据根在 eval.local.yaml 的 dataset_root）：
    D:/Anaconda_envs/envs/intel/python.exe scripts/vision/eval_seg.py
    ... scripts/vision/eval_seg.py --limit 20      # 快速冒烟
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.dataset_paths import load_eval_config, resolve
from src.vision.segmentation.postprocess import (
    run_openvino_adas_forward,
    run_openvino_pidnet_forward,
    logits_chw_to_label_map,
)

# 模型定义：(显示名, 配置键, 输入HW, road类别索引, 前向函数)
MODEL_DEFS = [
    ("road-adas", "road_adas", (512, 896), 1, run_openvino_adas_forward),
    ("PIDNet-S", "pidnet_fp32", (1024, 1024), 0, run_openvino_pidnet_forward),
]


def road_iou(pred_road: np.ndarray, gt: np.ndarray) -> float:
    """pred_road 为布尔；gt 为 {0,1,255}。在 valid(gt!=255) 上算 road IoU。"""
    valid = gt != 255
    pred = pred_road & valid
    g = (gt == 1) & valid
    union = int(np.sum(pred | g))
    if union == 0:
        return 1.0 if int(np.sum(g)) == 0 else 0.0
    return float(np.sum(pred & g)) / union


def main() -> None:
    ap = argparse.ArgumentParser(description="分割 road IoU 评测（manifest 驱动）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时每个数据集最多评这么多张（冒烟用）")
    ap.add_argument("--models", nargs="*", default=None, help="只评指定模型（road-adas / PIDNet-S）")
    args = ap.parse_args()

    cfg = load_eval_config()
    ds_root = cfg.get("dataset_root")
    if not ds_root:
        print("错误：eval.local.yaml 里缺 dataset_root"); return
    ds_root = Path(ds_root)
    manifest = ds_root / "test_plan" / "seg_manifest.csv"
    if not manifest.is_file():
        print(f"错误：找不到 {manifest}"); return

    # 引入数据集自带的统一标签工具
    sys.path.insert(0, str(ds_root / "tools"))
    from unify_seg_labels import to_road_mask

    # 读 manifest
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if args.limit > 0:
        by_ds = defaultdict(list)
        for r in rows:
            if len(by_ds[r["dataset"]]) < args.limit:
                by_ds[r["dataset"]].append(r)
        rows = [r for rs in by_ds.values() for r in rs]
    print(f"manifest: {len(rows)} 行  数据根: {ds_root}")

    seg_models = cfg.get("models", {}).get("segmentation", {})
    core = ov.Core()
    models = []
    for name, key, hw, road_idx, fwd in MODEL_DEFS:
        if args.models and name not in args.models:
            continue
        xml = resolve(seg_models[key])
        if not xml.is_file():
            print(f"[跳过] {name} 模型不存在: {xml}"); continue
        compiled = core.compile_model(str(xml), "CPU")
        models.append((name, compiled, compiled.input(0).get_any_name(), hw, road_idx, fwd))

    # 结果累积：iou_by[model][dataset] = [iou...]；acdc 另按天气
    iou_by = {m[0]: defaultdict(list) for m in models}
    acdc_by_weather = {m[0]: defaultdict(list) for m in models}

    for i, r in enumerate(rows, 1):
        ds = r["dataset"]
        img = cv2.imread(str(ds_root / r["image"]))
        if img is None:
            continue
        gt = to_road_mask(ds, str(ds_root / r["label"]))
        gh, gw = gt.shape[:2]
        for name, compiled, iname, (ih, iw), road_idx, fwd in models:
            logits = fwd(compiled, iname, img, ih, iw)
            label = logits_chw_to_label_map(logits)
            pred = cv2.resize(label.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST)
            iou = road_iou(pred == road_idx, gt)
            iou_by[name][ds].append(iou)
            if ds == "acdc":
                acdc_by_weather[name][r.get("weather", "unknown")].append(iou)
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}")

    # ---- 报表 ----
    def mean_pct(xs): return round(statistics.mean(xs) * 100, 2) if xs else 0.0

    result = {"dataset_root": str(ds_root), "n_rows": len(rows), "models": {}}
    print("\n" + "=" * 70)
    print("road IoU（按数据集，单位 %）")
    print("=" * 70)
    datasets = ["cityscapes", "idd", "acdc", "bdd", "camvid"]
    header = f"{'模型':<12}" + "".join(f"{d:>11}" for d in datasets) + f"{'macro均值':>11}"
    print(header)
    print("-" * len(header))
    for name in iou_by:
        per = {d: mean_pct(iou_by[name][d]) for d in datasets}
        macro = round(statistics.mean([v for v in per.values()]), 2)
        print(f"{name:<12}" + "".join(f"{per[d]:>11}" for d in datasets) + f"{macro:>11}")
        result["models"][name] = {"per_dataset": per, "macro_mean": macro,
                                  "counts": {d: len(iou_by[name][d]) for d in datasets}}

    print("\nACDC 按天气 road IoU（%）")
    print("-" * 50)
    # 动态取实际出现的天气桶（不硬编码，避免 manifest 用别的天气字串时漏显示）
    weathers = sorted({w for m in acdc_by_weather for w in acdc_by_weather[m]})
    print(f"{'模型':<12}" + "".join(f"{w + f'({len(acdc_by_weather[list(acdc_by_weather)[0]][w])})':>12}" for w in weathers))
    for name in acdc_by_weather:
        per_w = {w: mean_pct(acdc_by_weather[name][w]) for w in weathers}
        print(f"{name:<12}" + "".join(f"{per_w[w]:>12}" for w in weathers))
        result["models"][name]["acdc_by_weather"] = per_w

    print("\n> macro 均值仅参考（各集难度差异大、不应池化）；主看每个数据集与 ACDC 各天气分项。")

    out_dir = resolve(cfg.get("output", {}).get("accuracy_dir", "runs/accuracy_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seg_iou.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存：{out_dir / 'seg_iou.json'}")


if __name__ == "__main__":
    main()
