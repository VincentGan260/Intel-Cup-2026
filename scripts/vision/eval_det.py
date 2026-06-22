"""检测精度评测（manifest 驱动，类别无关障碍物检测）。

按 测试说明.md / 检测类别对照.md：
  - 三源（BDD / IDD / DAWN）**分别报，不混**；
  - 障碍物 = 人 + 各类车（按各源映射排除 traffic sign/light/train）；类别无关匹配；
  - 指标：mAP@0.5、mAP@0.5:0.95、小/中/大目标 AP（torchmetrics 内置 area 划分）、
          整体 recall@0.5、**两轮车 recall**（安全关键）；
  - BDD 额外：has_twowheel=0 的 617 张 = 无偏总体；夜间 recall。

评测对象：OpenVINO FP32 yolo26n（ultralytics 加载 openvino_model 目录）。

运行：
    D:/Anaconda_envs/envs/intel/python.exe scripts/vision/eval_det.py
    ... --limit 30      # 冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.dataset_paths import load_eval_config, resolve

CONF_OP = 0.25            # recall 操作点
IMGSZ = 640
# COCO 里算「障碍物」的类别 id（yolo 预测）
COCO_OBSTACLE = {0, 1, 2, 3, 5, 7}   # person, bicycle, car, motorcycle, bus, truck

# 各源：排除类（非障碍物）+ 两轮车类
SRC_RULES = {
    "bdd":  {"exclude": {"traffic sign", "traffic light", "train"},
             "twowheel": {"bike", "rider", "motor"}},
    "idd":  {"exclude": {"traffic sign"},
             "twowheel": {"motorcycle", "rider", "autorickshaw"}},
    "dawn": {"exclude": set(),
             "twowheel": {"Bicycle", "Motorcycle"}},
}

_json_cache: dict[str, dict] = {}


def _load_json(path: Path) -> dict:
    key = str(path)
    if key not in _json_cache:
        _json_cache[key] = json.load(path.open(encoding="utf-8"))
    return _json_cache[key]


def load_gt(ds: str, ds_root: Path, image_rel: str, source_rel: str):
    """返回 (boxes_xyxy[N,4], twowheel_flags[N] bool)。"""
    rules = SRC_RULES[ds]
    boxes, tw = [], []
    src = ds_root / source_rel
    if ds == "bdd":
        data = _load_json(src)  # 逐图 json（每个 source 各一份；缓存避免重复读）
        objs = (data.get("frames") or [{}])[0].get("objects", [])
        for o in objs:
            if "box2d" not in o:
                continue
            cat = o.get("category", "")
            if cat in rules["exclude"]:
                continue
            b = o["box2d"]
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            tw.append(cat in rules["twowheel"])
    else:  # idd / dawn：大 json 按文件名索引
        data = _load_json(src)
        entry = data.get(Path(image_rel).name)
        for o in (entry or {}).get("objects", []):
            cat = o.get("category", "")
            if cat in rules["exclude"]:
                continue
            x, y, w, h = o["bbox"]
            boxes.append([x, y, x + w, y + h])
            tw.append(cat in rules["twowheel"])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(tw, dtype=bool)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a[N,4] b[M,4] xyxy -> IoU[N,M]。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(pred_xyxy, pred_conf, gt_xyxy, conf=CONF_OP, iou_thr=0.5):
    """类别无关贪心匹配。返回 (matched_gt_bool, n_tp, n_pred_used)。
    pred 先按 conf 过滤+降序；n_tp=命中的预测数，n_pred_used=参与的预测数（用于 precision）。"""
    keep = pred_conf >= conf
    pb = pred_xyxy[keep]
    pc = pred_conf[keep]
    order = np.argsort(-pc)
    pb = pb[order]
    matched = np.zeros(len(gt_xyxy), dtype=bool)
    n_used = len(pb)
    if len(gt_xyxy) == 0 or len(pb) == 0:
        return matched, 0, n_used
    ious = box_iou(pb, gt_xyxy)
    tp = 0
    for i in range(len(pb)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not matched[j]:
            matched[j] = True
            tp += 1
    return matched, tp, n_used


def main() -> None:
    ap = argparse.ArgumentParser(description="检测障碍物评测（manifest 驱动）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时每源最多评这么多张")
    ap.add_argument("--weights", default=None,
                    help="覆盖检测模型（OpenVINO 目录或 .xml）；默认用 eval.yaml 的 fp32")
    ap.add_argument("--tag", default=None,
                    help="给输出 json 加后缀，如 finetune → det_obstacle_finetune.json")
    ap.add_argument("--manifest", default="det_manifest.csv",
                    help="test_plan 下的 manifest 文件名（默认 det_manifest.csv）")
    args = ap.parse_args()

    import csv
    from ultralytics import YOLO
    from torchmetrics.detection import MeanAveragePrecision
    import torch

    cfg = load_eval_config()
    ds_root = cfg.get("dataset_root")
    if not ds_root:
        print("错误：eval.local.yaml 缺 dataset_root"); return
    ds_root = Path(ds_root)
    manifest = ds_root / "test_plan" / args.manifest
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if args.limit > 0:
        by = defaultdict(list)
        for r in rows:
            if len(by[r["dataset"]]) < args.limit:
                by[r["dataset"]].append(r)
        rows = [r for rs in by.values() for r in rs]

    if args.weights:
        wp = resolve(args.weights)
        model_dir = wp if wp.is_dir() else wp.parent
    else:
        model_dir = resolve(cfg["models"]["detection"]["fp32"]).parent
    model = YOLO(str(model_dir))   # ultralytics 加载 openvino_model 目录
    print(f"模型: {model_dir}\nmanifest: {len(rows)} 行")

    # 每源累积：torchmetrics 用 preds/targets；recall 用每图记录
    per_src = defaultdict(lambda: {"preds": [], "targets": [], "img": []})

    for i, r in enumerate(rows, 1):
        ds = r["dataset"]
        img_path = ds_root / r["image"]
        gt_xyxy, gt_tw = load_gt(ds, ds_root, r["image"], r["source"])

        res = model.predict(str(img_path), imgsz=IMGSZ, conf=0.001, device="cpu", verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            cls = res.boxes.cls.cpu().numpy().astype(int)
            keep = np.isin(cls, list(COCO_OBSTACLE))
            pb = res.boxes.xyxy.cpu().numpy()[keep]
            pc = res.boxes.conf.cpu().numpy()[keep]
        else:
            pb = np.zeros((0, 4), np.float32); pc = np.zeros((0,), np.float32)

        per_src[ds]["preds"].append({
            "boxes": torch.tensor(pb, dtype=torch.float32),
            "scores": torch.tensor(pc, dtype=torch.float32),
            "labels": torch.zeros(len(pb), dtype=torch.int64)})
        per_src[ds]["targets"].append({
            "boxes": torch.tensor(gt_xyxy, dtype=torch.float32),
            "labels": torch.zeros(len(gt_xyxy), dtype=torch.int64)})
        per_src[ds]["img"].append({"pb": pb, "pc": pc, "gt": gt_xyxy, "tw": gt_tw, "row": r})
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}")

    def recall_over(records, subset=None, tw_only=False):
        m = t = 0
        for rec in records:
            if subset and not subset(rec["row"]):
                continue
            matched, _, _ = match_image(rec["pb"], rec["pc"], rec["gt"])
            sel = rec["tw"] if tw_only else np.ones(len(rec["gt"]), bool)
            m += int(np.sum(matched & sel)); t += int(np.sum(sel))
        return round(100.0 * m / t, 2) if t else 0.0

    def precision_over(records, subset=None):
        tp = used = 0
        for rec in records:
            if subset and not subset(rec["row"]):
                continue
            _, n_tp, n_used = match_image(rec["pb"], rec["pc"], rec["gt"])
            tp += n_tp; used += n_used
        return round(100.0 * tp / used, 2) if used else 0.0

    def map_over(preds, targets, idxs=None):
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        if idxs is None:
            metric.update(preds, targets)
        else:
            metric.update([preds[i] for i in idxs], [targets[i] for i in idxs])
        r = metric.compute()
        g = lambda k: round(float(r[k]) * 100, 2) if float(r[k]) >= 0 else -1.0
        return {"map50": g("map_50"), "map50_95": g("map"),
                "map_small": g("map_small"), "map_medium": g("map_medium"), "map_large": g("map_large")}

    out = {"dataset_root": str(ds_root), "sources": {}}
    print("\n" + "=" * 78)
    print("检测：按源分别报（障碍物，类别无关）")
    print("=" * 78)
    for ds in ["bdd", "idd", "dawn"]:
        if ds not in per_src:
            continue
        P, T, R = per_src[ds]["preds"], per_src[ds]["targets"], per_src[ds]["img"]
        m = map_over(P, T)
        rec_all = recall_over(R)
        prec_all = precision_over(R)
        rec_tw = recall_over(R, tw_only=True)
        print(f"\n[{ds}]  {len(R)} 张")
        print(f"  mAP@0.5={m['map50']}  mAP@0.5:0.95={m['map50_95']}  "
              f"AP小/中/大={m['map_small']}/{m['map_medium']}/{m['map_large']}")
        print(f"  recall@0.5={rec_all}%  precision@0.5={prec_all}%  ★两轮车 recall={rec_tw}%")
        out["sources"][ds] = {"n": len(R), **m, "recall": rec_all,
                              "precision": prec_all, "twowheel_recall": rec_tw}

        if ds == "bdd":
            idx_unbiased = [i for i, rec in enumerate(R) if rec["row"].get("has_twowheel") == "0"]
            mu = map_over(P, T, idx_unbiased)
            ru = recall_over([R[i] for i in idx_unbiased])
            rec_night = recall_over(R, subset=lambda row: row.get("timeofday") == "night")
            n_night = sum(1 for rec in R if rec["row"].get("timeofday") == "night")
            print(f"  ├ 无偏总体(617随机,has_twowheel=0,{len(idx_unbiased)}张): mAP@0.5={mu['map50']} recall={ru}%")
            print(f"  └ 夜间({n_night}张) recall={rec_night}%")
            out["sources"][ds]["unbiased_617"] = {"n": len(idx_unbiased), "map50": mu["map50"], "recall": ru}
            out["sources"][ds]["night_recall"] = rec_night

    print("\n> 三源不混；两轮车 recall 是安全关键；BDD 617 随机=无偏总体；其余各自报。")
    out_dir = resolve(cfg.get("output", {}).get("accuracy_dir", "runs/accuracy_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"det_obstacle_{args.tag}.json" if args.tag else "det_obstacle.json"
    (out_dir / fname).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已存：{out_dir / fname}")


if __name__ == "__main__":
    main()
