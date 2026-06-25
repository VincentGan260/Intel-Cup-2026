"""V2 模型 FP16 vs INT8 精度与速度对比评测脚本。

对比 yolo26n_v2 模型在 FP16 和 INT8 精度下的检测精度和推理速度。
使用所有数据集（BDD + IDD + DAWN），统一使用 GPU。

指标：
- 精度：mAP@0.5、mAP@0.5:0.95、小/中/大目标 AP、recall@0.5、两轮车 recall
- 速度：平均推理时间、FPS、首帧延迟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

CONF_OP = 0.25
IMGSZ = 640
COCO_OBSTACLE = {0, 1, 2, 3, 5, 7}

SRC_RULES = {
    "bdd": {"exclude": {"traffic sign", "traffic light", "train"},
            "twowheel": {"bike", "rider", "motor"}},
    "idd": {"exclude": {"traffic sign"},
            "twowheel": {"motorcycle", "rider", "autorickshaw"}},
    "dawn": {"exclude": set(),
             "twowheel": {"Bicycle", "Motorcycle"}},
}

MODEL_CONFIG = {
    "v2_fp32": "models/yolo26n_v2_openvino_model/best.xml",
    "v2_fp16": "models/yolo26n_v2_fp16_openvino_model/best.xml",
    "v2_int8": "models/yolo26n_v2_int8_openvino_model/best.xml",
}

_json_cache: dict[str, dict] = {}


def _load_json(path: Path) -> dict:
    key = str(path)
    if key not in _json_cache:
        _json_cache[key] = json.load(path.open(encoding="utf-8"))
    return _json_cache[key]


def load_gt(ds: str, ds_root: Path, image_rel: str, source_rel: str):
    rules = SRC_RULES[ds]
    boxes, tw = [], []
    src = ds_root / source_rel
    if ds == "bdd":
        data = _load_json(src)
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
    else:
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


def load_manifest(ds_root: Path, manifest_file: str = "det_manifest.csv"):
    import csv
    manifest = ds_root / "test_plan" / manifest_file
    return list(csv.DictReader(manifest.open(encoding="utf-8")))


class YOLO26Detector:
    def __init__(self, model_path: str, device: str = "GPU"):
        self.core = ov.Core()
        self.model = self.core.compile_model(str(model_path), device)
        self.infer_request = self.model.create_infer_request()
        self.input_tensor = self.model.input(0)
        self.input_shape = self.input_tensor.shape
        self.input_h, self.input_w = self.input_shape[2], self.input_shape[3]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        scale = min(self.input_w / img_w, self.input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        input_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor, scale, pad_x, pad_y, img_w, img_h

    def _postprocess(self, output, scale, pad_x, pad_y, img_w, img_h, conf_thres=0.001):
        predictions = output[0]
        boxes = []
        confs = []
        clses = []

        for det in predictions:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf >= conf_thres:
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                boxes.append([x1, y1, x2, y2])
                confs.append(conf)
                clses.append(int(cls))

        return np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32), np.array(clses, dtype=np.int32)

    def predict(self, image_path: str):
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

        input_tensor, scale, pad_x, pad_y, img_w, img_h = self._preprocess(image)
        self.infer_request.infer([input_tensor])
        output = self.infer_request.get_output_tensor(0).data

        boxes, confs, clses = self._postprocess(output, scale, pad_x, pad_y, img_w, img_h)

        keep = np.isin(clses, list(COCO_OBSTACLE))
        return boxes[keep], confs[keep]

    def predict_with_timing(self, image_path: str, warmup: int = 3):
        """带计时的推理，返回结果和推理时间（毫秒）"""
        image = cv2.imread(str(image_path))
        if image is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.0

        input_tensor, scale, pad_x, pad_y, img_w, img_h = self._preprocess(image)

        # Warmup
        for _ in range(warmup):
            self.infer_request.infer([input_tensor])

        # Timed inference
        start = time.perf_counter()
        self.infer_request.infer([input_tensor])
        end = time.perf_counter()
        infer_time_ms = (end - start) * 1000

        output = self.infer_request.get_output_tensor(0).data
        boxes, confs, clses = self._postprocess(output, scale, pad_x, pad_y, img_w, img_h)

        keep = np.isin(clses, list(COCO_OBSTACLE))
        return boxes[keep], confs[keep], infer_time_ms


def evaluate_model(precision: str, device: str = "GPU", limit: int = 0, measure_speed: bool = True):
    """评估单个模型，返回精度指标和速度指标。"""
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    cfg = load_eval_config()
    ds_root = cfg.get("dataset_root")
    if not ds_root:
        print("错误：eval.local.yaml 缺 dataset_root")
        return None
    ds_root = Path(ds_root)

    model_key = f"v2_{precision}"
    model_path = resolve(MODEL_CONFIG[model_key])
    print(f"\n{'='*78}")
    print(f"评估模型: V2 ({precision.upper()})")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    print(f"{'='*78}")

    detector = YOLO26Detector(str(model_path), device)

    rows = load_manifest(ds_root)
    if limit > 0:
        by = defaultdict(list)
        for r in rows:
            if len(by[r["dataset"]]) < limit:
                by[r["dataset"]].append(r)
        rows = [r for rs in by.values() for r in rs]

    per_src = defaultdict(lambda: {"preds": [], "targets": [], "img": []})
    infer_times = []

    start_time = time.time()
    for i, r in enumerate(rows, 1):
        ds = r["dataset"]
        img_path = ds_root / r["image"]
        gt_xyxy, gt_tw = load_gt(ds, ds_root, r["image"], r["source"])

        if measure_speed:
            pb, pc, infer_time = detector.predict_with_timing(str(img_path))
            infer_times.append(infer_time)
        else:
            pb, pc = detector.predict(str(img_path))

        per_src[ds]["preds"].append({
            "boxes": torch.tensor(pb, dtype=torch.float32),
            "scores": torch.tensor(pc, dtype=torch.float32),
            "labels": torch.zeros(len(pb), dtype=torch.int64)})
        per_src[ds]["targets"].append({
            "boxes": torch.tensor(gt_xyxy, dtype=torch.float32),
            "labels": torch.zeros(len(gt_xyxy), dtype=torch.int64)})
        per_src[ds]["img"].append({"pb": pb, "pc": pc, "gt": gt_xyxy, "tw": gt_tw, "row": r})
        if i % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  {i}/{len(rows)} ({elapsed:.1f}s)")

    elapsed_total = time.time() - start_time

    # 计算速度指标
    speed_metrics = {}
    if measure_speed and infer_times:
        infer_times = np.array(infer_times)
        speed_metrics = {
            "avg_infer_time_ms": round(float(np.mean(infer_times)), 2),
            "min_infer_time_ms": round(float(np.min(infer_times)), 2),
            "max_infer_time_ms": round(float(np.max(infer_times)), 2),
            "std_infer_time_ms": round(float(np.std(infer_times)), 2),
            "fps": round(1000.0 / np.mean(infer_times), 2),
            "p50_infer_time_ms": round(float(np.percentile(infer_times, 50)), 2),
            "p95_infer_time_ms": round(float(np.percentile(infer_times, 95)), 2),
            "p99_infer_time_ms": round(float(np.percentile(infer_times, 99)), 2),
        }

    def recall_over(records, subset=None, tw_only=False):
        m = t = 0
        for rec in records:
            if subset and not subset(rec["row"]):
                continue
            matched, _, _ = match_image(rec["pb"], rec["pc"], rec["gt"])
            sel = rec["tw"] if tw_only else np.ones(len(rec["gt"]), bool)
            m += int(np.sum(matched & sel))
            t += int(np.sum(sel))
        return round(100.0 * m / t, 2) if t else 0.0

    def precision_over(records, subset=None):
        tp = used = 0
        for rec in records:
            if subset and not subset(rec["row"]):
                continue
            _, n_tp, n_used = match_image(rec["pb"], rec["pc"], rec["gt"])
            tp += n_tp
            used += n_used
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

    result = {"model": "v2", "precision": precision, "device": device,
              "total_images": len(rows), "elapsed_time": round(elapsed_total, 2),
              "speed": speed_metrics, "sources": {}}

    for ds in ["bdd", "idd", "dawn"]:
        if ds not in per_src:
            continue
        P, T, R = per_src[ds]["preds"], per_src[ds]["targets"], per_src[ds]["img"]
        m = map_over(P, T)
        rec_all = recall_over(R)
        prec_all = precision_over(R)
        rec_tw = recall_over(R, tw_only=True)

        print(f"\n[{ds}]  {len(R)} 张")
        print(f"  mAP@0.5={m['map50']}  mAP@0.5:0.95={m['map50_95']}")
        print(f"  AP小/中/大={m['map_small']}/{m['map_medium']}/{m['map_large']}")
        print(f"  recall@0.5={rec_all}%  precision@0.5={prec_all}%")
        print(f"  ★两轮车 recall={rec_tw}%")

        result["sources"][ds] = {"n": len(R), **m, "recall": rec_all,
                                 "precision": prec_all, "twowheel_recall": rec_tw}

        if ds == "bdd":
            idx_unbiased = [i for i, rec in enumerate(R) if rec["row"].get("has_twowheel") == "0"]
            mu = map_over(P, T, idx_unbiased)
            ru = recall_over([R[i] for i in idx_unbiased])
            rec_night = recall_over(R, subset=lambda row: row.get("timeofday") == "night")
            n_night = sum(1 for rec in R if rec["row"].get("timeofday") == "night")
            print(f"  ├ 无偏总体({len(idx_unbiased)}张): mAP@0.5={mu['map50']} recall={ru}%")
            print(f"  └ 夜间({n_night}张) recall={rec_night}%")
            result["sources"][ds]["unbiased"] = {"n": len(idx_unbiased), "map50": mu["map50"], "recall": ru}
            result["sources"][ds]["night_recall"] = rec_night

    return result


def print_summary(all_results):
    """打印对比汇总表。"""
    print("\n" + "=" * 78)
    print("V2 模型 FP16 vs INT8 精度与速度对比汇总")
    print("=" * 78)

    # 速度对比
    print("\n【速度指标对比】")
    print("-" * 78)
    print(f"| {'精度':<8} | {'平均耗时(ms)':>12} | {'FPS':>8} | {'P50(ms)':>10} | {'P95(ms)':>10} | {'P99(ms)':>10} |")
    print("-" * 78)
    for r in all_results:
        s = r.get("speed", {})
        print(f"| {r['precision'].upper():<8} | {s.get('avg_infer_time_ms', '-'):>12} | {s.get('fps', '-'):>8} | {s.get('p50_infer_time_ms', '-'):>10} | {s.get('p95_infer_time_ms', '-'):>10} | {s.get('p99_infer_time_ms', '-'):>10} |")
    print("-" * 78)

    # 计算加速比
    if len(all_results) == 2:
        fp16_time = all_results[0].get("speed", {}).get("avg_infer_time_ms", 1)
        int8_time = all_results[1].get("speed", {}).get("avg_infer_time_ms", 1)
        if fp16_time > 0 and int8_time > 0:
            speedup = fp16_time / int8_time
            print(f"\nINT8 相比 FP16 加速比: {speedup:.2f}x")

    # 精度对比
    print("\n【精度指标对比】")

    print("\n1. mAP@0.5 对比:")
    print("-" * 78)
    print(f"| {'精度':<8} | {'BDD':>8} | {'IDD':>8} | {'DAWN':>8} |")
    print("-" * 78)
    for r in all_results:
        bdd = r["sources"].get("bdd", {}).get("map50", "-")
        idd = r["sources"].get("idd", {}).get("map50", "-")
        dawn = r["sources"].get("dawn", {}).get("map50", "-")
        print(f"| {r['precision'].upper():<8} | {bdd:>8} | {idd:>8} | {dawn:>8} |")
    print("-" * 78)

    # 计算精度差异
    if len(all_results) == 2:
        print("\n精度变化 (INT8 vs FP16):")
        for ds in ["bdd", "idd", "dawn"]:
            fp16_val = all_results[0]["sources"].get(ds, {}).get("map50", 0)
            int8_val = all_results[1]["sources"].get(ds, {}).get("map50", 0)
            diff = int8_val - fp16_val
            print(f"  {ds}: {diff:+.2f}%")

    print("\n2. mAP@0.5:0.95 对比:")
    print("-" * 78)
    print(f"| {'精度':<8} | {'BDD':>8} | {'IDD':>8} | {'DAWN':>8} |")
    print("-" * 78)
    for r in all_results:
        bdd = r["sources"].get("bdd", {}).get("map50_95", "-")
        idd = r["sources"].get("idd", {}).get("map50_95", "-")
        dawn = r["sources"].get("dawn", {}).get("map50_95", "-")
        print(f"| {r['precision'].upper():<8} | {bdd:>8} | {idd:>8} | {dawn:>8} |")
    print("-" * 78)

    print("\n3. Recall@0.5 对比:")
    print("-" * 78)
    print(f"| {'精度':<8} | {'BDD':>8} | {'IDD':>8} | {'DAWN':>8} |")
    print("-" * 78)
    for r in all_results:
        bdd = r["sources"].get("bdd", {}).get("recall", "-")
        idd = r["sources"].get("idd", {}).get("recall", "-")
        dawn = r["sources"].get("dawn", {}).get("recall", "-")
        print(f"| {r['precision'].upper():<8} | {bdd:>8} | {idd:>8} | {dawn:>8} |")
    print("-" * 78)

    print("\n4. 两轮车 Recall（安全关键）对比:")
    print("-" * 78)
    print(f"| {'精度':<8} | {'BDD':>8} | {'IDD':>8} | {'DAWN':>8} |")
    print("-" * 78)
    for r in all_results:
        bdd = r["sources"].get("bdd", {}).get("twowheel_recall", "-")
        idd = r["sources"].get("idd", {}).get("twowheel_recall", "-")
        dawn = r["sources"].get("dawn", {}).get("twowheel_recall", "-")
        print(f"| {r['precision'].upper():<8} | {bdd:>8} | {idd:>8} | {dawn:>8} |")
    print("-" * 78)

    print("\n5. 按目标大小 AP 对比:")
    print("-" * 78)
    for ds in ["bdd", "idd", "dawn"]:
        print(f"\n[{ds}]")
        print(f"| {'精度':<8} | {'小目标':>8} | {'中目标':>8} | {'大目标':>8} |")
        print("-" * 78)
        for r in all_results:
            src = r["sources"].get(ds, {})
            sm = src.get("map_small", "-")
            md = src.get("map_medium", "-")
            lg = src.get("map_large", "-")
            print(f"| {r['precision'].upper():<8} | {sm:>8} | {md:>8} | {lg:>8} |")


def main():
    ap = argparse.ArgumentParser(description="V2 模型 FP16 vs INT8 精度与速度对比评测")
    ap.add_argument("--device", default="GPU", help="推理设备（GPU/CPU/NPU）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时每源最多评这么多张")
    ap.add_argument("--output", default="runs/accuracy_eval/compare_v2_fp16_int8.json", help="结果输出路径")
    ap.add_argument("--no-speed", action="store_true", help="跳过速度测试")
    args = ap.parse_args()

    print(f"V2 模型 FP32 vs FP16 vs INT8 精度与速度对比评测")
    print(f"设备: NPU")
    print(f"数据集: BDD + IDD + DAWN")
    print(f"精度: FP32 vs FP16 vs INT8")
    if args.limit > 0:
        print(f"限制: 每源最多 {args.limit} 张")

    all_results = []

    for precision in ["fp16", "int8"]:
        result = evaluate_model(precision, args.device, args.limit, measure_speed=not args.no_speed)
        if result:
            all_results.append(result)

    print_summary(all_results)

    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()