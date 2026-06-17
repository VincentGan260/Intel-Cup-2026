"""road-adas 多设备性能对比测试脚本。

对比维度：CPU vs GPU(iGPU) vs NPU vs AUTO
测试指标：
  - 速度性能：单帧耗时、平均帧率、P95延迟
  - 资源占用：CPU占用率、内存使用、GPU内存
  - 运行稳定性：标准差、最大/最小波动
  - 分割精度：road IoU、像素准确率

控制变量：
  - 统一输入尺寸：1024x1024
  - 统一推理精度：FP32
  - 统一模型：road-adas
  - 统一测试数据集：相同图片集

运行：
    python scripts/vision/compare_devices_road_adas.py --num_samples 50 --vis_count 3
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import psutil
import openvino as ov

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_paths import load_eval_config, resolve
from src.vision.segmentation.postprocess import (
    run_openvino_adas_forward,
    logits_chw_to_label_map,
)
from src.vision.common.visualize import blend_binary_mask, save_bgr


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, road_value: int, ignore_value: int = 255) -> float:
    valid = (gt_mask != ignore_value)
    pred_road = (pred_mask > 0) & valid
    gt_road = (gt_mask == road_value) & valid
    
    if np.sum(pred_road | gt_road) == 0:
        return 1.0 if np.sum(gt_road) == 0 else 0.0
    
    intersection = np.sum(pred_road & gt_road)
    union = np.sum(pred_road | gt_road)
    return intersection / union if union > 0 else 0.0


def compute_pixel_acc(pred_mask: np.ndarray, gt_mask: np.ndarray, road_value: int, ignore_value: int = 255) -> float:
    valid = (gt_mask != ignore_value)
    if np.sum(valid) == 0:
        return 1.0
    
    pred_road = (pred_mask > 0)
    gt_road = (gt_mask == road_value)
    correct = np.sum((pred_road == gt_road) & valid)
    return correct / np.sum(valid)


def load_gt_mask(label_path: Path, label_suffix: str, image_name: str, image_suffix: str) -> np.ndarray:
    label_name = image_name.replace(image_suffix, label_suffix)
    label_path_full = label_path / label_name
    if not label_path_full.exists():
        candidates = list(label_path.rglob(label_name))
        if candidates:
            label_path_full = candidates[0]
    
    if not label_path_full.exists():
        return None
    
    return cv2.imread(str(label_path_full), cv2.IMREAD_GRAYSCALE)


def find_images(dataset: dict) -> list[tuple[Path, str]]:
    images_dir = resolve(dataset["images_dir"])
    image_suffix = dataset["image_suffix"]
    recursive = dataset.get("recursive", False)
    
    if not images_dir.exists():
        return []
    
    if recursive:
        images = list(images_dir.rglob(f"*{image_suffix}"))
    else:
        images = list(images_dir.glob(f"*{image_suffix}"))
    
    return [(img, img.name) for img in sorted(images)]


def measure_cpu_usage(duration: float = 0.05) -> float:
    cpu_percentages = []
    start = time.time()
    while time.time() - start < duration:
        cpu_percentages.append(psutil.cpu_percent(interval=0.02))
    return statistics.mean(cpu_percentages) if cpu_percentages else 0.0


def measure_memory_usage() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def bench_single_device(model_xml: Path, device: str, warmup: int, iters: int, input_h: int, input_w: int) -> dict:
    try:
        core = ov.Core()
        model = core.read_model(str(model_xml))
        
        model.reshape([1, 3, input_h, input_w])
        
        compiled = core.compile_model(model, device)
        
        feed = {}
        for port in compiled.inputs:
            pshape = port.get_partial_shape()
            shape = [d.get_length() if d.is_static else 1 for d in pshape]
            feed[port.get_any_name()] = np.random.rand(*shape).astype(np.float32)
        
        for _ in range(warmup):
            compiled(feed)
        
        times = []
        cpu_usages = []
        memory_usages = []
        
        for _ in range(iters):
            t0 = time.perf_counter()
            compiled(feed)
            times.append((time.perf_counter() - t0) * 1000.0)
            
            cpu_usage = measure_cpu_usage(0.03)
            cpu_usages.append(cpu_usage)
            
            memory_usages.append(measure_memory_usage())
        
        times_sorted = sorted(times)
        p95_idx = min(len(times_sorted) - 1, int(round(0.95 * (len(times_sorted) - 1))))
        p95 = times_sorted[p95_idx]
        
        return {
            "mean_ms": round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(times_sorted[0], 2),
            "max_ms": round(times_sorted[-1], 2),
            "std_ms": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
            "fps": round(1000.0 / statistics.mean(times), 1),
            "mean_cpu_pct": round(statistics.mean(cpu_usages), 2),
            "mean_memory_mb": round(statistics.mean(memory_usages), 2),
            "success": True
        }
    except Exception as e:
        print(f"  [失败] {device}: {type(e).__name__}: {str(e)[:100]}")
        return {"success": False}


def evaluate_on_dataset(model_xml: Path, device: str, dataset: dict, num_samples: int, input_h: int, input_w: int) -> dict:
    try:
        core = ov.Core()
        model = core.read_model(str(model_xml))
        
        model.reshape([1, 3, input_h, input_w])
        
        compiled = core.compile_model(model, device)
        input_name = compiled.input(0).get_any_name()
    except Exception as e:
        print(f"  [编译失败] {device}: {type(e).__name__}: {str(e)[:80]}")
        return {"iou": 0.0, "pixel_acc": 0.0, "count": 0,
                "latencies": [], "mean_latency_ms": 0.0, "fps": 0.0,
                "cpu_usages": [], "mean_cpu_pct": 0.0, "pred_masks": {}, "success": False}
    
    images = find_images(dataset)[:num_samples]
    if not images:
        return {"iou": 0.0, "pixel_acc": 0.0, "count": 0,
                "latencies": [], "mean_latency_ms": 0.0, "fps": 0.0,
                "cpu_usages": [], "mean_cpu_pct": 0.0, "pred_masks": {}, "success": True}
    
    road_value = dataset["road_value"]
    ignore_value = dataset.get("ignore_value", 255)
    labels_dir = resolve(dataset["labels_dir"])
    
    ious = []
    pixel_accs = []
    latencies = []
    cpu_usages = []
    pred_masks = {}
    
    print(f"    测试 {len(images)} 张图片...")
    
    for img_path, img_name in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        gt_mask = load_gt_mask(labels_dir, dataset["label_suffix"], img_name, dataset["image_suffix"])
        if gt_mask is None:
            continue
        
        t0 = time.perf_counter()
        logits_chw = run_openvino_adas_forward(compiled, input_name, image, input_h, input_w)
        latency = (time.perf_counter() - t0) * 1000.0
        
        label_small = logits_chw_to_label_map(logits_chw)
        pred_mask = cv2.resize(label_small.astype(np.uint8), 
                               (gt_mask.shape[1], gt_mask.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        pred_mask = (pred_mask == 1).astype(np.uint8)
        
        pred_masks[img_name] = pred_mask
        
        iou = compute_iou(pred_mask, gt_mask, road_value, ignore_value)
        pixel_acc = compute_pixel_acc(pred_mask, gt_mask, road_value, ignore_value)
        
        cpu_usage = measure_cpu_usage(0.02)
        
        ious.append(iou)
        pixel_accs.append(pixel_acc)
        latencies.append(latency)
        cpu_usages.append(cpu_usage)
    
    if not ious:
        return {"iou": 0.0, "pixel_acc": 0.0, "count": 0,
                "latencies": [], "mean_latency_ms": 0.0, "fps": 0.0,
                "cpu_usages": [], "mean_cpu_pct": 0.0, "pred_masks": {}, "success": True}
    
    return {
        "iou": round(statistics.mean(ious) * 100, 2),
        "pixel_acc": round(statistics.mean(pixel_accs) * 100, 2),
        "count": len(ious),
        "latencies": latencies,
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "fps": round(1000.0 / statistics.mean(latencies), 1) if latencies else 0.0,
        "cpu_usages": cpu_usages,
        "mean_cpu_pct": round(statistics.mean(cpu_usages), 2),
        "pred_masks": pred_masks,
        "success": True
    }


def visualize_comparison(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: list[Tuple[str, np.ndarray]],
    road_value: int,
    output_path: Path,
    fps_info: dict[str, float],
    iou_info: dict[str, float],
) -> None:
    gt_road = (gt_mask == road_value).astype(np.uint8)
    
    titles = ["原图", "真值"] + [p[0] for p in pred_masks]
    
    images = [image_bgr]
    
    gt_overlay = blend_binary_mask(image_bgr, gt_road, (0, 200, 0), 0.4)
    images.append(gt_overlay)
    
    colors = [(0, 0, 255), (0, 200, 255), (255, 0, 200), (255, 200, 0)]
    for i, (device_name, pred_mask) in enumerate(pred_masks):
        pred_overlay = blend_binary_mask(image_bgr, pred_mask, colors[i], 0.4)
        images.append(pred_overlay)
    
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    
    resized_images = []
    for img in images:
        if img.shape[0] != max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            resized = cv2.resize(img, (new_width, max_height), interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized)
        else:
            resized_images.append(img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    title_height = 50
    
    final_images = []
    for i, img in enumerate(resized_images):
        title_img = np.full((title_height, img.shape[1], 3), 240, dtype=np.uint8)
        
        title_text = titles[i]
        if title_text in fps_info and title_text in iou_info:
            subtitle = f"FPS: {fps_info[title_text]:.1f} | IoU: {iou_info[title_text]:.1f}%"
            combined_text = f"{title_text}\n{subtitle}"
        else:
            combined_text = title_text
        
        lines = combined_text.split("\n")
        y_offset = 20
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            cv2.putText(title_img, line, (text_x, y_offset), font, font_scale, (0, 0, 0), font_thickness)
            y_offset += text_size[1] + 3
        
        final = np.vstack([title_img, img])
        final_images.append(final)
    
    combined = np.hstack(final_images)
    save_bgr(output_path, combined)


def print_table(title, headers, rows):
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    col_widths = [max(w, 6) for w in col_widths]
    
    total_width = sum(col_widths) + len(col_widths) * 3 + 1
    print("-" * total_width)
    print(title)
    print("-" * total_width)
    
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(f"| {header_str} |")
    print("-" * total_width)
    
    for row in rows:
        row_str = " | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths))
        print(f"| {row_str} |")
    print("-" * total_width)


def main() -> None:
    parser = argparse.ArgumentParser(description="road-adas 多设备性能对比测试")
    parser.add_argument("--num_samples", type=int, default=None, help="每个数据集评测样本数")
    parser.add_argument("--vis_count", type=int, default=3, help="每个数据集生成可视化对比图的数量")
    args = parser.parse_args()
    
    cfg = load_eval_config()
    
    UNIFIED_INPUT_HW = (1024, 1024)
    MODEL_XML = resolve(cfg["models"]["segmentation"]["road_adas"])
    
    devices = [
        {"name": "CPU", "description": "中央处理器"},
        {"name": "GPU", "description": "集成显卡 (iGPU)"},
        {"name": "NPU", "description": "神经处理单元"},
        {"name": "AUTO", "description": "自动选择最优设备"},
    ]
    
    datasets = cfg.get("segmentation", {}).get("datasets", [])
    num_samples = args.num_samples or cfg.get("segmentation", {}).get("num_samples", 50)
    
    print("\n" + "=" * 80)
    print("road-adas 多设备性能对比测试")
    print("=" * 80)
    print("\n【控制变量说明】")
    print("-" * 80)
    print(f"  {'测试模型':<20}: road-segmentation-adas-0001")
    print(f"  {'推理精度':<20}: FP32")
    print(f"  {'输入尺寸':<20}: {UNIFIED_INPUT_HW[0]}x{UNIFIED_INPUT_HW[1]} (统一)")
    print(f"  {'评测样本数':<20}: {num_samples} 张/数据集")
    print(f"  {'对比设备':<20}: {', '.join([d['name'] for d in devices])}")
    print("-" * 80 + "\n")
    
    out_dir = resolve(cfg.get("output", {}).get("accuracy_dir", "runs/accuracy_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("【速度基准测试】")
    print("-" * 80)
    speed_results = {}
    
    speed_rows = []
    for d in devices:
        result = bench_single_device(MODEL_XML, d["name"], 20, 200, UNIFIED_INPUT_HW[0], UNIFIED_INPUT_HW[1])
        if result["success"]:
            speed_rows.append([
                d["name"],
                result["mean_ms"],
                result["median_ms"],
                result["p95_ms"],
                result["std_ms"],
                result["fps"],
                result["mean_cpu_pct"],
                result["mean_memory_mb"]
            ])
            speed_results[d["name"]] = result
        else:
            speed_rows.append([d["name"], "-", "-", "-", "-", "-", "-", "-"])
            speed_results[d["name"]] = result
    
    print_table("", ["设备", "均值(ms)", "中位(ms)", "p95(ms)", "标准差(ms)", "FPS", "CPU(%)", "内存(MB)"], speed_rows)
    
    print("\n\n【各数据集精度评测】")
    accuracy_results = {}
    
    all_results_rows = []
    
    for ds in datasets:
        print(f"\n数据集: {ds['name']}")
        print("-" * 80)
        
        accuracy_results[ds["name"]] = {}
        
        ds_rows = []
        
        for d in devices:
            if not speed_results.get(d["name"], {}).get("success"):
                print(f"    跳过 {d['name']} (速度测试失败)")
                accuracy_results[ds["name"]][d["name"]] = {"success": False}
                ds_rows.append([d["name"], 0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            
            result = evaluate_on_dataset(MODEL_XML, d["name"], ds, num_samples, UNIFIED_INPUT_HW[0], UNIFIED_INPUT_HW[1])
            accuracy_results[ds["name"]][d["name"]] = result
            
            if result["success"]:
                ds_rows.append([
                    d["name"],
                    result["count"],
                    result["iou"],
                    result["pixel_acc"],
                    result["mean_latency_ms"],
                    result["fps"],
                    result["mean_cpu_pct"]
                ])
            else:
                ds_rows.append([d["name"], 0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print_table("", ["设备", "图片数", "road IoU(%)", "像素准确率(%)", "延迟(ms)", "FPS", "CPU(%)"], ds_rows)
        
        for row in ds_rows:
            all_results_rows.append([ds["name"]] + row)
        
        if args.vis_count > 0:
            images = find_images(ds)[:num_samples]
            road_value = ds["road_value"]
            labels_dir = resolve(ds["labels_dir"])
            
            vis_count = min(args.vis_count, len(images))
            vis_index = 0
            
            for img_path, img_name in images:
                if vis_index >= vis_count:
                    break
                
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                gt_mask = load_gt_mask(labels_dir, ds["label_suffix"], img_name, ds["image_suffix"])
                if gt_mask is None:
                    continue
                
                pred_masks_for_vis = []
                fps_info = {}
                iou_info = {}
                
                for d in devices:
                    if d["name"] in accuracy_results[ds["name"]]:
                        d_result = accuracy_results[ds["name"]][d["name"]]
                        if d_result.get("success") and img_name in d_result.get("pred_masks", {}):
                            pred_masks_for_vis.append((d["name"], d_result["pred_masks"][img_name]))
                            fps_info[d["name"]] = d_result["fps"]
                            iou_info[d["name"]] = d_result["iou"]
                
                if len(pred_masks_for_vis) >= 2:
                    vis_path = out_dir / f"device_comparison_{ds['name']}_{vis_index + 1}.jpg"
                    visualize_comparison(image, gt_mask, pred_masks_for_vis, road_value, vis_path, fps_info, iou_info)
                    vis_index += 1
            
            if vis_index > 0:
                print(f"    可视化对比图已保存至 {out_dir}")
    
    print("\n\n" + "=" * 80)
    print("【综合对比】")
    print("=" * 80)
    
    if speed_results:
        print("\n速度对比:")
        print("-" * 70)
        
        metrics = [
            ("平均延迟(ms)", "mean_ms"),
            ("P95延迟(ms)", "p95_ms"),
            ("延迟标准差(ms)", "std_ms"),
            ("FPS", "fps"),
            ("CPU占用(%)", "mean_cpu_pct"),
            ("内存(MB)", "mean_memory_mb"),
        ]
        
        speed_compare_rows = []
        for name, key in metrics:
            values = []
            for d in devices:
                val = speed_results[d["name"]].get(key, "-")
                values.append(val)
            speed_compare_rows.append([name] + values)
        
        headers = ["指标"] + [d["name"] for d in devices]
        print_table("", headers, speed_compare_rows)
    
    if accuracy_results:
        print("\n精度对比 (road IoU):")
        print("-" * 60)
        
        iou_compare_rows = []
        for ds_name, ds_results in accuracy_results.items():
            row = [ds_name]
            for d in devices:
                val = ds_results[d["name"]].get("iou", "-") if ds_results[d["name"]].get("success") else "-"
                row.append(val)
            iou_compare_rows.append(row)
        
        headers = ["数据集"] + [d["name"] for d in devices]
        print_table("", headers, iou_compare_rows)
    
    print("\n稳定性分析:")
    print("-" * 70)
    
    stability_rows = []
    for d in devices:
        speed_info = speed_results.get(d["name"], {})
        if speed_info.get("success"):
            lat_std = speed_info.get("std_ms", 0)
            lat_mean = speed_info.get("mean_ms", 1)
            lat_fluct = (lat_std / lat_mean * 100) if lat_mean > 0 else 0
            
            cpu_std = 0
            cpu_mean = speed_info.get("mean_cpu_pct", 1)
            
            for ds_results in accuracy_results.values():
                d_result = ds_results.get(d["name"], {})
                if d_result.get("success") and d_result.get("cpu_usages"):
                    cpu_std = max(cpu_std, statistics.stdev(d_result["cpu_usages"]) if len(d_result["cpu_usages"]) > 1 else 0)
            
            cpu_fluct = (cpu_std / cpu_mean * 100) if cpu_mean > 0 else 0
            
            stability_rows.append([d["name"], lat_std, f"{lat_fluct:.1f}%", f"{cpu_fluct:.1f}%"])
    
    print_table("", ["设备", "延迟标准差(ms)", "延迟波动(%)", "CPU波动(%)"], stability_rows)
    
    clean_accuracy = {}
    for ds_name, ds_results in accuracy_results.items():
        clean_accuracy[ds_name] = {}
        for d_name, d_results in ds_results.items():
            clean_result = {k: v for k, v in d_results.items() if k != "pred_masks"}
            clean_result["latencies"] = [float(x) for x in clean_result.get("latencies", [])]
            clean_result["cpu_usages"] = [float(x) for x in clean_result.get("cpu_usages", [])]
            clean_accuracy[ds_name][d_name] = clean_result
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "road-segmentation-adas-0001",
        "precision": "FP32",
        "input_size": f"{UNIFIED_INPUT_HW[0]}x{UNIFIED_INPUT_HW[1]}",
        "num_samples": num_samples,
        "speed": speed_results,
        "accuracy": clean_accuracy
    }
    
    out_path = out_dir / "road_adas_device_comparison.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    csv_path = out_dir / "road_adas_device_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["数据集", "设备", "输入尺寸", "图片数", "road IoU(%)", "像素准确率(%)", 
                         "平均延迟(ms)", "FPS", "CPU占用(%)", "内存(MB)"])
        
        for ds_name, ds_results in accuracy_results.items():
            for d_name, d_results in ds_results.items():
                speed_info = speed_results.get(d_name, {})
                
                writer.writerow([
                    ds_name, d_name, f"{UNIFIED_INPUT_HW[0]}x{UNIFIED_INPUT_HW[1]}",
                    d_results.get("count", 0),
                    d_results.get("iou", 0), d_results.get("pixel_acc", 0),
                    d_results.get("mean_latency_ms", 0), d_results.get("fps", 0),
                    d_results.get("mean_cpu_pct", 0),
                    speed_info.get("mean_memory_mb", 0)
                ])
    
    print(f"\n\n结果已保存至: {out_dir}")
    print(f"  - road_adas_device_comparison.json")
    print(f"  - road_adas_device_comparison.csv")
    print(f"  - device_comparison_*.jpg (可视化对比图)")


if __name__ == "__main__":
    main()