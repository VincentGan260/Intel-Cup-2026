"""语义分割模型对比：PIDNet-S vs road-adas（精度 + 速度 + 可视化）。

在有标注的数据集（如 BDD100K）上对比两个模型：
  - 精度指标：road IoU（主）、像素准确率
  - 速度指标：延迟（均值/中位/p95）、FPS、模型体积
  - 可视化：并排展示原图、真值标注、两个模型的分割结果
  - 输出：终端表格 + JSON 结果 + CSV 结果 + 可视化对比图

运行：
    conda run -n intel python scripts/vision/compare_segmentation_models.py
    conda run -n intel python scripts/vision/compare_segmentation_models.py --datasets bdd100k --vis_count 5
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
import openvino as ov

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_paths import load_eval_config, resolve
from src.vision.segmentation.postprocess import (
    run_openvino_adas_forward,
    run_openvino_pidnet_forward,
    logits_chw_to_label_map,
)
from src.vision.common.visualize import blend_binary_mask, save_bgr


def ir_size_mb(xml: Path) -> float:
    bin_ = xml.with_suffix(".bin")
    return round(bin_.stat().st_size / 1e6, 1) if bin_.is_file() else 0.0


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, road_value: int, ignore_value: int = 255) -> float:
    """计算 road IoU"""
    valid = (gt_mask != ignore_value)
    pred_road = (pred_mask > 0) & valid
    gt_road = (gt_mask == road_value) & valid
    
    if np.sum(pred_road | gt_road) == 0:
        return 1.0 if np.sum(gt_road) == 0 else 0.0
    
    intersection = np.sum(pred_road & gt_road)
    union = np.sum(pred_road | gt_road)
    return intersection / union if union > 0 else 0.0


def compute_pixel_acc(pred_mask: np.ndarray, gt_mask: np.ndarray, road_value: int, ignore_value: int = 255) -> float:
    """计算像素准确率（只在 valid 区域上）"""
    valid = (gt_mask != ignore_value)
    if np.sum(valid) == 0:
        return 1.0
    
    pred_road = (pred_mask > 0)
    gt_road = (gt_mask == road_value)
    correct = np.sum((pred_road == gt_road) & valid)
    return correct / np.sum(valid)


def load_gt_mask(label_path: Path, label_suffix: str, image_name: str, image_suffix: str) -> np.ndarray:
    """加载真值标注"""
    label_name = image_name.replace(image_suffix, label_suffix)
    label_path_full = label_path / label_name
    if not label_path_full.exists():
        # 支持多层目录查找（如 ACDC: weather/train/video_seq/*.png）
        candidates = list(label_path.rglob(label_name))
        if candidates:
            label_path_full = candidates[0]
    
    if not label_path_full.exists():
        return None
    
    return cv2.imread(str(label_path_full), cv2.IMREAD_GRAYSCALE)


def visualize_comparison(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray,
    pred_masks: list[Tuple[str, np.ndarray]],
    road_value: int,
    output_path: Path,
) -> None:
    """生成可视化对比图：原图 | 真值 | 模型1结果 | 模型2结果"""
    # 将真值mask转为二值（road区域）
    gt_road = (gt_mask == road_value).astype(np.uint8)
    
    # 创建标题栏
    titles = ["原图", "真值", pred_masks[0][0], pred_masks[1][0]]
    
    # 准备所有图像
    images = [image_bgr]
    
    # 添加真值可视化（绿色叠加）
    gt_overlay = blend_binary_mask(image_bgr, gt_road, (0, 200, 0), 0.4)
    images.append(gt_overlay)
    
    # 添加模型预测结果
    colors = [(0, 0, 255), (255, 0, 0)]  # road-adas: 红色, PIDNet-S: 蓝色
    for i, (model_name, pred_mask) in enumerate(pred_masks):
        pred_overlay = blend_binary_mask(image_bgr, pred_mask, colors[i], 0.4)
        images.append(pred_overlay)
    
    # 统一高度
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    
    # 调整所有图像到相同高度
    resized_images = []
    for img in images:
        if img.shape[0] != max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            resized = cv2.resize(img, (new_width, max_height), interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized)
        else:
            resized_images.append(img)
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    title_height = 40
    
    final_images = []
    for i, img in enumerate(resized_images):
        # 添加标题栏
        title_img = np.full((title_height, img.shape[1], 3), 255, dtype=np.uint8)
        text_size = cv2.getTextSize(titles[i], font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        cv2.putText(title_img, titles[i], (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
        
        # 拼接标题和图像
        final = np.vstack([title_img, img])
        final_images.append(final)
    
    # 横向拼接所有图像
    combined = np.hstack(final_images)
    
    # 保存
    save_bgr(output_path, combined)


def find_images(dataset: dict) -> list[tuple[Path, str]]:
    """查找数据集中的图片"""
    images_dir = resolve(dataset["images_dir"])
    image_suffix = dataset["image_suffix"]
    recursive = dataset.get("recursive", False)
    
    if not images_dir.exists():
        return []
    
    if recursive:
        # 支持多层目录结构（如 ACDC: weather/train/video_seq/*.png）
        images = list(images_dir.rglob(f"*{image_suffix}"))
    else:
        images = list(images_dir.glob(f"*{image_suffix}"))
    
    return [(img, img.name) for img in sorted(images)]


def bench_model(model_xml: Path, device: str, warmup: int, iters: int, input_h: int, input_w: int, model_name: str) -> dict:
    """基准测试单个模型的速度"""
    try:
        core = ov.Core()
        compiled = core.compile_model(str(model_xml), device)
        
        feed = {}
        for port in compiled.inputs:
            pshape = port.get_partial_shape()
            shape = [d.get_length() if d.is_static else 1 for d in pshape]
            feed[port.get_any_name()] = np.random.rand(*shape).astype(np.float32)
        
        for _ in range(warmup):
            compiled(feed)
        
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            compiled(feed)
            times.append((time.perf_counter() - t0) * 1000.0)
        
        times_sorted = sorted(times)
        p95 = times_sorted[min(len(times_sorted) - 1, int(round(0.95 * (len(times_sorted) - 1))))]
        
        return {
            "mean_ms": round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(times_sorted[0], 2),
            "max_ms": round(times_sorted[-1], 2),
            "fps": round(1000.0 / statistics.mean(times), 1),
            "size_mb": ir_size_mb(model_xml),
            "success": True
        }
    except Exception as e:
        print(f"  [失败] {device}: {type(e).__name__}: {str(e)[:60]}")
        return {"success": False}


def evaluate_model(model_xml: Path, model_name: str, dataset: dict, num_samples: int) -> dict:
    """在数据集上评估模型精度"""
    core = ov.Core()
    compiled = core.compile_model(str(model_xml), "CPU")
    input_name = compiled.input(0).get_any_name()
    
    if model_name == "PIDNet-S":
        input_h, input_w = 1024, 1024
        road_class_index = 0
    else:
        input_h, input_w = 512, 896
        road_class_index = 1
    
    images = find_images(dataset)[:num_samples]
    if not images:
        return {"iou": 0.0, "pixel_acc": 0.0, "count": 0, "latencies": [], "mean_latency_ms": 0.0, "fps": 0.0}
    
    road_value = dataset["road_value"]
    ignore_value = dataset.get("ignore_value", 255)
    labels_dir = resolve(dataset["labels_dir"])
    
    ious = []
    pixel_accs = []
    latencies = []
    
    print(f"    测试 {len(images)} 张图片...")
    
    for img_path, img_name in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        gt_mask = load_gt_mask(labels_dir, dataset["label_suffix"], img_name, dataset["image_suffix"])
        if gt_mask is None:
            continue
        
        t0 = time.perf_counter()
        if model_name == "PIDNet-S":
            logits_chw = run_openvino_pidnet_forward(compiled, input_name, image, input_h, input_w)
        else:
            logits_chw = run_openvino_adas_forward(compiled, input_name, image, input_h, input_w)
        latency = (time.perf_counter() - t0) * 1000.0
        
        label_small = logits_chw_to_label_map(logits_chw)
        pred_mask = cv2.resize(label_small.astype(np.uint8), 
                               (gt_mask.shape[1], gt_mask.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        pred_mask = (pred_mask == road_class_index).astype(np.uint8)
        
        iou = compute_iou(pred_mask, gt_mask, road_value, ignore_value)
        pixel_acc = compute_pixel_acc(pred_mask, gt_mask, road_value, ignore_value)
        
        ious.append(iou)
        pixel_accs.append(pixel_acc)
        latencies.append(latency)
    
    if not ious:
        return {"iou": 0.0, "pixel_acc": 0.0, "count": 0, "latencies": [], "mean_latency_ms": 0.0, "fps": 0.0}
    
    return {
        "iou": round(statistics.mean(ious) * 100, 2),
        "pixel_acc": round(statistics.mean(pixel_accs) * 100, 2),
        "count": len(ious),
        "latencies": latencies,
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "fps": round(1000.0 / statistics.mean(latencies), 1) if latencies else 0.0
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PIDNet-S vs road-adas 语义分割模型对比")
    parser.add_argument("--datasets", action="append", default=None, help="指定数据集（如 bdd100k）")
    parser.add_argument("--num_samples", type=int, default=None, help="每个数据集评测样本数")
    parser.add_argument("--device", type=str, default="CPU", help="测试设备（CPU/GPU/NPU）")
    parser.add_argument("--vis_count", type=int, default=5, help="生成可视化对比图的数量")
    args = parser.parse_args()
    
    cfg = load_eval_config()
    
    models = [
        {
            "name": "road-adas",
            "xml": resolve(cfg["models"]["segmentation"]["road_adas"]),
            "input_hw": (512, 896),
            "road_class_index": 1
        },
        {
            "name": "PIDNet-S",
            "xml": resolve(cfg["models"]["segmentation"]["pidnet_fp32"]),
            "input_hw": (1024, 1024),
            "road_class_index": 0
        }
    ]
    
    datasets = cfg.get("segmentation", {}).get("datasets", [])
    if args.datasets:
        datasets = [d for d in datasets if d["name"] in args.datasets]
    
    num_samples = args.num_samples or cfg.get("segmentation", {}).get("num_samples", 200)
    
    print("=" * 78)
    print("语义分割模型对比：PIDNet-S vs road-adas")
    print("=" * 78)
    print(f"测试设备: {args.device}")
    print(f"评测样本: {num_samples} 张/数据集")
    print(f"可视化数量: {args.vis_count} 张")
    print("-" * 78)
    
    # 速度基准测试
    print("\n【速度基准测试】")
    print("-" * 78)
    print(f"{'模型':<12} {'输入尺寸':<12} {'均值(ms)':>10} {'中位(ms)':>10} {'p95(ms)':>10} {'FPS':>8} {'体积(MB)':>10}")
    print("-" * 78)
    
    speed_results = {}
    for m in models:
        if not m["xml"].exists():
            print(f"{m['name']:<12} {'-':<12} {'模型不存在':>50}")
            continue
        
        result = bench_model(m["xml"], args.device, 20, 200, m["input_hw"][0], m["input_hw"][1], m["name"])
        if result["success"]:
            print(f"{m['name']:<12} {f'{m['input_hw'][0]}x{m['input_hw'][1]}':<12} "
                  f"{result['mean_ms']:>10} {result['median_ms']:>10} {result['p95_ms']:>10} "
                  f"{result['fps']:>8} {result['size_mb']:>10}")
            speed_results[m["name"]] = result
    
    # 精度评测（包含可视化）
    print("\n【精度评测（road IoU）】")
    accuracy_results = {}
    
    out_dir = resolve(cfg.get("output", {}).get("accuracy_dir", "runs/accuracy_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ds in datasets:
        print(f"\n数据集: {ds['name']}")
        print("-" * 78)
        print(f"{'模型':<12} {'图片数':>8} {'road IoU(%)':>14} {'像素准确率(%)':>16} {'延迟(ms)':>10} {'FPS':>8}")
        print("-" * 78)
        
        accuracy_results[ds["name"]] = {}
        
        # 编译所有模型
        compiled_models = {}
        for m in models:
            if m["xml"].exists():
                core = ov.Core()
                compiled_models[m["name"]] = {
                    "compiled": core.compile_model(str(m["xml"]), "CPU"),
                    "input_name": core.compile_model(str(m["xml"]), "CPU").input(0).get_any_name(),
                    "input_hw": m["input_hw"],
                    "road_class_index": m["road_class_index"]
                }
        
        # 获取图像列表
        images = find_images(ds)[:num_samples]
        if not images:
            for m in models:
                accuracy_results[ds["name"]][m["name"]] = {
                    "iou": 0.0, "pixel_acc": 0.0, "count": 0, 
                    "latencies": [], "mean_latency_ms": 0.0, "fps": 0.0
                }
                print(f"{m['name']:<12} {0:>8} {0.0:>14} {0.0:>16} {0.0:>10} {0.0:>8}")
            continue
        
        road_value = ds["road_value"]
        ignore_value = ds.get("ignore_value", 255)
        labels_dir = resolve(ds["labels_dir"])
        
        # 为每个模型收集结果
        model_results = {m["name"]: {"ious": [], "pixel_accs": [], "latencies": [], "pred_masks": {}} for m in models}
        vis_count = min(args.vis_count, len(images))
        vis_index = 0
        
        print(f"    测试 {len(images)} 张图片...")
        
        for img_path, img_name in images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            gt_mask = load_gt_mask(labels_dir, ds["label_suffix"], img_name, ds["image_suffix"])
            if gt_mask is None:
                continue
            
            # 存储当前图像的所有预测结果用于可视化
            current_pred_masks = []
            
            for m in models:
                if m["name"] not in compiled_models:
                    continue
                
                comp = compiled_models[m["name"]]
                input_name = comp["input_name"]
                input_h, input_w = comp["input_hw"]
                road_class_index = comp["road_class_index"]
                
                t0 = time.perf_counter()
                if m["name"] == "PIDNet-S":
                    logits_chw = run_openvino_pidnet_forward(comp["compiled"], input_name, image, input_h, input_w)
                else:
                    logits_chw = run_openvino_adas_forward(comp["compiled"], input_name, image, input_h, input_w)
                latency = (time.perf_counter() - t0) * 1000.0
                
                label_small = logits_chw_to_label_map(logits_chw)
                pred_mask = cv2.resize(label_small.astype(np.uint8), 
                                       (gt_mask.shape[1], gt_mask.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
                pred_mask = (pred_mask == road_class_index).astype(np.uint8)
                
                # 保存预测mask用于可视化
                model_results[m["name"]]["pred_masks"][img_name] = pred_mask
                
                iou = compute_iou(pred_mask, gt_mask, road_value, ignore_value)
                pixel_acc = compute_pixel_acc(pred_mask, gt_mask, road_value, ignore_value)
                
                model_results[m["name"]]["ious"].append(iou)
                model_results[m["name"]]["pixel_accs"].append(pixel_acc)
                model_results[m["name"]]["latencies"].append(latency)
            
            # 生成可视化对比图
            if vis_index < vis_count:
                pred_masks_for_vis = [
                    ("road-adas", model_results["road-adas"]["pred_masks"][img_name]),
                    ("PIDNet-S", model_results["PIDNet-S"]["pred_masks"][img_name])
                ]
                vis_path = out_dir / f"comparison_{ds['name']}_{vis_index + 1}.jpg"
                visualize_comparison(image, gt_mask, pred_masks_for_vis, road_value, vis_path)
                vis_index += 1
        
        # 汇总结果
        for m in models:
            res = model_results[m["name"]]
            if res["ious"]:
                mean_iou = round(statistics.mean(res["ious"]) * 100, 2)
                mean_pixel_acc = round(statistics.mean(res["pixel_accs"]) * 100, 2)
                mean_latency = round(statistics.mean(res["latencies"]), 2)
                fps = round(1000.0 / mean_latency, 1) if mean_latency > 0 else 0.0
                count = len(res["ious"])
            else:
                mean_iou = 0.0
                mean_pixel_acc = 0.0
                mean_latency = 0.0
                fps = 0.0
                count = 0
            
            accuracy_results[ds["name"]][m["name"]] = {
                "iou": mean_iou,
                "pixel_acc": mean_pixel_acc,
                "count": count,
                "latencies": res["latencies"],
                "mean_latency_ms": mean_latency,
                "fps": fps
            }
            
            print(f"{m['name']:<12} {count:>8} {mean_iou:>14} "
                  f"{mean_pixel_acc:>16} {mean_latency:>10} {fps:>8}")
        
        if vis_count > 0:
            print(f"    可视化对比图已保存至 {out_dir}")
    
    # 汇总对比
    print("\n" + "=" * 78)
    print("【综合对比】")
    print("=" * 78)
    
    if speed_results:
        print("\n速度对比:")
        print(f"{'模型':<12} {'延迟降低':>12} {'FPS提升':>12} {'体积(MB)':>10}")
        print("-" * 50)
        road_adas_speed = speed_results.get("road-adas", {})
        pidnet_speed = speed_results.get("PIDNet-S", {})
        
        if road_adas_speed and pidnet_speed:
            lat_diff = ((pidnet_speed["mean_ms"] - road_adas_speed["mean_ms"]) / road_adas_speed["mean_ms"]) * 100
            fps_diff = ((pidnet_speed["fps"] - road_adas_speed["fps"]) / road_adas_speed["fps"]) * 100
            
            print(f"road-adas  : {'基准':>12} {'基准':>12} {road_adas_speed['size_mb']:>10}")
            print(f"PIDNet-S   : {lat_diff:>11.1f}% {'↑' if lat_diff < 0 else '↓'} "
                  f"{fps_diff:>11.1f}% {'↑' if fps_diff > 0 else '↓'} {pidnet_speed['size_mb']:>10}")
    
    if accuracy_results:
        print("\n精度对比 (road IoU):")
        for ds_name, ds_results in accuracy_results.items():
            road_adas_acc = ds_results.get("road-adas", {})
            pidnet_acc = ds_results.get("PIDNet-S", {})
            
            if road_adas_acc and pidnet_acc:
                iou_diff = pidnet_acc["iou"] - road_adas_acc["iou"]
                print(f"{ds_name}: road-adas={road_adas_acc['iou']}%  PIDNet-S={pidnet_acc['iou']}%  "
                      f"({'+' if iou_diff > 0 else ''}{iou_diff:.2f}%)")
    
    # 保存结果
    out_dir = resolve(cfg.get("output", {}).get("accuracy_dir", "runs/accuracy_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "num_samples": num_samples,
        "speed": speed_results,
        "accuracy": accuracy_results
    }
    
    out_path = out_dir / "segmentation_comparison.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 生成 CSV
    csv_path = out_dir / "segmentation_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["数据集", "模型", "图片数", "road IoU(%)", "像素准确率(%)", 
                         "平均延迟(ms)", "FPS", "模型体积(MB)"])
        
        for ds_name, ds_results in accuracy_results.items():
            for m_name, m_results in ds_results.items():
                speed_info = speed_results.get(m_name, {})
                writer.writerow([
                    ds_name, m_name, m_results.get("count", 0),
                    m_results.get("iou", 0), m_results.get("pixel_acc", 0),
                    m_results.get("mean_latency_ms", 0), m_results.get("fps", 0),
                    speed_info.get("size_mb", 0)
                ])
    
    print(f"\n结果已保存至: {out_dir}")
    print(f"  - segmentation_comparison.json")
    print(f"  - segmentation_comparison.csv")


if __name__ == "__main__":
    main()
