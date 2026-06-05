"""
综合模型测试脚本
测试目标检测和语义分割模型在多个数据集上的性能
"""
from pathlib import Path
from ultralytics import YOLO
import time
import csv
import psutil
import statistics
import json

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
SPLITS_DIR = ROOT / "data" / "splits"
DATASETS_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "runs" / "comprehensive_eval"

# 模型配置
DETECTION_MODEL = ROOT / "yolo26n.pt"
SEG_MODEL = ROOT / "models" / "openvino" / "road-segmentation-adas-0001"

# 测试参数
IMG_SIZE = 640
CONF_THRES = 0.25

# 类别定义
PEDESTRIAN_NAMES = {"person", "pedestrian"}
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle", "motorbike", "bike", "bicycle"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_image_paths_from_split(split_file: Path, dataset_dir: Path) -> list:
    """从划分文件获取图片路径"""
    if not split_file.exists():
        return []
    
    with open(split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    
    images = []
    for name in names:
        # 查找图片文件
        for ext in [".jpg", ".png", ".jpeg"]:
            # 尝试直接路径
            img_path = dataset_dir / name
            if img_path.exists():
                images.append(img_path)
                break
            # 尝试递归查找
            matches = list(dataset_dir.rglob(name))
            if matches:
                images.append(matches[0])
                break
    return images


def test_detection_dataset(model, image_paths: list, dataset_name: str) -> dict:
    """测试目标检测模型在单个数据集上的性能"""
    print(f"\n{'='*60}")
    print(f"测试数据集: {dataset_name}")
    print(f"图片数量: {len(image_paths)}")
    print(f"{'='*60}")
    
    if len(image_paths) == 0:
        print(f"警告: {dataset_name} 没有找到图片")
        return {}
    
    # 热身
    print("热身推理...")
    for img in image_paths[:3]:
        model.predict(source=str(img), imgsz=IMG_SIZE, conf=CONF_THRES, device="cpu", verbose=False)
    
    # 正式测试
    latencies = []
    fps_values = []
    vehicle_counts = []
    pedestrian_counts = []
    total_counts = []
    
    csv_path = OUTPUT_DIR / f"{dataset_name}_detection.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "latency_ms", "fps", "vehicle_count", "pedestrian_count", "total_count"])
        
        for idx, img_path in enumerate(image_paths, 1):
            start = time.perf_counter()
            results = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRES, device="cpu", verbose=False)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0
            
            # 统计目标
            r = results[0]
            vehicle_count = 0
            pedestrian_count = 0
            total_count = 0
            
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                total_count = len(cls_ids)
                for cls_id in cls_ids:
                    cls_name = str(model.names.get(cls_id, str(cls_id))).lower()
                    if cls_name in VEHICLE_NAMES:
                        vehicle_count += 1
                    if cls_name in PEDESTRIAN_NAMES:
                        pedestrian_count += 1
            
            latencies.append(latency_ms)
            fps_values.append(fps)
            vehicle_counts.append(vehicle_count)
            pedestrian_counts.append(pedestrian_count)
            total_counts.append(total_count)
            
            writer.writerow([img_path.name, round(latency_ms, 2), round(fps, 2), vehicle_count, pedestrian_count, total_count])
            
            if idx % 50 == 0:
                print(f"  已测试 {idx}/{len(image_paths)} 张")
    
    # 统计结果
    stats = {
        "dataset": dataset_name,
        "task": "detection",
        "image_count": len(image_paths),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "mean_fps": round(statistics.mean(fps_values), 2),
        "min_fps": round(min(fps_values), 2),
        "max_fps": round(max(fps_values), 2),
        "mean_vehicle_count": round(statistics.mean(vehicle_counts), 2),
        "mean_pedestrian_count": round(statistics.mean(pedestrian_counts), 2),
        "mean_total_count": round(statistics.mean(total_counts), 2),
        "csv_file": str(csv_path.name)
    }
    
    print(f"\n结果统计:")
    print(f"  平均延迟: {stats['mean_latency_ms']} ms")
    print(f"  平均 FPS: {stats['mean_fps']}")
    print(f"  平均车辆数: {stats['mean_vehicle_count']}")
    print(f"  平均行人数: {stats['mean_pedestrian_count']}")
    
    return stats


def main():
    print("="*60)
    print("综合模型测试")
    print("="*60)
    
    # 检查模型
    if not DETECTION_MODEL.exists():
        print(f"错误: 未找到检测模型 {DETECTION_MODEL}")
        return
    
    print(f"加载检测模型: {DETECTION_MODEL}")
    model = YOLO(str(DETECTION_MODEL))
    print(f"模型类别: {model.names}")
    
    # 数据集配置
    datasets = [
        {
            "name": "bdd100k",
            "split_file": SPLITS_DIR / "bdd_selected.txt",
            "image_dir": DATASETS_DIR / "bdd100k_subset_500" / "images" / "val"
        },
        {
            "name": "cityscapes",
            "split_file": SPLITS_DIR / "cityscapes_selected.txt",
            "image_dir": DATASETS_DIR / "cityscapes"
        },
        {
            "name": "acdc",
            "split_file": SPLITS_DIR / "acdc_selected.txt",
            "image_dir": DATASETS_DIR / "acdc"
        },
        {
            "name": "idd",
            "split_file": SPLITS_DIR / "idd_lite_selected.txt",
            "image_dir": DATASETS_DIR / "idd"
        }
    ]
    
    all_stats = []
    
    for ds in datasets:
        # 获取图片路径
        if ds["image_dir"].exists():
            if ds["name"] == "bdd100k":
                # BDD100K 直接从目录获取
                images = sorted(list(ds["image_dir"].glob("*.jpg")))
            else:
                # 其他数据集从划分文件获取
                images = get_image_paths_from_split(ds["split_file"], ds["image_dir"])
        else:
            images = []
        
        if len(images) > 0:
            stats = test_detection_dataset(model, images, ds["name"])
            if stats:
                all_stats.append(stats)
    
    # 保存汇总结果
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    # 生成汇总 CSV
    summary_csv_path = OUTPUT_DIR / "summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "task", "image_count", "mean_latency_ms", "median_latency_ms",
            "min_latency_ms", "max_latency_ms", "mean_fps", "min_fps", "max_fps",
            "mean_vehicle_count", "mean_pedestrian_count", "mean_total_count"
        ])
        for stats in all_stats:
            writer.writerow([
                stats["dataset"], stats["task"], stats["image_count"],
                stats["mean_latency_ms"], stats["median_latency_ms"],
                stats["min_latency_ms"], stats["max_latency_ms"],
                stats["mean_fps"], stats["min_fps"], stats["max_fps"],
                stats["mean_vehicle_count"], stats["mean_pedestrian_count"],
                stats["mean_total_count"]
            ])
    
    # 打印最终汇总
    print("\n" + "="*60)
    print("测试完成汇总")
    print("="*60)
    print(f"{'数据集':<15} {'图片数':>8} {'平均延迟(ms)':>12} {'平均FPS':>10} {'平均车辆':>10} {'平均行人':>10}")
    print("-"*60)
    for stats in all_stats:
        print(f"{stats['dataset']:<15} {stats['image_count']:>8} {stats['mean_latency_ms']:>12} {stats['mean_fps']:>10} {stats['mean_vehicle_count']:>10} {stats['mean_pedestrian_count']:>10}")
    
    print(f"\n结果保存至: {OUTPUT_DIR}")
    print(f"  - summary.json: 详细统计")
    print(f"  - summary.csv: 汇总表格")
    for stats in all_stats:
        print(f"  - {stats['csv_file']}: {stats['dataset']} 详细结果")


if __name__ == "__main__":
    main()
