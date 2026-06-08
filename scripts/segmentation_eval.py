"""
语义分割模型测试脚本
测试 OpenVINO road-segmentation-adas-0001 模型

路径全部读自 configs/vision/datasets.yaml（可被 datasets.local.yaml 覆盖），
改路径不用动这个文件。
"""
from pathlib import Path
import sys
import time
import csv
import statistics
import json
import numpy as np
from openvino import Core
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_paths import load_path_config, resolve

_PATHS = load_path_config()
OUTPUT_DIR = resolve(_PATHS.get("eval_output_dir", "runs/segmentation_eval"))
SEG_MODEL_XML = resolve(
    _PATHS.get("model_xml", "models/openvino/road-segmentation-adas-0001/road-segmentation-adas-0001.xml")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image_path: Path, target_height: int, target_width: int) -> tuple:
    """预处理图片"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"警告: 无法读取图片 {image_path}")
        return None, None, None
    
    original_h, original_w = img.shape[:2]
    
    if original_h <= 0 or original_w <= 0:
        print(f"警告: 图片尺寸无效 {image_path}")
        return None, None, None
    
    # 调整大小
    img_resized = cv2.resize(img, (target_width, target_height))
    
    # 转换为 RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 归一化并转换为 NCHW
    img_input = img_rgb.transpose(2, 0, 1)  # HWC -> CHW
    img_input = np.expand_dims(img_input, 0)  # CHW -> NCHW
    
    return img_input, (original_h, original_w), img


def postprocess_mask(output: np.ndarray, original_size: tuple) -> np.ndarray:
    """后处理分割掩码"""
    # output shape: [1, 1, H, W] 或 [1, H, W] 或 [H, W]
    mask = output.squeeze()
    
    # 确保是 2D
    if mask.ndim > 2:
        mask = mask[0]
    
    # 调整回原始大小
    if original_size[0] > 0 and original_size[1] > 0:
        mask_resized = cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]))
    else:
        mask_resized = mask.astype(np.uint8)
    
    return mask_resized


def get_image_paths_from_split(split_file: Path, dataset_dir: Path) -> list:
    """从划分文件获取图片路径"""
    if not split_file.exists():
        return []
    
    with open(split_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    
    images = []
    for name in names:
        for ext in [".jpg", ".png", ".jpeg"]:
            img_path = dataset_dir / name
            if img_path.exists():
                images.append(img_path)
                break
            matches = list(dataset_dir.rglob(name))
            if matches:
                images.append(matches[0])
                break
    return images


def test_segmentation_dataset(model, input_blob, output_blob, image_paths: list, dataset_name: str) -> dict:
    """测试语义分割模型"""
    print(f"\n{'='*60}")
    print(f"测试数据集: {dataset_name}")
    print(f"图片数量: {len(image_paths)}")
    print(f"{'='*60}")
    
    if len(image_paths) == 0:
        print(f"警告: {dataset_name} 没有找到图片")
        return {}
    
    # 获取输入尺寸
    input_shape = model.input(input_blob).shape
    _, _, input_h, input_w = input_shape
    print(f"模型输入尺寸: {input_w}x{input_h}")
    
    # 热身
    print("热身推理...")
    for img_path in image_paths[:3]:
        img_input, _, _ = preprocess_image(img_path, input_h, input_w)
        if img_input is not None:
            model.infer_new_request({input_blob: img_input})
    
    # 正式测试
    latencies = []
    fps_values = []
    drivable_ratios = []
    
    csv_path = OUTPUT_DIR / f"{dataset_name}_segmentation.csv"
    
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "latency_ms", "fps", "drivable_ratio"])
        
        for idx, img_path in enumerate(image_paths, 1):
            img_input, original_size, original_img = preprocess_image(img_path, input_h, input_w)
            
            if img_input is None:
                continue
            
            start = time.perf_counter()
            result = model.infer_new_request({input_blob: img_input})
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0
            
            # 计算可行驶区域比例
            output = result[output_blob]
            mask = postprocess_mask(output, original_size)
            drivable_ratio = np.sum(mask > 0) / mask.size
            
            latencies.append(latency_ms)
            fps_values.append(fps)
            drivable_ratios.append(drivable_ratio)
            
            writer.writerow([img_path.name, round(latency_ms, 2), round(fps, 2), round(drivable_ratio, 4)])
            
            if idx % 50 == 0:
                print(f"  已测试 {idx}/{len(image_paths)} 张")
    
    # 统计结果
    stats = {
        "dataset": dataset_name,
        "task": "segmentation",
        "image_count": len(latencies),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "mean_fps": round(statistics.mean(fps_values), 2),
        "min_fps": round(min(fps_values), 2),
        "max_fps": round(max(fps_values), 2),
        "mean_drivable_ratio": round(statistics.mean(drivable_ratios), 4),
        "csv_file": str(csv_path.name)
    }
    
    print(f"\n结果统计:")
    print(f"  平均延迟: {stats['mean_latency_ms']} ms")
    print(f"  平均 FPS: {stats['mean_fps']}")
    print(f"  平均可行驶区域比例: {stats['mean_drivable_ratio']}")
    
    return stats


def main():
    print("="*60)
    print("语义分割模型测试")
    print("="*60)
    
    # 检查模型
    if not SEG_MODEL_XML.exists():
        print(f"错误: 未找到分割模型 {SEG_MODEL_XML}")
        return
    
    print(f"加载分割模型: {SEG_MODEL_XML}")
    
    # 初始化 OpenVINO
    core = Core()
    model = core.compile_model(str(SEG_MODEL_XML), "CPU")
    
    # 获取输入输出名称
    input_blob = next(iter(model.inputs))
    output_blob = next(iter(model.outputs))
    print(f"输入: {input_blob.any_name}, 输出: {output_blob.any_name}")
    
    # 数据集配置（来自 configs/vision/datasets.yaml 的 eval_datasets）
    datasets = [
        {
            "name": ds["name"],
            "split_file": resolve(ds["split_file"]),
            "image_dir": resolve(ds["image_dir"]),
        }
        for ds in _PATHS.get("eval_datasets", [])
    ]
    if not datasets:
        print("错误：configs/vision/datasets.yaml 里没有 eval_datasets 配置")
        return
    
    all_stats = []
    
    for ds in datasets:
        if ds["image_dir"].exists():
            if ds["name"] == "bdd100k":
                images = sorted(list(ds["image_dir"].glob("*.jpg")))
            else:
                images = get_image_paths_from_split(ds["split_file"], ds["image_dir"])
        else:
            images = []
        
        if len(images) > 0:
            stats = test_segmentation_dataset(model, input_blob.any_name, output_blob.any_name, images, ds["name"])
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
            "mean_drivable_ratio"
        ])
        for stats in all_stats:
            writer.writerow([
                stats["dataset"], stats["task"], stats["image_count"],
                stats["mean_latency_ms"], stats["median_latency_ms"],
                stats["min_latency_ms"], stats["max_latency_ms"],
                stats["mean_fps"], stats["min_fps"], stats["max_fps"],
                stats["mean_drivable_ratio"]
            ])
    
    # 打印最终汇总
    print("\n" + "="*60)
    print("测试完成汇总")
    print("="*60)
    print(f"{'数据集':<15} {'图片数':>8} {'平均延迟(ms)':>12} {'平均FPS':>10} {'可行驶比例':>12}")
    print("-"*60)
    for stats in all_stats:
        print(f"{stats['dataset']:<15} {stats['image_count']:>8} {stats['mean_latency_ms']:>12} {stats['mean_fps']:>10} {stats['mean_drivable_ratio']:>12}")
    
    print(f"\n结果保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
