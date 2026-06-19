"""
YOLO26 目标检测模型多设备性能对比脚本
对比 OpenVINO 在 CPU、GPU、NPU、AUTO 四种设备上的推理性能
- FP32 精度
- 只测速度指标（延迟、FPS、稳定性）
- 使用完整检测测试集（BDD 1500 + IDD 500 + DAWN 300）
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm


# 统一目标类别 (COCO 类名)
UNIFIED_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "rider"}

# BDD100K 类别映射
BDD_TO_UNIFIED = {
    "person": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
    "bike": "bicycle",
}

# IDD 类别映射
IDD_TO_UNIFIED = {
    "person": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "autorickshaw": "motorcycle",
}

# DAWN 类别映射
DAWN_TO_UNIFIED = {
    "Person": "person",
    "Car": "car",
    "Truck": "truck",
    "Bus": "bus",
    "Motorcycle": "motorcycle",
    "Bicycle": "bicycle",
}


class DatasetLoader:
    """从 det_manifest.csv 加载完整检测测试集"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.manifest_path = self.base_dir / "test_plan" / "det_manifest.csv"
    
    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载所有数据集样本"""
        result = {"bdd": [], "idd": [], "dawn": []}
        
        if not self.manifest_path.exists():
            print(f"警告: manifest 文件不存在: {self.manifest_path}")
            return result
        
        with open(self.manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row["dataset"]
                img_path = self.base_dir / row["image"]
                
                if not img_path.exists():
                    continue
                
                # 加载标注
                annotations = self._load_annotations(dataset, row["source"], row["image"])
                
                if annotations:
                    result[dataset].append({
                        "image_path": str(img_path),
                        "annotations": annotations,
                        "dataset": dataset,
                        "timeofday": row.get("timeofday", "unknown"),
                        "weather": row.get("weather", "unknown"),
                    })
        
        return result
    
    def _load_annotations(self, dataset: str, source: str, image_path: str) -> List[Dict[str, Any]]:
        """加载单个样本的标注"""
        annotations = []
        img_name = Path(image_path).name
        
        try:
            if dataset == "bdd":
                # BDD: 每张图片有单独的 JSON 文件
                source_path = self.base_dir / source
                with open(source_path, "r") as f:
                    data = json.load(f)
                for frame in data.get("frames", []):
                    for obj in frame.get("objects", []):
                        category = obj.get("category", "")
                        unified_cat = BDD_TO_UNIFIED.get(category)
                        if unified_cat and unified_cat in UNIFIED_CLASSES:
                            box = obj.get("box2d", {})
                            annotations.append({
                                "category": unified_cat,
                                "bbox": [box.get("x1", 0), box.get("y1", 0), 
                                         box.get("x2", 0), box.get("y2", 0)]
                            })
            elif dataset == "idd":
                # IDD: 所有标注在统一的 labels_val.json 中
                labels_file = self.base_dir / "det" / "idd" / "labels_val.json"
                with open(labels_file, "r") as f:
                    all_data = json.load(f)
                img_data = all_data.get(img_name, {})
                for obj in img_data.get("objects", []):
                    category = obj.get("category", "")
                    unified_cat = IDD_TO_UNIFIED.get(category)
                    if unified_cat and unified_cat in UNIFIED_CLASSES:
                        bbox = obj.get("bbox", [0, 0, 0, 0])
                        x1, y1, w, h = bbox
                        annotations.append({
                            "category": unified_cat,
                            "bbox": [x1, y1, x1 + w, y1 + h]
                        })
            elif dataset == "dawn":
                # DAWN: 所有标注在统一的 labels.json 中
                labels_file = self.base_dir / "det" / "dawn" / "labels.json"
                with open(labels_file, "r") as f:
                    all_data = json.load(f)
                img_data = all_data.get(img_name, {})
                for obj in img_data.get("objects", []):
                    category = obj.get("category", "")
                    unified_cat = DAWN_TO_UNIFIED.get(category)
                    if unified_cat and unified_cat in UNIFIED_CLASSES:
                        bbox = obj.get("bbox", [0, 0, 0, 0])
                        x1, y1, w, h = bbox
                        annotations.append({
                            "category": unified_cat,
                            "bbox": [x1, y1, x1 + w, y1 + h]
                        })
        except Exception:
            pass
        
        return annotations


class YOLO26Detector:
    """YOLO26 OpenVINO 检测器"""
    
    def __init__(self, model_path: str, device: str, conf_thres: float = 0.25):
        self.model_path = Path(model_path)
        self.device = device.upper()
        self.conf_thres = conf_thres
        self.image_size = 640
        
        # 初始化 OpenVINO
        self.core = ov.Core()
        
        # 检查设备可用性
        available_devices = self.core.available_devices
        if self.device not in available_devices and self.device != "AUTO":
            raise ValueError(f"设备 {self.device} 不可用，可用设备: {available_devices}")
        
        # 编译模型
        print(f"加载模型到 {self.device}...")
        self.model = self.core.compile_model(str(self.model_path), self.device)
        self.infer_request = self.model.create_infer_request()
        
        # 获取输入输出信息
        self.input_tensor = self.model.input(0)
        self.output_tensor = self.model.output(0)
        self.input_shape = self.input_tensor.shape
        self.input_h, self.input_w = self.input_shape[2], self.input_shape[3]
        
        # 获取类别名称（YOLO COCO 80类）
        self.names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "traffic sign", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
            49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
            59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
            69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
            74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
            79: "toothbrush"
        }
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        img_h, img_w = image.shape[:2]
        
        # 计算缩放比例和填充
        scale = min(self.input_w / img_w, self.input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建填充图像
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # 转换为模型输入格式
        input_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, 0)
        
        return input_tensor
    
    def infer(self, image: np.ndarray) -> float:
        """执行推理，返回延迟（毫秒）"""
        input_data = self._preprocess(image)
        
        start_time = time.perf_counter()
        self.infer_request.infer([input_data])
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return elapsed


def benchmark_device(detector: YOLO26Detector, samples: List[Dict[str, Any]], 
                     warmup: int = 5) -> Dict[str, Any]:
    """对单个设备进行性能测试"""
    
    if not samples:
        return {"error": "无样本数据"}
    
    # 预热
    print(f"  预热 {warmup} 次...")
    warmup_samples = samples[:warmup]
    for sample in warmup_samples:
        img = cv2.imread(sample["image_path"])
        if img is not None:
            detector.infer(img)
    
    # 正式测试
    latencies = []
    successful = 0
    
    print(f"  测试 {len(samples)} 张图片...")
    for sample in tqdm(samples, desc=f"Benchmarking {detector.device}"):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue
        
        latency = detector.infer(img)
        latencies.append(latency)
        successful += 1
    
    if not latencies:
        return {"error": "无有效测试结果"}
    
    # 计算统计指标
    latencies = np.array(latencies)
    results = {
        "device": detector.device,
        "sample_count": successful,
        "avg_latency_ms": float(np.mean(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "std_latency_ms": float(np.std(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
        "latency_range_ms": float(np.max(latencies) - np.min(latencies)),
    }
    
    return results


def print_results_table(results: Dict[str, Dict[str, Any]]):
    """打印结果表格"""
    print("\n" + "=" * 80)
    print("YOLO26 多设备性能对比结果 (OpenVINO FP32)")
    print("=" * 80)
    
    # 速度对比表
    print("\n速度对比:")
    print("-" * 80)
    print(f"| {'设备':<8} | {'样本数':>6} | {'平均延迟(ms)':>12} | {'P95延迟(ms)':>10} | {'FPS':>8} | {'标准差(ms)':>10} |")
    print("-" * 80)
    
    for device, result in results.items():
        if "error" in result:
            print(f"| {device:<8} | ERROR: {result['error']}")
        else:
            print(f"| {device:<8} | {result['sample_count']:>6} | "
                  f"{result['avg_latency_ms']:>12.2f} | {result['p95_latency_ms']:>10.2f} | "
                  f"{result['fps']:>8.2f} | {result['std_latency_ms']:>10.2f} |")
    print("-" * 80)
    
    # 相对 CPU 的加速比
    if "CPU" in results and "error" not in results["CPU"]:
        cpu_latency = results["CPU"]["avg_latency_ms"]
        cpu_fps = results["CPU"]["fps"]
        
        print("\n相对于 CPU 的性能对比:")
        print("-" * 80)
        print(f"| {'设备':<8} | {'延迟比':>10} | {'FPS提升':>10} | {'稳定性':>10} |")
        print("-" * 80)
        
        for device, result in results.items():
            if device == "CPU" or "error" in result:
                continue
            latency_ratio = result["avg_latency_ms"] / cpu_latency
            fps_ratio = result["fps"] / cpu_fps
            stability = "更稳定" if result["std_latency_ms"] < results["CPU"]["std_latency_ms"] else "较不稳定"
            print(f"| {device:<8} | {latency_ratio:>10.2f}x | {fps_ratio:>10.2f}x | {stability:>10} |")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="YOLO26 多设备性能对比 (OpenVINO)")
    parser.add_argument("--devices", type=str, default="CPU,GPU,NPU,AUTO",
                        help="要测试的设备列表，逗号分隔")
    parser.add_argument("--warmup", type=int, default=5, help="预热次数")
    parser.add_argument("--output_dir", type=str, default="runs/device_comparison",
                        help="结果输出目录")
    args = parser.parse_args()
    
    # 路径设置
    root_dir = Path(__file__).resolve().parents[2]
    model_path = root_dir / "models" / "yolo26n_openvino_model" / "yolo26n.xml"
    dataset_dir = root_dir / "datasets"
    output_dir = root_dir / args.output_dir
    
    # 检查模型文件
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 检查 OpenVINO 可用设备
    core = ov.Core()
    available_devices = core.available_devices
    print(f"OpenVINO 可用设备: {available_devices}")
    
    # 解析要测试的设备
    devices_to_test = [d.strip().upper() for d in args.devices.split(",")]
    
    # 过滤不可用的设备
    valid_devices = []
    for device in devices_to_test:
        if device == "AUTO" or device in available_devices:
            valid_devices.append(device)
        else:
            print(f"警告: 设备 {device} 不可用，跳过")
    
    if not valid_devices:
        print("错误: 没有可用的测试设备")
        return
    
    print(f"\n将测试设备: {valid_devices}")
    
    # 加载完整测试集
    print("\n加载检测测试集...")
    loader = DatasetLoader(str(dataset_dir))
    all_samples = loader.load_all()
    
    # 合并所有数据集样本
    all_samples_list = []
    for dataset_name, samples in all_samples.items():
        all_samples_list.extend(samples)
        print(f"  {dataset_name}: {len(samples)} 张")
    
    print(f"总计: {len(all_samples_list)} 张测试图片")
    
    if not all_samples_list:
        print("错误: 无测试样本")
        return
    
    # 对每个设备进行测试
    results = {}
    
    for device in valid_devices:
        print(f"\n{'='*60}")
        print(f"测试设备: {device}")
        print(f"{'='*60}")
        
        try:
            detector = YOLO26Detector(str(model_path), device)
            result = benchmark_device(detector, all_samples_list, args.warmup)
            results[device] = result
        except Exception as e:
            print(f"错误: {e}")
            results[device] = {"error": str(e)}
    
    # 打印结果
    print_results_table(results)
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 格式
    with open(output_dir / "device_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # CSV 格式
    with open(output_dir / "device_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "sample_count", "avg_latency_ms", "min_latency_ms", 
                        "max_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                        "std_latency_ms", "fps", "latency_range_ms"])
        for device, result in results.items():
            if "error" not in result:
                writer.writerow([
                    device, result["sample_count"], result["avg_latency_ms"],
                    result["min_latency_ms"], result["max_latency_ms"],
                    result["p50_latency_ms"], result["p95_latency_ms"],
                    result["p99_latency_ms"], result["std_latency_ms"],
                    result["fps"], result["latency_range_ms"]
                ])
    
    print(f"\n结果已保存至: {output_dir}")
    print(f"  - device_comparison.json")
    print(f"  - device_comparison.csv")


if __name__ == "__main__":
    main()