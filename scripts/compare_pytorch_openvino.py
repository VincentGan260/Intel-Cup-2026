"""
PyTorch vs OpenVINO 速度对比脚本
对比 PyTorch 模型和 OpenVINO 优化模型的推理速度差异
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from ultralytics import YOLO
from openvino import Core

# 使用相对路径（基于脚本位置推断项目根目录）
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# 威胁类别索引（COCO 类别）
THREAT_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

def test_pytorch_model(model_path, images_dir, num_images=50, conf_threshold=0.15):
    """测试 PyTorch 模型性能"""
    print("\n" + "="*60)
    print("测试 PyTorch 模型")
    print("="*60)
    
    # 加载模型
    print(f"加载 PyTorch 模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 获取图片列表
    img_files = list(images_dir.glob("*.jpg"))[:num_images]
    print(f"测试图片数量: {len(img_files)}")
    
    # 时间统计变量
    inference_times = []
    total_detections = 0
    
    # 预热（避免首次推理影响结果）
    print("预热中...")
    for img_path in img_files[:3]:
        img = cv2.imread(str(img_path))
        _ = model.predict(source=img, imgsz=640, conf=conf_threshold, verbose=False)
    
    # 正式测试
    print("开始测试...")
    for i, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        
        # 推理
        start_time = time.time()
        results = model.predict(source=img, imgsz=640, conf=conf_threshold, verbose=False)
        end_time = time.time()
        
        # 记录时间
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # 统计检测数量
        for det in results[0].boxes:
            cls = int(det.cls[0].item())
            if cls in THREAT_CLASSES:
                total_detections += 1
        
        if (i + 1) % 10 == 0:
            print(f"已处理: {i+1}/{len(img_files)} 张图片")
    
    # 计算统计信息
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    fps = 1.0 / avg_time
    
    results = {
        'model_type': 'PyTorch',
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': fps,
        'total_detections': total_detections,
        'avg_detections': total_detections / len(img_files)
    }
    
    print(f"\n平均推理时间: {avg_time*1000:.2f} ms")
    print(f"最小推理时间: {min_time*1000:.2f} ms")
    print(f"最大推理时间: {max_time*1000:.2f} ms")
    print(f"标准差: {std_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"总检测数: {total_detections}")
    print(f"平均检测数: {total_detections/len(img_files):.2f}")
    
    return results

def test_openvino_model(model_path, images_dir, num_images=50, conf_threshold=0.15):
    """测试 OpenVINO 模型性能"""
    print("\n" + "="*60)
    print("测试 OpenVINO 模型")
    print("="*60)
    
    # 加载模型
    print(f"加载 OpenVINO 模型: {model_path}")
    core = Core()
    model = core.compile_model(str(model_path), "CPU")
    
    # 获取输入输出信息
    input_layer = model.input(0)
    output_layer = model.output(0)
    input_shape = input_layer.shape
    input_w, input_h = input_shape[3], input_shape[2]
    print(f"模型输入形状: {input_shape}")
    
    # 获取图片列表
    img_files = list(images_dir.glob("*.jpg"))[:num_images]
    print(f"测试图片数量: {len(img_files)}")
    
    # 时间统计变量
    inference_times = []
    total_detections = 0
    
    # 预热（避免首次推理影响结果）
    print("预热中...")
    for img_path in img_files[:3]:
        img = cv2.imread(str(img_path))
        img_resized = cv2.resize(img, (input_w, input_h))
        input_tensor = np.expand_dims(img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0, 0)
        _ = model([input_tensor])[output_layer]
    
    # 正式测试
    print("开始测试...")
    for i, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        original_h, original_w = img.shape[:2]
        
        # 预处理
        img_resized = cv2.resize(img, (input_w, input_h))
        input_tensor = np.expand_dims(img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0, 0)
        
        # 推理
        start_time = time.time()
        result = model([input_tensor])[output_layer]
        end_time = time.time()
        
        # 记录时间
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # 解析检测结果
        output = result[0]
        for row in output:
            x1, y1, x2, y2, conf, cls = row
            
            if conf < conf_threshold:
                continue
            
            cls = int(cls)
            if cls in THREAT_CLASSES:
                total_detections += 1
        
        if (i + 1) % 10 == 0:
            print(f"已处理: {i+1}/{len(img_files)} 张图片")
    
    # 计算统计信息
    avg_time = np.mean(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    std_time = np.std(inference_times)
    fps = 1.0 / avg_time
    
    results = {
        'model_type': 'OpenVINO',
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': fps,
        'total_detections': total_detections,
        'avg_detections': total_detections / len(img_files)
    }
    
    print(f"\n平均推理时间: {avg_time*1000:.2f} ms")
    print(f"最小推理时间: {min_time*1000:.2f} ms")
    print(f"最大推理时间: {max_time*1000:.2f} ms")
    print(f"标准差: {std_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"总检测数: {total_detections}")
    print(f"平均检测数: {total_detections/len(img_files):.2f}")
    
    return results

def compare_results(pytorch_results, openvino_results):
    """对比两个模型的结果"""
    print("\n" + "="*60)
    print("性能对比结果")
    print("="*60)
    
    # 计算加速比
    speedup = pytorch_results['avg_time_ms'] / openvino_results['avg_time_ms']
    fps_improvement = (openvino_results['fps'] - pytorch_results['fps']) / pytorch_results['fps'] * 100
    
    print(f"\n{'指标':<20} {'PyTorch':<15} {'OpenVINO':<15} {'提升':<15}")
    print("-" * 65)
    print(f"{'平均推理时间 (ms)':<20} {pytorch_results['avg_time_ms']:<15.2f} {openvino_results['avg_time_ms']:<15.2f} {speedup:<15.2f}x")
    print(f"{'最小推理时间 (ms)':<20} {pytorch_results['min_time_ms']:<15.2f} {openvino_results['min_time_ms']:<15.2f} -")
    print(f"{'最大推理时间 (ms)':<20} {pytorch_results['max_time_ms']:<15.2f} {openvino_results['max_time_ms']:<15.2f} -")
    print(f"{'标准差 (ms)':<20} {pytorch_results['std_time_ms']:<15.2f} {openvino_results['std_time_ms']:<15.2f} -")
    print(f"{'FPS':<20} {pytorch_results['fps']:<15.2f} {openvino_results['fps']:<15.2f} {fps_improvement:<15.2f}%")
    print(f"{'总检测数':<20} {pytorch_results['total_detections']:<15} {openvino_results['total_detections']:<15} -")
    print(f"{'平均检测数':<20} {pytorch_results['avg_detections']:<15.2f} {openvino_results['avg_detections']:<15.2f} -")
    
    print(f"\n{'='*60}")
    print(f"OpenVINO 加速比: {speedup:.2f}x")
    print(f"FPS 提升: {fps_improvement:.2f}%")
    print(f"{'='*60}")
    
    return {
        'speedup': speedup,
        'fps_improvement': fps_improvement
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PyTorch vs OpenVINO 速度对比")
    parser.add_argument("--dataset", type=str, default="datasets/bdd100k/images/100k/val", 
                        help="图片目录路径")
    parser.add_argument("--num-images", type=int, default=50, help="测试图片数量")
    parser.add_argument("--conf-threshold", type=float, default=0.15, help="置信度阈值")
    args = parser.parse_args()
    
    # 路径设置
    images_dir = Path(args.dataset)
    pytorch_model_path = ROOT / "yolo26n.pt"
    openvino_model_path = ROOT / "models" / "yolo26n_openvino_model" / "yolo26n.xml"
    
    print("="*60)
    print("PyTorch vs OpenVINO 速度对比测试")
    print("="*60)
    print(f"测试图片目录: {images_dir}")
    print(f"PyTorch 模型: {pytorch_model_path}")
    print(f"OpenVINO 模型: {openvino_model_path}")
    print(f"测试图片数量: {args.num_images}")
    print(f"置信度阈值: {args.conf_threshold}")
    
    # 测试 PyTorch 模型
    pytorch_results = test_pytorch_model(pytorch_model_path, images_dir, args.num_images, args.conf_threshold)
    
    # 测试 OpenVINO 模型
    openvino_results = test_openvino_model(openvino_model_path, images_dir, args.num_images, args.conf_threshold)
    
    # 对比结果
    comparison = compare_results(pytorch_results, openvino_results)
    
    # 保存结果到 JSON
    output_dir = ROOT / "runs" / "speed_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    results = {
        'pytorch': pytorch_results,
        'openvino': openvino_results,
        'comparison': comparison,
        'test_config': {
            'dataset': str(images_dir),
            'num_images': args.num_images,
            'conf_threshold': args.conf_threshold
        }
    }
    
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存至: {output_dir / 'comparison_results.json'}")

if __name__ == "__main__":
    main()