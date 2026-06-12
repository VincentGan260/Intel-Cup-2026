"""语义分割性能基准测试，分析各阶段耗时。"""
import time
import cv2
import yaml
import numpy as np
from pathlib import Path
from src.vision.segmentation.segmenter import build_segmenter_from_config
from src.vision.common.visualize import blend_binary_mask

PROJECT_ROOT = Path(__file__).resolve().parent


def benchmark():
    # 加载配置
    config_path = PROJECT_ROOT / "configs/vision/segmentation_openvino.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("📦 正在构建语义分割器...")
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    
    # 加载测试图片
    test_image_path = PROJECT_ROOT / "data/sample/segmentation_test_data/frankfurt_street.png"
    frame = cv2.imread(str(test_image_path))
    if frame is None:
        print("❌ 无法加载测试图片")
        return
    
    print(f"\n=== 性能基准测试 ===")
    print(f"图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
    print(f"模型输入: {config['model']['input_width']}x{config['model']['input_height']}")
    print(f"推理设备: {config['openvino']['device']}")
    
    # 预热
    print("\n🔄 预热中...")
    for _ in range(5):
        segmenter.infer(frame)
    
    # 正式测试
    print("\n📊 开始测试 (100次推理)...")
    num_iterations = 100
    total_time = 0
    infer_times = []
    
    for i in range(num_iterations):
        start_time = time.perf_counter()
        result = segmenter.infer(frame)
        elapsed = time.perf_counter() - start_time
        infer_times.append(elapsed)
        total_time += elapsed
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_iterations}")
    
    # 统计结果
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    min_time = min(infer_times)
    max_time = max(infer_times)
    
    print("\n=== 测试结果 ===")
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"平均帧率: {fps:.2f} FPS")
    print(f"最快推理: {min_time*1000:.2f} ms")
    print(f"最慢推理: {max_time*1000:.2f} ms")
    print(f"标准差: {np.std(infer_times)*1000:.2f} ms")
    
    # 检查分割结果
    if result.drivable_mask is not None:
        print(f"\n✅ 分割成功")
        print(f"   可行驶区域占比: {result.drivable_ratio:.1%}")


if __name__ == "__main__":
    benchmark()
