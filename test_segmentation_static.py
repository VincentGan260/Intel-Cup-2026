"""使用静态图片测试语义分割功能。"""
import cv2
import yaml
import numpy as np
from pathlib import Path
from src.vision.segmentation.segmenter import build_segmenter_from_config
from src.vision.common.visualize import blend_binary_mask

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    # 加载配置
    config_path = PROJECT_ROOT / "configs/vision/segmentation_openvino.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("📦 正在构建语义分割器...")
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    print("✅ 语义分割器构建成功")

    # 获取可视化参数
    alpha = config.get("visualization", {}).get("alpha", 0.45)
    road_class_index = config.get("model", {}).get("road_class_index", 1)
    colors_bgr = config.get("visualization", {}).get("colors_bgr", [[0, 0, 0], [0, 200, 0], [0, 128, 255], [200, 200, 200]])
    road_color = tuple(colors_bgr[road_class_index])

    # 使用示例图片进行测试
    test_image_path = PROJECT_ROOT / "data/sample/segmentation_test_data/frankfurt_street.png"
    
    if not test_image_path.exists():
        test_image_path = PROJECT_ROOT / "bus.jpg"
    
    print(f"📷 正在加载测试图片: {test_image_path}")
    frame = cv2.imread(str(test_image_path))
    
    if frame is None:
        print("❌ 无法加载测试图片")
        return
    
    print(f"🖼️  图像尺寸: {frame.shape[1]} x {frame.shape[0]}")
    
    # 执行分割推理
    print("🔄 正在执行语义分割推理...")
    import time
    start_time = time.time()
    result = segmenter.infer(frame)
    inference_time = time.time() - start_time
    print(f"⏱️  推理耗时: {inference_time:.3f} 秒")
    
    # 检查分割结果
    if result.drivable_mask is not None:
        print(f"✅ 分割成功")
        print(f"🔲 Mask 尺寸: {result.drivable_mask.shape}")
        
        if result.drivable_ratio is not None:
            print(f"🚗 可行驶区域占比: {result.drivable_ratio:.1%}")
        
        # 保存分割结果
        overlay = blend_binary_mask(frame, result.drivable_mask, road_color, alpha)
        output_path = PROJECT_ROOT / "runs/vision/segmentation_test/result.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)
        print(f"💾 分割结果已保存到: {output_path}")
    else:
        print("❌ 分割失败，未生成可行驶区域 mask")


if __name__ == "__main__":
    main()
