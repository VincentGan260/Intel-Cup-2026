"""
语义分割结果可视化脚本
生成原始图片、标注mask和模型预测结果的对比图
添加时间统计功能，用于对比不同模型的速度差异
"""

import cv2
import numpy as np
import argparse
import time
from openvino import Core
from pathlib import Path

# 使用相对路径（基于脚本位置推断项目根目录）
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# 默认路径
DATA_DIR = ROOT / "datasets" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images" / "100k" / "val"
SEG_LABELS_DIR = DATA_DIR / "labels" / "bdd100k_drivable_maps" / "labels" / "val"
OUTPUT_DIR = ROOT / "runs" / "segmentation_visualization"

# 颜色映射（用于可视化）
COLOR_MAP = {
    0: [0, 0, 0],       # 黑色 - 背景
    1: [0, 255, 0],     # 绿色 - 可行驶区域
    2: [255, 255, 0],   # 黄色 - 谨慎区域
}

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="语义分割可视化脚本")
    parser.add_argument("--dataset", type=str, default=str(DATA_DIR), 
                        help="数据集根目录路径（默认: datasets/bdd100k）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="输出目录路径（默认: runs/segmentation_visualization）")
    parser.add_argument("--num-images", type=int, default=10, help="处理图片数量")
    args = parser.parse_args()
    
    # 使用命令行参数或默认值
    data_dir = Path(args.dataset)
    output_dir = Path(args.output)
    num_images = args.num_images
    
    # 更新路径
    images_dir = data_dir / "images" / "100k" / "val"
    seg_labels_dir = data_dir / "labels" / "bdd100k_drivable_maps" / "labels" / "val"
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型（使用相对路径）
    model_path = ROOT / "models" / "openvino" / "road-segmentation-adas-0001" / "road-segmentation-adas-0001.xml"
    core = Core()
    model = core.compile_model(str(model_path), "CPU")
    print(f"已加载模型: {model_path}")
    
    # 获取图片列表
    img_files = list(images_dir.glob("*.jpg"))[:num_images]
    print(f"找到 {len(img_files)} 张图片")
    
    # 时间统计变量
    total_inference_time = 0.0
    total_processing_time = 0.0
    inference_times = []
    
    for i, img_path in enumerate(img_files):
        # 记录处理开始时间
        process_start = time.time()
        
        # 读取图片
        img = cv2.imread(str(img_path))
        img_original = img.copy()
        
        # 预处理图片
        img_input = cv2.resize(img, (896, 512))
        input_tensor = np.expand_dims(img_input.transpose(2, 0, 1).astype(np.float32), 0)
        
        # 推理（记录推理时间）
        inference_start = time.time()
        result = model([input_tensor])[0]
        pred_mask = np.argmax(result, axis=1)[0]
        inference_end = time.time()
        
        # 计算推理时间
        inference_time = inference_end - inference_start
        inference_times.append(inference_time)
        total_inference_time += inference_time
        
        # 读取标注
        seg_file = seg_labels_dir / (img_path.stem + "_drivable_id.png")
        if seg_file.exists():
            gt_mask = cv2.imread(str(seg_file), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (896, 512), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask = np.zeros((512, 896), dtype=np.uint8)
            print(f"警告: 未找到标注文件 {seg_file}")
        
        # 创建可视化图片
        vis_img = create_visualization(img_input, gt_mask, pred_mask)
        
        # 保存结果
        output_path = output_dir / f"segmentation_result_{i:02d}.jpg"
        cv2.imwrite(str(output_path), vis_img)
        
        # 计算处理时间
        process_end = time.time()
        processing_time = process_end - process_start
        total_processing_time += processing_time
        
        print(f"已保存: {output_path} (推理时间: {inference_time*1000:.2f}ms)")
    
    # 计算统计信息
    if len(img_files) > 0:
        avg_inference_time = total_inference_time / len(img_files)
        avg_processing_time = total_processing_time / len(img_files)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        fps = len(img_files) / total_inference_time
        
        print(f"\n{'='*60}")
        print(f"语义分割时间统计（共 {len(img_files)} 张图片）")
        print(f"{'='*60}")
        print(f"平均推理时间: {avg_inference_time*1000:.2f} ms")
        print(f"最小推理时间: {min_inference_time*1000:.2f} ms")
        print(f"最大推理时间: {max_inference_time*1000:.2f} ms")
        print(f"平均处理时间: {avg_processing_time*1000:.2f} ms")
        print(f"推理FPS: {fps:.2f}")
        print(f"{'='*60}")
    
    print(f"\n可视化结果已保存至: {output_dir}")

def create_visualization(img, gt_mask, pred_mask):
    """创建可视化对比图"""
    # 创建标注mask可视化
    gt_vis = np.zeros_like(img)
    for label, color in COLOR_MAP.items():
        gt_vis[gt_mask == label] = color
    
    # 创建预测mask可视化
    pred_vis = np.zeros_like(img)
    # 模型输出标签1是道路
    pred_vis[pred_mask == 1] = [0, 255, 0]  # 绿色
    pred_vis[pred_mask == 2] = [0, 255, 255]  # 青色
    pred_vis[pred_mask == 3] = [255, 0, 0]  # 红色
    
    # 混合显示
    gt_overlay = cv2.addWeighted(img, 0.6, gt_vis, 0.4, 0)
    pred_overlay = cv2.addWeighted(img, 0.6, pred_vis, 0.4, 0)
    
    # 添加标题
    cv2.putText(img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(gt_overlay, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(pred_overlay, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 水平拼接
    result = np.hstack([img, gt_overlay, pred_overlay])
    
    return result

if __name__ == "__main__":
    main()