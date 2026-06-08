"""
语义分割可视化脚本
生成带分割掩码的样本图片，用于检查分割效果
"""
from pathlib import Path
from openvino import Core
import cv2
import numpy as np
import random

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
OUTPUT_DIR = ROOT / "runs" / "segmentation_vis"
MODEL_XML = ROOT / "models" / "openvino" / "road-segmentation-adas-0001" / "road-segmentation-adas-0001.xml"

# 输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 颜色映射（BGR 格式）
COLORS = {
    0: (0, 0, 0),        # 背景 - 黑色
    1: (0, 255, 0),      # 可行驶区域 - 绿色
    2: (255, 0, 0),      # 车道线 - 蓝色
}


def preprocess_image(image_path: Path, target_height: int, target_width: int) -> tuple:
    """预处理图片"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None
    
    original_h, original_w = img.shape[:2]
    img_resized = cv2.resize(img, (target_width, target_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, 0)
    
    return img_input.astype(np.float32), (original_h, original_w), img


def visualize_segmentation(original_img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """可视化分割结果"""
    # 创建彩色掩码
    colored_mask = np.zeros_like(original_img)
    
    for label, color in COLORS.items():
        colored_mask[mask == label] = color
    
    # 叠加到原图
    overlay = cv2.addWeighted(original_img, 1 - alpha, colored_mask, alpha, 0)
    
    # 添加边界线
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
    overlay[edges > 0] = (0, 255, 255)  # 黄色边界
    
    return overlay


def calculate_metrics(mask: np.ndarray) -> dict:
    """计算分割指标"""
    total_pixels = mask.size
    drivable_pixels = np.sum(mask == 1)
    lane_pixels = np.sum(mask == 2)
    background_pixels = np.sum(mask == 0)
    
    return {
        "drivable_ratio": drivable_pixels / total_pixels,
        "lane_ratio": lane_pixels / total_pixels,
        "background_ratio": background_pixels / total_pixels,
        "drivable_pixels": drivable_pixels,
        "total_pixels": total_pixels
    }


def main():
    print("="*60)
    print("语义分割可视化")
    print("="*60)
    
    # 加载模型
    print(f"\n加载模型: {MODEL_XML}")
    core = Core()
    model = core.compile_model(str(MODEL_XML), "CPU")
    
    # 获取输入输出信息
    input_tensor = model.input(0)
    output_tensor = model.output(0)
    
    input_shape = input_tensor.shape
    _, _, input_h, input_w = input_shape
    print(f"输入尺寸: {input_w}x{input_h}")
    
    # 获取测试图片
    test_dirs = [
        ROOT / "datasets" / "bdd100k_subset_500" / "images" / "val",
        ROOT / "datasets" / "cityscapes" / "leftImg8bit" / "val",
        ROOT / "datasets" / "acdc",
        ROOT / "datasets" / "idd"
    ]
    
    all_images = []
    for test_dir in test_dirs:
        if test_dir.exists():
            images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
            all_images.extend(images)
    
    if len(all_images) == 0:
        print("错误: 未找到测试图片")
        return
    
    print(f"找到 {len(all_images)} 张图片")
    
    # 随机选择 20 张图片
    num_samples = min(20, len(all_images))
    selected_images = random.sample(all_images, num_samples)
    
    print(f"\n生成 {num_samples} 张可视化样本...")
    
    # 创建汇总信息
    summary = []
    
    for idx, img_path in enumerate(selected_images, 1):
        print(f"\n处理 [{idx}/{num_samples}]: {img_path.name}")
        
        # 预处理
        img_input, original_size, original_img = preprocess_image(img_path, input_h, input_w)
        
        if img_input is None:
            print(f"  跳过: 无法读取图片")
            continue
        
        # 推理
        infer_request = model.create_infer_request()
        infer_request.infer({input_tensor: img_input})
        output = infer_request.get_output_tensor().data
        
        # 后处理
        mask = output.squeeze()
        if mask.ndim > 2:
            mask = mask[0]
        
        # 调整到原始尺寸
        mask_resized = cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]))
        
        # 可视化
        vis_img = visualize_segmentation(original_img, mask_resized, alpha=0.4)
        
        # 计算指标
        metrics = calculate_metrics(mask_resized)
        
        # 添加文字标注
        text_lines = [
            f"Image: {img_path.name}",
            f"Drivable: {metrics['drivable_ratio']*100:.1f}%",
            f"Lane: {metrics['lane_ratio']*100:.1f}%",
            f"Size: {original_size[1]}x{original_size[0]}"
        ]
        
        for i, line in enumerate(text_lines):
            cv2.putText(vis_img, line, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, line, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 保存结果
        output_path = OUTPUT_DIR / f"seg_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), vis_img)
        
        print(f"  可行驶区域: {metrics['drivable_ratio']*100:.1f}%")
        print(f"  保存至: {output_path.name}")
        
        # 记录汇总
        summary.append({
            "image": img_path.name,
            "drivable_ratio": metrics['drivable_ratio'],
            "lane_ratio": metrics['lane_ratio'],
            "output": output_path.name
        })
    
    # 保存原图对比
    print("\n生成对比图...")
    for idx, img_path in enumerate(selected_images[:5], 1):  # 只生成 5 张对比
        img_input, original_size, original_img = preprocess_image(img_path, input_h, input_w)
        if img_input is None:
            continue
        
        infer_request = model.create_infer_request()
        infer_request.infer({input_tensor: img_input})
        output = infer_request.get_output_tensor().data
        mask = output.squeeze()
        if mask.ndim > 2:
            mask = mask[0]
        mask_resized = cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]))
        
        vis_img = visualize_segmentation(original_img, mask_resized, alpha=0.4)
        
        # 创建对比图
        comparison = np.hstack([original_img, vis_img])
        comparison_path = OUTPUT_DIR / f"comparison_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"  对比图: {comparison_path.name}")
    
    # 保存汇总报告
    import csv
    summary_path = OUTPUT_DIR / "segmentation_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "drivable_ratio", "lane_ratio", "output"])
        writer.writeheader()
        writer.writerows(summary)
    
    # 打印汇总
    print("\n" + "="*60)
    print("可视化完成")
    print("="*60)
    print(f"\n生成文件:")
    print(f"  - 可视化图片: {num_samples} 张")
    print(f"  - 对比图片: 5 张")
    print(f"  - 汇总报告: segmentation_summary.csv")
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # 统计平均可行驶区域比例
    avg_drivable = sum(s['drivable_ratio'] for s in summary) / len(summary)
    print(f"\n平均可行驶区域比例: {avg_drivable*100:.2f}%")
    
    if avg_drivable < 0.3:
        print("⚠️  警告: 可行驶区域比例偏低，可能分割效果不佳")
    elif avg_drivable > 0.6:
        print("⚠️  注意: 可行驶区域比例偏高，可能包含过多区域")
    else:
        print("✅ 可行驶区域比例正常")


if __name__ == "__main__":
    main()
