"""
语义分割结果可视化脚本
生成原始图片、标注mask和模型预测结果的对比图
"""

import cv2
import numpy as np
from openvino import Core
from pathlib import Path

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
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
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model_path = ROOT / "models" / "openvino" / "road-segmentation-adas-0001" / "road-segmentation-adas-0001.xml"
    core = Core()
    model = core.compile_model(str(model_path), "CPU")
    
    # 获取图片列表
    img_files = list(IMAGES_DIR.glob("*.jpg"))[:10]  # 取前10张
    
    for i, img_path in enumerate(img_files):
        # 读取图片
        img = cv2.imread(str(img_path))
        img_original = img.copy()
        
        # 预处理图片
        img_input = cv2.resize(img, (896, 512))
        input_tensor = np.expand_dims(img_input.transpose(2, 0, 1).astype(np.float32), 0)
        
        # 推理
        result = model([input_tensor])[0]
        pred_mask = np.argmax(result, axis=1)[0]
        
        # 读取标注
        seg_file = SEG_LABELS_DIR / (img_path.stem + "_drivable_id.png")
        if seg_file.exists():
            gt_mask = cv2.imread(str(seg_file), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (896, 512), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask = np.zeros((512, 896), dtype=np.uint8)
        
        # 创建可视化图片
        vis_img = create_visualization(img_input, gt_mask, pred_mask)
        
        # 保存结果
        output_path = OUTPUT_DIR / f"segmentation_result_{i:02d}.jpg"
        cv2.imwrite(str(output_path), vis_img)
        print(f"已保存: {output_path}")
    
    print(f"\n可视化结果已保存至: {OUTPUT_DIR}")

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