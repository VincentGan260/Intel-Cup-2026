"""
目标检测结果可视化脚本（使用 OpenVINO yolo26n 模型）
生成原始图片和检测框的对比图，只显示对驾驶人有潜在威胁的目标
"""

import cv2
import numpy as np
from openvino import Core
from pathlib import Path

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
DATA_DIR = ROOT / "datasets" / "bdd100k"
IMAGES_DIR = DATA_DIR / "images" / "100k" / "val"
OUTPUT_DIR = ROOT / "runs" / "detection_visualization_threat"

# COCO 类别映射（yolo26n 使用 COCO 数据集）
COCO_CLASSES = {
    0: 'person',      # 行人
    1: 'bicycle',     # 自行车
    2: 'car',         # 汽车
    3: 'motorcycle',  # 摩托车
    5: 'bus',         # 公交车
    7: 'truck',       # 卡车
}

# 威胁类别索引（对驾驶人有潜在威胁的目标 - COCO 类别）
THREAT_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck

# 威胁类别颜色映射
THREAT_COLOR_MAP = {
    0: (0, 255, 255),    # person - 青色
    1: (0, 128, 255),    # bicycle - 橙色
    2: (0, 0, 255),      # car - 红色
    3: (128, 128, 0),    # motorcycle - 橄榄绿
    5: (0, 255, 0),      # bus - 绿色
    7: (255, 0, 0),      # truck - 蓝色
}

def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载 OpenVINO yolo26n 模型
    model_path = ROOT / "models" / "yolo26n_openvino_model" / "yolo26n.xml"
    core = Core()
    model = core.compile_model(str(model_path), "CPU")
    
    # 获取输入输出信息
    input_layer = model.input(0)
    output_layer = model.output(0)
    input_shape = input_layer.shape
    input_w, input_h = input_shape[3], input_shape[2]
    print(f"已加载模型: {model_path}")
    print(f"模型输入形状: {input_shape}")
    
    # 获取图片列表
    img_files = list(IMAGES_DIR.glob("*.jpg"))[:15]  # 取前15张
    
    # 推理参数（提高置信度阈值减少错误识别）
    conf_threshold = 0.15
    
    for i, img_path in enumerate(img_files):
        # 读取图片
        img = cv2.imread(str(img_path))
        original_h, original_w = img.shape[:2]
        
        # 预处理图片
        img_resized = cv2.resize(img, (input_w, input_h))
        input_tensor = np.expand_dims(img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0, 0)
        
        # 推理
        result = model([input_tensor])[output_layer]
        
        # 解析检测结果（YOLO26 OpenVINO 输出格式: [1, 300, 6] = [batch, num_boxes, [x1, y1, x2, y2, conf, cls]]）
        # 坐标范围: 0-640（输入图片尺寸）
        pred_boxes = []
        output = result[0]  # 取 batch=0
        
        for row in output:
            # 输出格式: [x1, y1, x2, y2, conf, cls]（坐标范围 0-640）
            x1, y1, x2, y2, conf, cls = row
            
            # 跳过低置信度检测
            if conf < conf_threshold:
                continue
            
            # 转换为整数类别
            cls = int(cls)
            
            # 只保留威胁类别
            if cls not in THREAT_CLASSES:
                continue
            
            # 将 640x640 范围内的坐标转换为原始图片尺寸
            x1 = int(x1 * original_w / input_w)
            y1 = int(y1 * original_h / input_h)
            x2 = int(x2 * original_w / input_w)
            y2 = int(y2 * original_h / input_h)
            
            pred_boxes.append({
                'class': cls,
                'bbox': [x1, y1, x2, y2],
                'conf': float(conf)
            })
        
        # 绘制检测框
        vis_img = draw_detections(img.copy(), pred_boxes)
        
        # 保存结果
        output_path = OUTPUT_DIR / f"detection_threat_{i:02d}.jpg"
        cv2.imwrite(str(output_path), vis_img)
        
        # 打印详细信息
        cls_stats = {}
        for box in pred_boxes:
            cls_name = COCO_CLASSES[box['class']]
            cls_stats[cls_name] = cls_stats.get(cls_name, 0) + 1
        
        print(f"已保存: {output_path} (共 {len(pred_boxes)} 个目标)")
        for cls_name, count in cls_stats.items():
            print(f"  - {cls_name}: {count}个")
    
    print(f"\n可视化结果已保存至: {OUTPUT_DIR}")
    print(f"使用参数: 置信度阈值={conf_threshold}, 输入尺寸={input_w}x{input_h}")

def draw_detections(img, pred_boxes):
    """绘制检测框（只显示威胁类别）"""
    for box in pred_boxes:
        # 获取边界框
        x1, y1, x2, y2 = box['bbox']
        
        # 获取类别和置信度
        cls = box['class']
        conf = box['conf']
        
        # 获取类别名称
        cls_name = COCO_CLASSES.get(cls, f'unknown({cls})')
        
        # 获取颜色
        color = THREAT_COLOR_MAP.get(cls, (255, 255, 255))
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签和置信度
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

if __name__ == "__main__":
    main()