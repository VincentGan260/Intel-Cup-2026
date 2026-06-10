"""
BDD100K 模型准确率评估脚本
使用标注文件计算目标检测和语义分割的准确率指标
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from openvino import Core

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
DATA_DIR = ROOT / "datasets" / "bdd100k"
LABELS_DIR = DATA_DIR / "labels" / "100k" / "val"
IMAGES_DIR = DATA_DIR / "images" / "100k" / "val"

# BDD100K 类别映射
BDD100K_CLASSES = [
    "car", "bus", "truck", "pedestrian", "rider", 
    "traffic light", "traffic sign", "bike", "motor", "train"
]

# 威胁类别索引（对驾驶人有潜在威胁的目标）
THREAT_CLASSES = [0, 1, 2, 3, 4, 7, 8]  # car, bus, truck, pedestrian, rider, bike, motor

def load_detection_labels():
    """加载目标检测标注（只关注威胁类别）"""
    labels = {}
    det_dir = LABELS_DIR
    
    for json_file in det_dir.glob("*.json"):
        img_name = json_file.stem.replace("_label", "") + ".jpg"
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        boxes = []
        for obj in data.get('frames', [{}])[0].get('objects', []):
            if 'box2d' in obj and obj['category'] in BDD100K_CLASSES:
                cls_idx = BDD100K_CLASSES.index(obj['category'])
                # 只保留威胁类别
                if cls_idx in THREAT_CLASSES:
                    box = obj['box2d']
                    boxes.append({
                        'class': cls_idx,
                        'bbox': [box['x1'], box['y1'], box['x2'], box['y2']],
                        'category': obj['category']
                    })
        labels[img_name] = boxes
    
    return labels

def load_segmentation_labels():
    """加载可行驶区域标注"""
    labels = {}
    seg_dir = DATA_DIR / "labels" / "bdd100k_drivable_maps" / "labels" / "val"
    
    for png_file in seg_dir.glob("*.png"):
        img_name = png_file.stem.replace("_drivable_id", "") + ".jpg"
        mask = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
        labels[img_name] = mask
    
    return labels

def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_detection(model, labels, num_samples=100):
    """评估目标检测准确率"""
    print("=== 评估目标检测准确率 ===")
    
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    total_labels = 0
    total_preds = 0
    
    img_names = list(labels.keys())[:num_samples]
    
    for i, img_name in enumerate(img_names):
        if i % 20 == 0:
            print(f"处理 {i}/{num_samples}...")
        
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            continue
        
        # 加载标注
        gt_boxes = labels[img_name]
        total_labels += len(gt_boxes)
        
        # 模型推理
        results = model.predict(source=str(img_path), imgsz=640, conf=0.25, verbose=False)
        pred_boxes = []
        for det in results[0].boxes:
            cls = int(det.cls[0].item())
            # 只保留威胁类别
            if cls in THREAT_CLASSES:
                pred_boxes.append({
                    'class': cls,
                    'bbox': det.xyxy[0].cpu().numpy().tolist(),
                    'conf': det.conf[0].item()
                })
        total_preds += len(pred_boxes)
        
        # 匹配预测和标注
        matched = set()
        for pred in pred_boxes:
            best_iou = 0
            best_idx = -1
            for j, gt in enumerate(gt_boxes):
                if j not in matched:
                    iou = compute_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_idx = j
            if best_idx >= 0:
                tp += 1
                matched.add(best_idx)
        
        fp += len(pred_boxes) - len(matched)
        fn += len(gt_boxes) - len(matched)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n目标检测评估结果:")
    print(f"样本数: {len(img_names)}")
    print(f"标注目标数: {total_labels}")
    print(f"预测目标数: {total_preds}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    
    return {
        'samples': len(img_names),
        'gt_count': total_labels,
        'pred_count': total_preds,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_segmentation(model, seg_labels, num_samples=50):
    """评估语义分割准确率"""
    print("\n=== 评估语义分割准确率 ===")
    
    total_iou = 0
    total_pixels = 0
    correct_pixels = 0
    
    img_names = list(seg_labels.keys())[:num_samples]
    
    for i, img_name in enumerate(img_names):
        if i % 10 == 0:
            print(f"处理 {i}/{num_samples}...")
        
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            continue
        
        # 加载标注
        gt_mask = seg_labels[img_name]
        if gt_mask is None:
            continue
        
        # 读取图片（使用模型期望的尺寸 896x512）
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (896, 512))
        
        # 模型推理
        input_tensor = np.expand_dims(img.transpose(2, 0, 1).astype(np.float32), 0)
        result = model([input_tensor])[0]
        pred_mask = np.argmax(result, axis=1)[0]
        
        # 调整标注mask大小以匹配预测结果
        gt_mask = cv2.resize(gt_mask, (896, 512), interpolation=cv2.INTER_NEAREST)
        
        # 计算 IoU（考虑所有可行驶区域，标签为 1 和 2）
        road_gt = ((gt_mask == 1) | (gt_mask == 2)).astype(np.float32)
        # 模型输出标签值为 0-3，标签 1 是主要的道路类别
        road_pred = (pred_mask == 1).astype(np.float32)
        
        intersection = np.sum(road_gt * road_pred)
        union = np.sum(road_gt) + np.sum(road_pred) - intersection
        iou = intersection / union if union > 0 else 0
        total_iou += iou
        
        # 计算像素准确率
        correct_pixels += np.sum(pred_mask == gt_mask)
        total_pixels += gt_mask.size
    
    # 计算指标
    miou = float(total_iou / len(img_names)) if len(img_names) > 0 else 0.0
    pixel_acc = float(correct_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    print(f"\n语义分割评估结果:")
    print(f"样本数: {len(img_names)}")
    print(f"mIoU (道路类别): {miou:.4f}")
    print(f"像素准确率: {pixel_acc:.4f}")
    
    return {
        'samples': len(img_names),
        'miou': miou,
        'pixel_acc': pixel_acc
    }

def main():
    print("="*60)
    print("BDD100K 模型准确率评估")
    print("="*60)
    
    # 加载标注
    print("加载标注文件...")
    det_labels = load_detection_labels()
    seg_labels = load_segmentation_labels()
    print(f"加载目标检测标注: {len(det_labels)} 张图片")
    print(f"加载语义分割标注: {len(seg_labels)} 张图片")
    
    # 加载模型
    print("\n加载模型...")
    det_model = YOLO("yolo26n.pt")
    seg_model = Core().compile_model("models/openvino/road-segmentation-adas-0001/road-segmentation-adas-0001.xml", "CPU")
    
    # 评估目标检测
    det_results = evaluate_detection(det_model, det_labels, num_samples=100)
    
    # 评估语义分割
    seg_results = evaluate_segmentation(seg_model, seg_labels, num_samples=50)
    
    # 保存结果
    output_dir = ROOT / "runs" / "accuracy_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "accuracy_results.json", 'w') as f:
        json.dump({
            'detection': det_results,
            'segmentation': seg_results
        }, f, indent=2)
    
    print(f"\n结果已保存至: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()