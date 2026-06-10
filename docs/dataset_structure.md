# BDD100K 数据集目录结构说明

## 一、数据集概览

本项目使用 **BDD100K** 数据集，包含用于自动驾驶场景的图像和标注数据。

### 数据集统计

| 类型 | 数量 |
|------|------|
| 图片文件（.jpg） | 110,000 张 |
| 标注文件（.json） | 100,000 个 |
| 语义分割标注 | 8,000 张 |
| 可行驶区域标注 | 10,000 张 |

---

## 二、目录结构

```
datasets/
└── bdd100k/                              # BDD100K 数据集根目录
    ├── images/                           # 图像数据
    │   ├── 100k/                         # 完整数据集（10万张）
    │   │   ├── train/                    # 训练集（70,000张）
    │   │   │   └── *.jpg
    │   │   ├── val/                      # 验证集（10,000张）
    │   │   │   └── *.jpg
    │   │   └── test/                     # 测试集（20,000张）
    │   │       └── *.jpg
    │   └── 10k/                          # 小型数据集（1万张）
    │       ├── train/                    # 训练集（7,000张）
    │       ├── val/                      # 验证集（1,000张）
    │       └── test/                     # 测试集（2,000张）
    │
    └── labels/                           # 标注数据
        ├── 100k/                         # 目标检测标注
        │   ├── train/                    # 训练集标注（7,000个）
        │   │   └── *_{image_name}_label.json
        │   ├── val/                      # 验证集标注（1,000个）
        │   │   └── *_{image_name}_label.json
        │   └── test/                     # 测试集标注（2,000个）
        │       └── *_{image_name}_label.json
        │
        ├── bdd100k_seg_maps/             # 语义分割标注
        │   ├── labels/                   # 灰度标注（类别索引）
        │   │   ├── train/                # 训练集（7,000张）
        │   │   └── val/                  # 验证集（1,000张）
        │   └── color_labels/             # 彩色标注（可视化）
        │       ├── train/
        │       └── val/
        │
        └── bdd100k_drivable_maps/        # 可行驶区域标注
            ├── labels/                   # 灰度标注
            │   ├── train/                # 训练集（7,000张）
            │   └── val/                  # 验证集（3,000张）
            └── color_labels/             # 彩色标注
                ├── train/
                └── val/
```

---

## 三、文件格式说明

### 3.1 图像文件

| 格式 | 尺寸 | 说明 |
|------|------|------|
| JPEG | 1280 × 720 | 原始街景图像 |

### 3.2 目标检测标注（JSON格式）

```json
{
  "name": "0000f77c-6257be58",
  "videoName": null,
  "width": 1280,
  "height": 720,
  "frames": [
    {
      "timestamp": 100000,
      "objects": [
        {
          "category": "car",
          "box2d": {
            "x1": 0.0,
            "y1": 416.0,
            "x2": 175.0,
            "y2": 561.0
          }
        }
      ]
    }
  ]
}
```

**目标检测类别（10类）：**

| 类别 | 说明 | 威胁级别 |
|------|------|----------|
| car | 小汽车 | 高 |
| bus | 公交车 | 高 |
| truck | 卡车 | 高 |
| pedestrian | 行人 | 高 |
| rider | 骑行者 | 高 |
| bike | 自行车 | 中 |
| motor | 摩托车 | 中 |
| traffic light | 交通灯 | 低 |
| traffic sign | 交通标志 | 低 |
| train | 火车 | 低 |

### 3.3 语义分割标注

| 标签值 | 类别 | 颜色 |
|--------|------|------|
| 0 | 背景 | 黑色 |
| 1 | 道路 | 灰色 |
| 2 | 人行道 | 浅蓝色 |
| 3 | 建筑 | 红色 |
| 4 | 围墙 | 棕色 |
| 5 | 围栏 | 深棕色 |
| 6 | 杆 | 深灰色 |
| 7 | 交通灯 | 黄色 |
| 8 | 交通标志 | 青色 |
| 9 | 植被 | 绿色 |
| 10 | 地形 | 浅绿色 |
| 11 | 天空 | 蓝色 |
| 12 | 人 | 紫色 |
| 13 | 骑手 | 粉红色 |
| 14 | 汽车 | 深蓝色 |
| 15 | 卡车 | 深蓝色 |
| 16 | 公共汽车 | 深蓝色 |
| 17 | 火车 | 深蓝色 |
| 18 | 摩托车 | 深蓝色 |
| 19 | 自行车 | 深蓝色 |

---

## 四、模型优化指南

### 4.1 当前使用的模型

| 模型类型 | 模型路径 | 说明 |
|----------|----------|------|
| 目标检测 | `models/yolo26n_openvino_model/yolo26n.xml` | YOLO26n OpenVINO格式 |
| 语义分割 | `models/openvino/road-segmentation-adas-0001/road-segmentation-adas-0001.xml` | 道路分割模型 |

### 4.2 模型优化建议

#### 目标检测模型优化

1. **数据集微调**：
   - 使用 `datasets/bdd100k/images/100k/train/`（70,000张）进行微调
   - 使用 `datasets/bdd100k/labels/100k/train/` 作为标注

2. **重点关注类别**（对驾驶人有威胁的目标）：
   - car, bus, truck, pedestrian, rider, bike, motor

3. **数据增强建议**：
   - 随机裁剪、翻转、旋转
   - 亮度/对比度调整
   - MixUp/CutMix

#### 语义分割模型优化

1. **数据集**：
   - 使用 `datasets/bdd100k/labels/bdd100k_drivable_maps/labels/train/`（7,000张）

2. **关注区域**：
   - 道路区域（标签1）
   - 可行驶区域

---

## 五、本地运行说明

### 5.1 环境要求

```bash
# 创建并激活环境
conda create -n intel python=3.14 -y
conda activate intel

# 安装依赖
pip install ultralytics openvino-dev opencv-python pandas numpy
```

### 5.2 运行检测脚本

```bash
# 目标检测可视化
python scripts/visualize_detection.py

# 语义分割可视化
python scripts/visualize_segmentation.py

# 模型准确率评估
python scripts/evaluate_accuracy.py

# 批量测试（FPS、延迟等）
python scripts/eval_bdd100k_speed_count.py
```

### 5.3 结果输出目录

| 目录 | 用途 |
|------|------|
| `runs/detection_visualization_threat/` | 威胁目标检测结果可视化 |
| `runs/segmentation_visualization/` | 语义分割结果可视化 |
| `runs/accuracy_eval/` | 模型准确率评估结果 |
| `runs/bdd100k_eval/` | 批量测试结果（FPS、延迟） |

---

## 六、文件匹配规则

### 图像与标注文件名对应关系

| 图像文件 | 标注文件 |
|----------|----------|
| `ca35c192-7f0eadba.jpg` | `ca35c192-7f0eadba_label.json` |

---

## 七、注意事项

1. **数据集大小**：完整数据集约 100GB，请确保有足够磁盘空间
2. **标注文件**：测试集（test）可能没有标注文件，仅用于最终评估
3. **Git 忽略**：数据集目录已添加到 `.gitignore`，不会被上传
4. **模型路径**：确保模型文件放在 `models/` 目录下

---

**文档版本**：v1.0  
**生成日期**：2026年6月  
**项目**：Intel Cup 2026 - 目标检测