# Vision Development Guide

本指南用于约束视觉模块的目录结构和文件职责。

当前阶段先跑通 YOLO26n/YOLOv8n 单张图片目标检测。语义分割、视觉融合、OpenVINO 部署等后续模块，等开始实现时再创建具体文件。

## 1. 视觉模块目录结构

```text
src/vision/
├─ __init__.py
├─ README.md
├─ detector/                   # 目标检测模块
│  ├─ __init__.py
│  ├─ yolo_detector.py
│  └─ postprocess.py
└─ common/                     # 视觉公共数据结构
   ├─ __init__.py
   └─ types.py

scripts/
└─ vision/
   ├─ README.md
   └─ 01_test_yolo_image.py

configs/
└─ vision/
   └─ detection.yaml           # 目标检测配置

models/
└─ vision/
   ├─ detection/               # YOLO26n 等目标检测权重
   └─ segmentation/            # 道路语义分割权重

data/
├─ raw/
│  ├─ detection/               # 目标检测原始数据
│  └─ segmentation/            # 语义分割原始数据
├─ samples/
│  ├─ detection/               # 少量检测测试样例
│  └─ segmentation/            # 少量分割测试样例
└─ processed/
   ├─ detection/               # 处理后的检测数据
   └─ segmentation/            # 处理后的分割数据

outputs/
└─ vision/
   ├─ detection/               # 检测结果
   ├─ segmentation/            # 分割结果
   └─ fusion/                  # 融合结果
```

## 2. 文件职责

- `src/vision/detector/`：目标检测代码，例如 YOLO26n 推理、类别筛选、检测框后处理。
- `src/vision/common/`：视觉通用数据结构，例如检测结果对象。
- `scripts/vision/`：视觉调试脚本，当前只保留单张图片检测脚本。
- `configs/vision/`：视觉模块配置，不写死在代码里。
- `models/vision/`：本地模型权重和导出模型，大文件不进入 Git。
- `outputs/vision/`：视觉模块运行输出，默认不进入 Git。

## 3. 目标检测和语义分割边界

目标检测负责回答：

```text
画面里有什么目标？
目标类别是什么？
目标框在哪里？
置信度是多少？
```

语义分割负责回答：

```text
哪些区域是道路或可行驶区域？
目标是否进入可行驶区域？
道路边界大概在哪里？
```

融合模块再把两者合成最终视觉结果。

注意：`segmenter/`、`fusion/`、视觉风险计算等文件暂时不创建空壳。等目标检测跑通并开始做对应功能时，再按本节边界新增。

## 4. 提交约定

- 可以提交：代码、配置、说明文档、少量测试样例。
- 不要提交：大模型权重、完整训练数据集、大量输出结果、本机绝对路径。
- 新增依赖时，同步更新 `requirements/train.txt` 和必要的环境说明。
- 改动运行命令、配置字段或目录结构时，同步更新本指南。
