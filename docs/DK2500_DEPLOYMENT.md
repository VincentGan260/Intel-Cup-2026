# DK 2500 平台部署指南

## 环境要求

- Ubuntu 22.04 LTS 或 24.04 LTS
- Intel OpenVINO 2024.x
- Python 3.8+

## 快速开始

### 1. 克隆代码

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/Intel-Cup-2026.git
cd Intel-Cup-2026
```

### 2. 安装依赖

```bash
# 安装 Python 依赖
pip install ultralytics openvino opencv-python pandas numpy psutil

# 或者使用 requirements.txt
pip install -r requirements.txt
```

### 3. 下载模型文件

YOLO26n 模型需要单独下载：

```bash
# 在项目根目录执行
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

### 4. 运行测试

#### 目标检测测试（推荐首先运行）

```bash
python scripts/comprehensive_eval.py
```

测试结果会保存在 `runs/comprehensive_eval/` 目录：
- `summary.csv` - 汇总表格
- `bdd100k_detection.csv` - BDD100K 详细结果
- `cityscapes_detection.csv` - Cityscapes 详细结果
- `acdc_detection.csv` - ACDC 详细结果
- `idd_detection.csv` - IDD 详细结果

#### 语义分割测试

```bash
python scripts/segmentation_eval.py
```

测试结果会保存在 `runs/segmentation_eval/` 目录：
- `summary.csv` - 汇总表格
- 各数据集详细 CSV 文件

#### 生成最终报告

```bash
python scripts/generate_report.py
```

报告保存在 `runs/final_report/test_report.md`

### 5. 多设备测试（可选）

如果需要在不同设备上测试：

```bash
# CPU 模式
OPENVINO_DEVICE=CPU python scripts/comprehensive_eval.py

# GPU 模式（如已安装 GPU 驱动）
OPENVINO_DEVICE=GPU python scripts/comprehensive_eval.py
```

## 目录结构

```
Intel-Cup-2026/
├── scripts/
│   ├── comprehensive_eval.py    # 目标检测综合测试
│   ├── segmentation_eval.py    # 语义分割测试
│   ├── generate_report.py      # 生成报告
│   ├── select_datasets.py      # 数据集筛选
│   ├── make_subset.py          # 创建子集
│   └── ...
├── datasets/                   # 数据集目录
│   ├── bdd100k/              # BDD100K 主数据集
│   ├── bdd100k_subset_500/   # 测试子集 (500张)
│   ├── cityscapes/           # Cityscapes 数据集
│   ├── acdc/                 # ACDC 数据集
│   └── idd/                  # IDD 数据集
├── runs/                      # 测试结果输出目录
│   ├── comprehensive_eval/   # 目标检测结果
│   ├── segmentation_eval/    # 分割结果
│   └── final_report/         # 最终报告
├── models/                   # 模型文件目录
│   └── openvino/            # OpenVINO 模型
├── data/                     # 数据配置
│   ├── splits/               # 数据划分文件
│   └── dataset_configs/      # 数据集配置
└── docs/                     # 文档
```

## 数据集准备

测试数据集需要单独下载：

1. **BDD100K** (可选，如已有子集)
   - 下载地址: https://bdd-data.berkeley.edu/

2. **测试子集** (已包含在代码中)
   - `datasets/bdd100k_subset_500/images/val/` 包含 500 张测试图片

## 常见问题

### Q: 缺少模型文件
```bash
# 重新下载 YOLO 模型
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

### Q: 缺少数据集
数据集需要从官方来源下载，测试子集 `bdd100k_subset_500` 包含 500 张图片可直接使用。

### Q: OpenVINO 版本不兼容
```bash
# 升级 OpenVINO
pip install --upgrade openvino
```

## 输出结果解读

### 性能指标

- **latency_ms**: 单张图片推理延迟（毫秒）
- **fps**: 每秒处理图片数
- **vehicle_count**: 平均每张图片检测到的车辆数
- **pedestrian_count**: 平均每张图片检测到的行人数

### 性能目标参考

| 模型 | 目标 FPS | 目标延迟 |
|------|----------|----------|
| YOLO26n | ≥ 25 FPS | ≤ 40 ms |
| 语义分割 | ≥ 50 FPS | ≤ 20 ms |

## 联系与支持

如有问题，请提交 Issue 或联系开发团队。
