# 数据集管理说明

## 目录结构

```
data/
├── README.md          # 本文件，数据说明
├── samples/           # 少量测试样例图片（可提交 Git）
│   └── detection/     # 检测测试样例
├── splits/            # 训练/验证数据划分文件（可提交 Git）
└── dataset_configs/   # 数据集配置文件（可提交 Git）
```

## 数据集存放规范

完整数据集**不存放**在项目目录内，应放在外部目录：

- **Linux**: `/home/xxx/datasets/` 或 `/data/datasets/`
- **macOS**: `/Users/xxx/datasets/`
- **Windows**: `D:\datasets\`

### 外部数据集目录结构

```
datasets/
├── bdd100k/              # BDD100K 数据集
│   ├── images/           # 图片文件
│   └── labels/           # 标注文件
├── bdd100k_subset_500/   # BDD100K 测试子集（500张）
│   └── images/
│       └── val/
├── cityscapes/           # Cityscapes 数据集
├── idd/                  # IDD/IDD Lite 数据集
├── acdc/                 # ACDC 恶劣条件数据集
└── rider_self_collected/ # 自采车把视角数据
    ├── videos/           # 原始视频
    └── frames/           # 抽帧图片
```

## 数据划分文件

`splits/` 目录存放各数据集的图片列表：

- `bdd_selected.txt` - BDD100K 筛选图片列表
- `cityscapes_selected.txt` - Cityscapes 筛选图片列表
- `idd_lite_selected.txt` - IDD Lite 筛选图片列表
- `acdc_selected.txt` - ACDC 筛选图片列表
- `self_collected_selected.txt` - 自采数据筛选列表

## 配置文件

`dataset_configs/` 目录存放数据集配置：

- `bdd_detection.yaml` - BDD100K 检测配置
- `bdd_drivable.yaml` - BDD100K 可行驶区域配置
- `cityscapes_seg.yaml` - Cityscapes 分割配置
- `self_collected.yaml` - 自采数据配置

## 已完成的工作

### BDD100K 子集
- ✅ 从 BDD100K test 集中随机抽取 500 张图片
- ✅ 存放于 `datasets/bdd100k_subset_500/images/val/`
- ✅ 用于模型推理测试和性能评估

## 使用说明

1. **推理测试**：使用 `bdd100k_subset_500` 子集
2. **语义分割**：使用 Cityscapes val 集
3. **鲁棒性测试**：使用 ACDC night/rain/fog
4. **最终验证**：使用自采车把视角数据

## 注意事项

- 完整数据集不提交 Git，通过 `.gitignore` 排除
- 只提交配置文件、划分文件和少量样例
- 配置文件中不要写死个人绝对路径
