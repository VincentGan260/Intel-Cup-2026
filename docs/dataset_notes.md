# 数据集详细说明

## 1. BDD100K

### 基本信息
- **数据集名称**: BDD100K
- **下载来源**: [BDD100K Official Website](https://bdd-data.berkeley.edu/)
- **下载日期**: 2026年
- **许可证**: BSD-3-Clause License
- **存放路径**: `datasets/bdd100k/`

### 使用任务
- ✅ 目标检测
- ✅ 可行驶区域分割
- ✅ 场景泛化测试

### 筛选标准
从 BDD100K test 集中筛选：
- 白天正常道路：100 张
- 傍晚/弱光：50 张
- 路口/斑马线：50 张
- 行人明显出现：50 张
- 自行车/摩托车/电动车：50 张
- 前方车辆明显出现：50 张

### 当前状态
- ✅ 已筛选 500 张图片作为测试子集
- ✅ 子集存放于 `datasets/bdd100k_subset_500/images/val/`
- ✅ 完整数据集已下载 (100,000 张)
- ⏳ 标注文件待下载

### 类别映射
| 原始类别 | 项目类别 | 风险类型 |
|---------|---------|---------|
| person | person | pedestrian |
| bicycle | bicycle | non_motor_vehicle |
| motorcycle | motorcycle | non_motor_vehicle |
| car | car | motor_vehicle |
| bus | bus | motor_vehicle |
| truck | truck | motor_vehicle |

---

## 2. Cityscapes

### 基本信息
- **数据集名称**: Cityscapes
- **下载来源**: [Cityscapes Official Website](https://www.cityscapes-dataset.com/)
- **下载日期**: 2026年6月
- **许可证**: CC BY-NC-SA 4.0
- **存放路径**: `datasets/cityscapes/`

### 使用任务
- ✅ 语义分割基准测试
- ✅ PIDNet 模型测试

### 筛选标准
从 val 集中筛选：
- 道路清晰的图片：100～200 张
- 优先选择行人车辆明显的场景

### 当前状态
- ✅ 已下载 (20,000 张)
- ✅ 已筛选 150 张图片
- ✅ 划分文件已生成

### 分割类别合并
| 原始类别 | 合并后 |
|---------|-------|
| road | drivable_area |
| sidewalk | drivable_area |
| terrain | drivable_area |
| 其他 | non_drivable_area |

---

## 3. IDD Lite

### 基本信息
- **数据集名称**: IDD Lite
- **下载来源**: [IDD Dataset](https://idd.insaan.iiit.ac.in/)
- **下载日期**: 2026年6月
- **许可证**: CC BY 4.0
- **存放路径**: `datasets/idd/`

### 使用任务
- ✅ 复杂道路场景测试
- ✅ 非规则道路测试

### 使用场景
- 道路边界混乱
- 行人、两轮车、汽车混杂
- 非机动车和机动车交织
- 路口规则不明显

### 当前状态
- ✅ 已下载 (3,214 张)
- ✅ 已筛选 80 张图片
- ✅ 划分文件已生成

---

## 4. ACDC

### 基本信息
- **数据集名称**: ACDC (Adverse Conditions Dataset)
- **下载来源**: [ACDC Dataset](https://acdc.vision.ee.ethz.ch/)
- **下载日期**: 2026年6月
- **许可证**: CC BY-NC-SA 4.0
- **存放路径**: `datasets/acdc/`

### 使用任务
- ✅ 弱光鲁棒性测试
- ✅ 恶劣天气测试

### 筛选计划
| 场景类型 | 数量 | 优先级 |
|---------|-----|-------|
| night | 50 | 高 |
| rain | 50 | 高 |
| fog | 50 | 中 |

### 当前状态
- ✅ 已下载 (8,012 张)
- ✅ 已筛选 150 张图片 (night/rain/fog 各 50 张)
- ✅ 划分文件已生成

---

## 5. 自采数据

### 基本信息
- **数据集名称**: Rider Self-collected
- **采集方式**: 车把视角视频录制
- **许可证**: 项目内部使用
- **存放路径**: `datasets/rider_self_collected/`

### 使用任务
- ✅ 最终验证
- ✅ 模型微调
- ✅ INT8 量化校准

### 采集内容
| 场景类型 | 描述 |
|---------|------|
| 白天正常道路 | 常规骑行场景 |
| 傍晚/弱光 | 低光照条件 |
| 逆光 | 太阳直射 |
| 路口行人横穿 | 行人检测 |
| 前方电动车/自行车 | 非机动车检测 |
| 前方机动车 | 机动车检测 |
| 路边停车遮挡 | 障碍物检测 |
| 颠簸运动模糊 | 运动模糊鲁棒性 |

### 抽帧规则
- 普通视频：每秒抽 1 帧
- 行人横穿/前车急停：每秒抽 3～5 帧
- 删除模糊严重、重复过高的帧

### 当前状态
- ⏳ 待采集
- ⏳ 目标：5～10 段视频，抽帧 300～800 张

---

## 数据集使用优先级

| 优先级 | 数据集 | 用途 |
|-------|-------|------|
| P0 | BDD100K | 主测试集，目标检测 |
| P0 | 自采数据 | 最终验证，INT8校准 |
| P1 | Cityscapes | 分割基准 |
| P2 | ACDC | 鲁棒性测试 |
| P2 | IDD Lite | 复杂场景测试 |

---

## 数据准备检查清单

### 阶段 1：模型初步测试
- [x] BDD100K 子集 100～200 张 ✅
- [ ] Cityscapes val 50～100 张 ⏳
- [ ] 自采车把视角 50～100 张 ⏳

### 阶段 2：检测+分割融合测试
- [ ] BDD100K Drivable Area 100～200 张 ⏳
- [ ] 自采车把视角 100～200 张 ⏳

### 阶段 3：轻量微调
- [ ] BDD100K Detection 标注 ⏳
- [ ] 自采数据人工标注 ⏳

### 阶段 4：鲁棒性测试
- [ ] ACDC night/rain/fog ⏳
- [ ] IDD Lite ⏳

### 阶段 5：INT8 校准
- [ ] 自采真实车把视角 300～1000 张 ⏳
