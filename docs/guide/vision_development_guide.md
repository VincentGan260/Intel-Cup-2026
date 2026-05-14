# Vision Development Guide

本指南用于约束视觉模块的目录结构和文件职责。

当前阶段：目标检测（YOLO）与道路语义分割（OpenVINO `road-segmentation-adas-0001`）各有最小 demo；融合与整机流程后续再加。

## 1. 视觉模块目录结构

```text
src/vision/
├─ __init__.py
├─ common/
│  ├─ types.py              # DetectionResult / SegmentationResult / VisionResult
│  ├─ interfaces.py         # BaseDetector / BaseSegmenter
│  ├─ preprocess.py         # 读图、BGR 校验等
│  └─ visualize.py          # 检测框、mask 叠加、保存图像
├─ detection/
│  ├─ detector.py           # 从配置构建检测器
│  ├─ postprocess.py        # YOLO 输出 -> DetectionResult
│  ├─ class_mapping.py      # COCO -> 风险类别 + CLASS_BASE_RISK
│  └─ models/
│     ├─ yolo_ultralytics.py
│     └─ yolo_openvino.py   # 占位
├─ segmentation/
│  ├─ segmenter.py
│  ├─ postprocess.py        # logits -> label / drivable
│  ├─ mask_utils.py
│  └─ models/
│     ├─ road_adas_openvino.py
│     ├─ pidnet_pytorch.py  # 占位
│     └─ pidnet_openvino.py # 占位
├─ perception/
│  ├─ vision_pipeline.py
│  └─ target_on_road.py
└─ risk/
   └─ visual_risk.py

scripts/vision/
├─ 01_test_yolo_image.py
├─ 02_test_segmentation_openvino.py
└─ 03_test_vision_pipeline.py

configs/vision/
├─ detection.yaml
├─ segmentation_openvino.yaml
└─ vision_pipeline.yaml
```

## 2. 文件职责

- **主流程**只依赖 `BaseDetector` / `BaseSegmenter`（见 `common/interfaces.py`），不直接 import Ultralytics / OpenVINO 具体类。
- **`detection/models/`** 与 **`segmentation/models/`**：仅负责加载权重与推理；类别筛选、可行驶判断、风险分数在 `detection/postprocess.py`、`perception/`、`risk/`。
- **`perception/VisionPipeline`**：检测 + 分割 + `in_drivable_area` + `visual_risk`；输出 `VisionResult`（含 `drivable_mask` 副本便于 fusion）。
- **`risk/visual_risk.py`**：仅视觉初判；系统级融合放在未来的 `src/fusion/`。
- **权重文件**：OpenVINO IR 等放在仓库根目录 `models/openvino/`（大文件 `.gitignore` 忽略）；YOLO 等路径由 `configs/vision/detection.yaml` 指定。

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

注意：系统级多传感器风险融合放在 `src/fusion/`（待建），不在 `src/vision/risk/`。

## 4. 提交约定

- 可以提交：代码、配置、说明文档、少量测试样例。
- 不要提交：大模型权重、完整训练数据集、大量运行结果、本机绝对路径。
- 新增依赖时，同步更新 `requirements/train.txt` 和必要的环境说明。
- 改动运行命令、配置字段或目录结构时，同步更新本指南。

## 5. 关于 `data/raw`、`data/samples`、`data/processed`（概念说明）

很多项目会把数据分成三类目录，含义一般是：

- **`raw`**：原始采集数据，体积大、未整理，通常不进 Git。
- **`samples`**：少量可随仓库携带的测试样例，用来让队友快速复现。
- **`processed`**：清洗、裁剪、转格式、划分训练集验证集之后的数据。

当前仓库为了减轻第一阶段查找成本，**只保留 `data/samples/detection/`** 放检测测试图。等你们开始做自采数据集、标注流水线时，再按需加回 `raw/`、`processed/` 即可。
