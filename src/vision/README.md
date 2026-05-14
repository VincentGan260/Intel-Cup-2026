# Vision Module（视觉子系统）

骑手前向安全预警中的**视觉部分**：目标检测、语义分割、检测分割融合、视觉风险初判。

## 目录约定

- **`common/`**：统一数据结构（`types.py`）、抽象接口（`interfaces.py`）、通用预处理与可视化。
- **`detection/`**：只负责检测；具体 YOLO 实现在 `detection/models/`，通过 `BaseDetector` 与管线解耦。
- **`segmentation/`**：只负责分割与可行驶 mask；OpenVINO ADAS、后续 PIDNet 放在 `segmentation/models/`。
- **`perception/`**：`VisionPipeline` 组合检测与分割，并调用 `target_on_road` 做几何融合。
- **`risk/`**：仅视觉侧 `visual_risk` 初判；雷达、IMU、GPS、TTC 等放在未来的 `src/fusion/`。

**权重路径**在仓库根目录 `models/` 下配置；**不写死**具体框架，换 PyTorch / OpenVINO 时只替换 `*/models/` 内实现类，并在 `detector.py` / `segmenter.py` 工厂中挂接。

## 脚本入口

```bash
python scripts/vision/01_test_yolo_image.py
python scripts/vision/02_test_segmentation_openvino.py
python scripts/vision/03_test_vision_pipeline.py
```

配置分别为：

- `configs/vision/detection.yaml`
- `configs/vision/segmentation_openvino.yaml`
- `configs/vision/vision_pipeline.yaml`

## 输出位置

- 视觉输出统一在 **`runs/vision/`** 下：检测为 `runs/vision/detect/<project>/<name>/`（`detection.yaml` 的 `output`），分割叠加图见 `segmentation_openvino.yaml` 的 `output.overlay_path`，管线调试图见 `vision_pipeline.yaml`。
- `outputs/vision/`：预留项目自产结构化结果（如 JSON），非 Ultralytics 默认输出。
