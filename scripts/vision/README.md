# Vision Scripts

视觉模块调试脚本目录。

当前脚本列表：

1. `01_test_yolo_image.py`：单张图片目标检测（Ultralytics YOLO）。
2. `02_test_segmentation_openvino.py`：OpenVINO 道路分割 `road-segmentation-adas-0001` 最小 demo。
3. `03_test_vision_pipeline.py`：检测 + 分割 + 可行驶区域 + 视觉风险最小闭环。

## OpenVINO 道路分割 demo

1. 下载模型（项目根目录执行）：

```bash
omz_downloader --name road-segmentation-adas-0001 --output_dir models/openvino
```

2. OMZ 默认会落在 `models/openvino/intel/.../FP32/`。将其中 **`.xml` 与 `.bin`** 复制到 `models/openvino/road-segmentation-adas-0001/`，与配置里短路径一致（见 `segmentation_openvino.yaml` 的 `model.xml_path`）。

3. 确认 `configs/vision/segmentation_openvino.yaml` 里 `model.xml_path` 指向该 `.xml`。

4. 运行：

```bash
python scripts/vision/02_test_segmentation_openvino.py
```

叠加图保存路径由配置项 `output.overlay_path` 决定（默认在 `runs/vision/segmentation_openvino/demo/`）。

## VisionPipeline 最小闭环

```bash
python scripts/vision/03_test_vision_pipeline.py
```

配置：`configs/vision/vision_pipeline.yaml`（内含 `detection_config` 与 `segmentation_config` 路径）。

## 单张图片检测

安装依赖：

```bash
pip install -r requirements/train.txt
```

运行默认示例图：

```bash
python scripts/vision/01_test_yolo_image.py
```

更换图片、模型、置信度、输出目录等参数时，修改：

```text
configs/vision/detection.yaml
```

例如 `yolo26n.pt` 暂时不可用时，把配置里的 `detector.model_path` 改成 `yolov8n.pt`。

## 结果保存位置

Ultralytics 会把带框图片保存到仓库根目录下：

```text
runs/vision/detect/<output.project>/<output.name>/
```

默认配置对应为：

```text
runs/vision/detect/yolo26n_demo/pred/
```

`outputs/vision/` 目前只作占位，用于以后你们自己导出 JSON、风险分数等非 Ultralytics 默认文件。
