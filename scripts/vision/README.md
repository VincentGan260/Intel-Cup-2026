# Vision Scripts

视觉模块调试脚本目录。

当前只保留第一阶段真正要运行的脚本：

1. `01_test_yolo_image.py`：先跑通单张图片目标检测。

摄像头检测、语义分割、视觉融合脚本等到对应阶段开始时再创建，避免目录里堆太多空文件。

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
