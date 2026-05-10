# Vision Module

视觉感知模块负责前向摄像头相关能力，包括目标检测、道路语义分割，以及检测结果与道路区域的视觉融合。

## 目录结构

```text
src/vision/
├─ detector/        # 目标检测，当前先做 YOLO26n/YOLOv8n
└─ common/          # 视觉通用数据结构
```

## 当前开发顺序

1. 在 `detector/` 中跑通 YOLO26n/YOLOv8n 图片检测。
2. 跑通摄像头实时检测。
3. 做人、非机动车、机动车类别映射和过滤。
4. 后续再创建 `segmenter/` 做道路语义分割。
5. 最后接入 OpenVINO 推理后端。

## 运行第一阶段检测

```bash
python scripts/vision/01_test_yolo_image.py
```

默认配置文件是 `configs/vision/detection.yaml`。

当前目录只保留第一阶段实际会用到的文件，避免过多空文件干扰查找。
