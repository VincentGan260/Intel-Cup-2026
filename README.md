# 2026-Intel-Cup-Undergraduate-Electronic-Design-Contest

## YOLO与OpenVIVO相关参考链接

### 3 倍提速！Ultralytics YOLO 模型 OpenVINO 全流程部署指南
https://blog.csdn.net/gitblog_00505/article/details/151413307

### CPU 也能跑模型：OpenVINO 模型部署入门教程
https://zhuanlan.zhihu.com/p/5097186050

### YOLO26 官网
https://docs.ultralytics.com/zh/models/yolo26/

### YOLO 使用 OpenVINO 官方文档
https://docs.ultralytics.com/zh/integrations/openvino/#inference-with-openvino-runtime

## Team Docs 用于指导团队规范协作
- [Team Development Guide](docs/guide/team_development_guide.md)
- [Vision Development Guide](docs/guide/vision_development_guide.md)

## 当前视觉开发入口

- 视觉源码目录：`src/vision/`
- 视觉调试脚本：`scripts/vision/`
- 视觉配置目录：`configs/vision/`
- 本地 OpenVINO IR：`models/openvino/`（具体子路径见 `configs/vision/segmentation_openvino.yaml`）；YOLO 权重路径见 `configs/vision/detection.yaml`
- 视觉默认可视化根目录：`runs/vision/`（检测在 `detect/`，分割在 `segmentation_openvino/` 等，见各 `configs/vision/*.yaml`）
- 视觉闭环配置：`configs/vision/vision_pipeline.yaml`
- 项目自建导出占位：`outputs/vision/`（例如 JSON）
