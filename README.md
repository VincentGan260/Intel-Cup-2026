# 2026-Intel-Cup-Undergraduate-Electronic-Design-Contest

## 当前正式入口

- `start_demo.py`：比赛演示入口，不写训练录制。
- `start_collection.py`：真实传感器对齐采集入口。
- `run_dashboard.py`：Dashboard底层入口，正式预警使用规则融合链路。
- `configs/warning_rules.yaml`：当前规则参数唯一共享默认配置。
- `scripts/replay_warning_session.py`：录制数据风险回放与设备故障注入。

`main_integrated.py`、`src/fusion/risk_model.py`和`configs/risk_params.yaml`仅保留给旧加权模型测试与兼容模式，不是当前比赛预警规则的参数来源。

运行环境使用：

```bash
python -m pip install -r requirements.txt
python scripts/test_warning_config.py
python scripts/test_warning_scenario_matrix.py
```

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
- [Warning Rule Workflow](docs/guide/warning_rule_workflow.md)

## 当前视觉开发入口

- 视觉源码目录：`src/vision/`
- 视觉调试脚本：`scripts/vision/`
- 视觉配置目录：`configs/vision/`
- 本地 OpenVINO IR：`models/openvino/`（具体子路径见 `configs/vision/segmentation_openvino.yaml`）；YOLO 权重路径见 `configs/vision/detection.yaml`
- 视觉默认可视化根目录：`runs/vision/`（检测在 `detect/`，分割在 `segmentation_openvino/` 等，见各 `configs/vision/*.yaml`）
- 视觉闭环配置：`configs/vision/vision_pipeline.yaml`
- 项目自建导出占位：`outputs/vision/`（例如 JSON）
