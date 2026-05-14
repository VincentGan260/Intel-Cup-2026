# 环境说明

## 1. 目前DK-2500的基本情况

- Ubuntu 25.10  
- Python 3.13
- OpenVINO 2026.1  
- AnaConda 2025.12-2
- No Bluetooth
- No WLAN

## 2. `environment/` 和 `configs/` 有什么区别？

它们解决的是两类不同问题，不要混用。

- **`environment/` 里的 `*.yml`**：描述 **Python 运行环境**（用 Conda 装哪些包、Python 版本等），用来在一台新电脑上 **复现依赖**。例如 `environment.train.win.yml` 对应 Windows 训练机环境。当前有的文件里依赖列表可能还是空的，需要你们按实际补齐。

- **`configs/` 里的 `*.yaml`**：描述 **程序运行参数**（用哪个模型、输入图片路径、置信度阈值、设备、输出目录名等），用来 **同一套代码在不同场景下换配置**。例如 `configs/vision/detection.yaml` 只服务目标检测脚本。

简单记：

```text
environment → 这台机器能不能跑起来（装什么包）
configs       → 跑的时候用什么参数（模型、阈值、路径）
```