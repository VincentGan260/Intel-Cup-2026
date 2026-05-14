## 一、电脑上测试

第一阶段：目标检测
从YOLO26n开始
1. 配好 Python 环境：`ultralytics、opencv、openvino`。
2. 用 YOLOv8n 检测一张图片，确认能框出人和车。
3. 用摄像头实时检测，确认能实时框出前方行人、车辆、电动车。
4. 只保留项目需要的类别：

```text
person → 行人
bicycle / motorcycle → 非机动车
car / bus / truck → 机动车
```

5. 做一个简单的视觉风险判断：

```text
目标越靠近画面中心 → 风险越高
目标框越大 → 说明可能越近 → 风险越高
行人、机动车、非机动车 → 作为主要危险目标
```

这一阶段的目标是：**先做出一个能实时识别人车的 demo**。

---

二、再做语义分割，识别道路区域

目标检测跑通之后，再做语义分割。

第一版：
YOLO26n + road-segmentation-adas-0001
目标：快速做出检测 + 道路 mask + 风险判断 demo

第二版：
YOLO26n + PIDNet-S
目标：提升骑行场景道路分割效果

你不需要一开始分很多类别，先做最简单的：

```text
可行驶区域 / 非可行驶区域
```

要做的事：

1. 找一个现成的道路分割模型先跑通。
2. 输入骑行画面，输出道路 mask。
3. 判断 YOLO 检测框的底部中心点是否落在道路区域里。

逻辑大概是：

```text
检测到行人/车辆
↓
看它是否在可行驶区域内
↓
如果在道路区域内，风险提高
如果不在道路区域内，风险降低
```

这一阶段的目标是：**让系统不只是“看到人车就报警”，而是判断目标是否真的进入骑行通道**。

---

第三阶段：把检测和分割融合起来

你最终视觉模块要输出类似这样的信息：

```json
{
  "class": "person",
  "confidence": 0.91,
  "bbox": [320, 180, 410, 420],
  "in_drivable_area": true,
  "visual_risk": 0.82
}
```

也就是告诉队友：

```text
前方有什么目标
目标在哪里
目标是否在可行驶区域
视觉风险分数是多少
```
视觉风险大致思路：
道路面积大 + 前方没人车 → 风险低
道路面积大 + 前方车很近 → 风险仍然高
道路面积小 + 没有障碍物 → 不一定高风险
道路面积小 + 行人横穿 → 风险明显升高


你不用一开始负责完整风险融合，但你要给风险融合模块提供可靠的视觉结果。

---

## 二、再用 OpenVINO 部署加速

功能跑通后，再做 OpenVINO。

顺序是：

```text
PyTorch 模型
→ 导出 ONNX / OpenVINO
→ 在 Intel DK-2500 上运行
→ 测 CPU / GPU / NPU 哪个最快
→ 再考虑 FP16 / INT8 优化
```
IR 模型就是把 PyTorch / ONNX / TensorFlow 等模型转换成 OpenVINO 更容易高效推理的格式。

OpenVINO 官方文档说明，IR 通常由两个文件组成：.xml 和 .bin。其中 .xml 描述模型结构，.bin 保存模型权重和二进制参数。

.xml：模型结构
比如输入尺寸、卷积层、激活函数、各层连接关系等

.bin：模型参数
比如卷积核权重、bias、BN 参数等




你要做的事：

1. 把 YOLO 导出 OpenVINO：

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="openvino")
```

2. 用 OpenVINO 跑检测。
3. 用 `benchmark_app` 测试不同设备：

```bash
benchmark_app -m model.xml -d CPU -hint latency
benchmark_app -m model.xml -d GPU -hint latency
benchmark_app -m model.xml -d AUTO -hint latency
```

如果支持 NPU，再测：

```bash
benchmark_app -m model.xml -d NPU -hint latency
```

4. 记录性能表：

```text
模型
设备
延迟
FPS
CPU 占用率
是否稳定
```

这一阶段的目标是：**证明你确实调用了 Intel 开发板性能，而不是只在普通 Python 里跑模型**。

---

## 五、做工程优化

等模型能跑之后，再做这些优化：

```text
目标检测每帧跑
道路分割每 2～3 帧跑一次
风险融合每帧更新
摄像头队列只保留最新帧
不要处理过期画面
```

推荐分工：

```text
GPU / NPU：跑视觉模型
CPU：处理摄像头、雷达、IMU、GPS、风险融合
```

先不要一开始死磕 INT8。正确顺序是：

```text
先 FP32 跑通
再 FP16 加速
最后 INT8 量化
```

---

## 六、最后再考虑训练和微调

第一版可以不训练，直接用 YOLO 预训练模型。

等系统流程跑通后，再做：

```text
自采骑行图片/视频
标注人、车、电动车
用 YOLOv8n / YOLO11n 微调
采集道路画面
微调道路分割模型
用真实骑行数据做 INT8 校准
```

这一步是为了提高你们自己场景下的准确率，尤其是：

```text
逆光
弱光
颠簸
远处小目标
电动车/摩托车混淆
道路边界不清楚
```

---

## 你现在最应该立刻做的 5 件事

按顺序来：

1. **安装环境**：Python + Ultralytics + OpenCV + OpenVINO。
2. **跑通 YOLOv8n 图片检测**。
3. **跑通 YOLOv8n 摄像头实时检测**。
4. **筛选人、车、电动车相关类别，并计算简单视觉风险分数**。
5. **再开始做道路语义分割，不要一开始就被分割卡住**。

一句话总结：

> 你现在先把 **YOLO 目标检测 demo** 做出来；然后加入 **道路分割 mask**；再做 **检测 + 分割融合风险判断**；最后再用 **OpenVINO 在 DK-2500 上加速和测试性能**。
