# 宏观目标
现在先做这几件事：

1. 优先下载 BDD100K，筛 300～500 张道路、人、车相关图片。
2. 下载 Cityscapes，筛 100～200 张用于道路分割测试。
3. 下载 IDD Lite，筛 50～100 张复杂道路图片。
4. 下载 ACDC 的 night/rain/fog，筛 150～300 张弱光/恶劣条件图片。
5. 自采车把视角视频 5～10 段，抽帧 300～800 张。
6. 所有完整数据集放 D:\datasets，不提交 Git。
7. 项目仓库只保存 data/README.md、samples、splits、dataset_configs 和数据说明文档。

最终目标不是“数据越多越好”，而是先形成一套覆盖：

正常道路
行人横穿
前方车辆
非机动车
弱光/逆光
复杂道路
真实车把视角

的小而高质量测试集。

具体细节如下：

# 推荐数据集
1. BDD100K：主数据集，优先级最高

用途：目标检测 + 可行驶区域分割 + 场景泛化测试。

BDD100K 是大规模驾驶场景数据集，包含道路目标检测、可行驶区域分割、语义分割、车道线等任务，比较适合我们这种前向交通环境感知任务。官方工具仓库也说明 BDD100K 包含 image tagging、lane detection、drivable area segmentation、road object detection、semantic segmentation 等任务。

建议下载内容：

1.目标检测：
BDD100K Images + Detection Labels

2.道路分割：
BDD100K Drivable Area

可选：
BDD100K Semantic Segmentation

处理方式：

不要一开始全量使用。
先随机抽取一小部分 + 人工筛选。

建议先筛：

白天正常道路：100 张
傍晚/弱光：50 张
路口/斑马线：50 张
行人明显出现：50 张
自行车/摩托车/电动车类似目标：50 张
前方车辆明显出现：50 张

第一阶段先凑 300～500 张图片用于测试模型效果，不要一开始下载后就全量训练。

2. Cityscapes：语义分割基准数据集

用途：测试和训练道路语义分割模型，例如 PIDNet-S。

Cityscapes 是经典城市街景语义分割数据集，官网说明它包含 5000 张高质量标注图像、20000 张粗标注图像，覆盖 50 个城市。

建议下载内容：

leftImg8bit_trainvaltest
gtFine_trainvaltest

处理方式：

不用于目标检测主训练。
主要用于语义分割模型测试 / 微调 / 对比。

建议先用：

val 集中抽 100～200 张道路清楚的图
把 road / sidewalk / terrain 等类别整理成“可行驶 / 不可行驶”二分类

注意：Cityscapes 是汽车视角，和骑手车把视角不完全一致，所以它适合做分割模型基准，不适合作为最终验收数据。

3. IDD / IDD Lite：复杂道路场景补充数据集

用途：测试复杂交通、非规则道路、道路边界不清晰的情况。

IDD 官方介绍中说明它面向非结构化道路场景，包含约 10000 张精细标注图像、34 个类别，采集自印度道路。

建议优先看：

IDD Lite
IDD Segmentation
IDD Detection

处理方式：

优先下载 IDD Lite。
如果 IDD Lite 跑通后觉得有价值，再考虑 IDD Segmentation。

使用场景：

道路边界混乱
行人、两轮车、汽车混杂
非机动车和机动车交织
路口规则不明显

它和国内道路不完全一致，但适合测试模型在复杂场景下会不会崩。

4. ACDC：弱光、夜晚、雨雾雪鲁棒性测试

用途：专门测试夜晚、雨天、雾天、雪天等恶劣条件。

ACDC 是 Adverse Conditions Dataset，官方说明它面向 fog、night、rain、snow 等恶劣视觉条件，下载页说明包含 train、val、test 的恶劣条件图像，共约 4006 张。

处理方式：

不作为第一主训练集。
只作为鲁棒性测试集。

建议筛选：

night：50～100 张
rain：50～100 张
fog：50 张
snow：可选，国内场景不一定需要

对我们项目最有价值的是：

night
rain
fog

因为方案书里也要求测试白天、傍晚、逆光、弱光、颠簸等复杂外部环境。

5. 自采数据：最终必须有

用途：最终验证、少量微调、INT8 量化校准。

公开数据集大多是汽车前视角，不是外卖骑手车把视角。所以最后一定要有我们自己的数据。

自采内容建议：

白天正常骑行道路
傍晚 / 弱光
逆光
路口行人横穿
前方电动车 / 自行车
前方机动车
路边停车遮挡
颠簸导致的运动模糊
障碍物逼近
前车急停模拟

处理方式：

先录视频，再抽帧。
每段视频按 1 秒 1～3 帧抽取。
不要连续保存大量重复帧。

第一批目标：

自采原始视频：5～10 段
抽帧图片：300～800 张
人工精选有效图片：200～400 张

这批数据最重要，因为它和真实部署视角最接近。

三、各数据集使用场景总结
数据集	主要用途	是否优先	处理方式
BDD100K	目标检测、可行驶区域、普通道路测试	最高	先抽样 + 人工筛选
Cityscapes	语义分割基准、PIDNet 测试	高	只取分割相关数据
IDD / IDD Lite	复杂道路、混乱交通测试	中	先用 Lite
ACDC	夜晚、雨雾、弱光鲁棒性测试	中	只筛关键恶劣场景
自采数据	最终验证、微调、INT8 校准	最高	必须人工筛选
四、每个阶段要用哪些数据
阶段 1：模型初步测试

目标：看 YOLO26n 和道路分割模型能不能基本工作。

使用：

BDD100K 抽样图片：100～200 张
Cityscapes val 抽样：50～100 张
自采车把视角图片：50～100 张

不需要训练，只做推理测试。

输出：

检测结果图
道路 mask 图
误检/漏检记录表
阶段 2：检测 + 分割融合测试

目标：测试“检测框底部中心点是否落在可行驶区域”。

使用：

BDD100K Drivable Area：100～200 张
自采车把视角：100～200 张
路口/行人横穿场景：重点筛选

输出：

bbox + road mask 叠加图
in_drivable_area 判断结果
典型错误案例

你们的视觉计划里已经明确要判断 YOLO 检测框底部中心点是否落在道路区域内，并输出目标是否在可行驶区域、视觉风险分数等信息。

阶段 3：轻量微调

目标：如果预训练模型效果不够，再微调。

检测微调用：

BDD100K Detection 中筛选 person、bicycle、motorcycle、car、bus、truck
自采数据中人工标注行人、非机动车、机动车

分割微调用：

BDD100K Drivable Area
Cityscapes road 类别
自采道路 mask 少量标注

处理方式：

公开数据集不要全量乱用。
先筛选和骑行场景接近的图。
类别统一成项目需要的简化类别。
阶段 4：鲁棒性测试

目标：测试弱光、逆光、雨雾、复杂道路下模型是否稳定。

使用：

ACDC night / rain / fog
IDD Lite / IDD Segmentation
自采弱光 / 逆光 / 颠簸视频抽帧

输出：

弱光漏检案例
道路 mask 崩溃案例
误报警案例
不同场景准确率统计
阶段 5：OpenVINO / INT8 校准

目标：部署到 Intel 板子后做量化和性能测试。

使用：

自采真实车把视角图片：300～1000 张
BDD100K 少量补充：100～300 张

注意：

INT8 校准数据必须尽量接近真实摄像头视角。
不要只用 Cityscapes 或 BDD100K。

你们计划里后续也提到要用真实骑行数据做 INT8 校准，以提高逆光、弱光、颠簸、远处小目标等场景下的表现。

五、数据集放不放在项目目录下？

结论：

完整数据集不要放进 Git 仓库，也不要提交到项目代码目录。项目里只放说明、样例、划分文件和配置。

推荐做法：

1. 完整数据放仓库外

Windows 可以放：

D:\datasets\

例如：

D:\datasets\
├─ bdd100k\
├─ cityscapes\
├─ idd\
├─ acdc\
└─ rider_self_collected\

Linux 可以放：

/home/xxx/datasets/

或者：

/data/datasets/
2. 项目目录里只放这些
Intel-Cup-2026/
├─ data/
│  ├─ README.md
│  ├─ samples/
│  │  ├─ road_test.jpg
│  │  └─ detection_test.jpg
│  ├─ splits/
│  │  ├─ detection_train.txt
│  │  ├─ detection_val.txt
│  │  ├─ segmentation_train.txt
│  │  └─ segmentation_val.txt
│  └─ dataset_configs/
│     ├─ bdd_detection.yaml
│     ├─ bdd_drivable.yaml
│     ├─ cityscapes_seg.yaml
│     └─ self_collected.yaml

说明：

data/samples/：只放少量测试图片，可以提交 Git
data/splits/：放训练/验证划分文件，可以提交 Git
data/dataset_configs/：放数据路径配置，可以提交 Git，但不要写死个人绝对路径
完整图片、视频、标签：不提交 Git

.gitignore 里加：

datasets/
data/raw/
data/processed/
data/videos/
*.mp4
*.avi
六、队友执行 To-do List
第 1 步：建立数据目录

在电脑上建立：

D:\datasets\
├─ bdd100k\
├─ cityscapes\
├─ idd\
├─ acdc\
└─ rider_self_collected\

项目仓库内建立：

data/
├─ README.md
├─ samples/
├─ splits/
└─ dataset_configs/
第 2 步：优先下载 BDD100K

先下载：

BDD100K Images
Detection Labels
Drivable Area Labels

只要能先拿到一部分 val / sample，也可以先开始筛选。

任务：

筛选 300～500 张与骑行场景相近的图
重点包含 person、bicycle、motorcycle、car、bus、truck
重点包含道路、路口、斑马线、非机动车场景

交付物：

data/splits/bdd_selected.txt
docs/dataset_notes.md 中记录筛选标准
第 3 步：下载 Cityscapes

下载：

leftImg8bit_trainvaltest
gtFine_trainvaltest

任务：

先筛 val 集 100～200 张
优先选择道路清楚、行人车辆明显的图
记录 road / sidewalk 等类别如何合并成 drivable / non-drivable

交付物：

data/splits/cityscapes_selected.txt
docs/dataset_notes.md 中记录处理方式
第 4 步：下载 IDD Lite

优先下载：

IDD Lite

任务：

用于复杂道路测试
先选 50～100 张
不用一开始全量处理

交付物：

data/splits/idd_lite_selected.txt
第 5 步：下载 ACDC

优先下载：

night
rain
fog

任务：

每类筛 50～100 张
用于弱光和恶劣条件测试

交付物：

data/splits/acdc_selected.txt
第 6 步：自采数据

任务：

采集 5～10 段车把视角视频
每段 30 秒～2 分钟
覆盖白天、傍晚、逆光、路口、行人、电动车、前车等

抽帧规则：

普通视频：每秒抽 1 帧
行人横穿/前车急停：每秒抽 3～5 帧
删除模糊严重、重复过高、没有有效道路信息的帧

交付物：

D:\datasets\rider_self_collected\videos\
D:\datasets\rider_self_collected\frames\
data/splits/self_collected_selected.txt
docs/self_collection_notes.md
第 7 步：整理标注类别

目标检测统一类别：

person
bicycle
motorcycle
car
bus
truck

项目风险类别映射：

person → pedestrian
bicycle / motorcycle → non_motor_vehicle
car / bus / truck → motor_vehicle

语义分割统一为：

drivable_area
non_drivable_area

不要一开始保留太多语义类别，否则会增加训练和后处理复杂度。

第 8 步：建立数据说明文档

在项目里写：

data/README.md
docs/dataset_notes.md
docs/self_collection_notes.md

至少记录：

数据集名称
下载来源
下载日期
许可证/使用限制
存放路径
使用任务：检测/分割/鲁棒性测试
筛选标准
当前已筛选数量
是否已标注
是否可用于训练