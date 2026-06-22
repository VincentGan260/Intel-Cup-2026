新增

configs/vision/eval.yaml —— 测试统一配置(所有路径/采样数/标签约定)
configs/vision/eval.local.yaml.example —— 个人覆盖示例
scripts/vision/check_eval_paths.py —— 路径体检脚本(随时跑,自动列出缺什么)
改动

scripts/dataset_paths.py —— 加 load_eval_config()(支持嵌套覆盖)
.gitignore —— 排除 eval.local.yaml
7 个硬编码 Mac 路径的脚本(comprehensive_eval / eval_bdd100k_speed_count / generate_report / make_subset / select_datasets / summarize_results / test_single_image)→ 改成可移植路径,现在 Windows 上能跑了
你要去填的路径(都在 eval.yaml)
配置项	填什么	现在的默认值
detection.images_dir	BDD 图片目录	datasets/bdd100k/images/100k/val
detection.labels_dir	BDD 逐图 JSON 标注目录	datasets/bdd100k/labels/100k/val
segmentation.datasets[].images_dir	Cityscapes 图片目录	datasets/cityscapes/leftImg8bit/val
segmentation.datasets[].labels_dir	Cityscapes 标注目录	datasets/cityscapes/gtFine/val
quantization.calib_images_dir	量化校准图(用 BDD 图即可)	同 detection.images_dir
路径不对就直接改 eval.yaml;如果你和队友路径不同,各自建 eval.local.yaml(不进 git)。

⚠️ 一个要你确认的标签约定(最易错):Cityscapes 我按 *_gtFine_labelIds.png(road=7)填的。如果你下载的是 *_gtFine_labelTrainIds.png,那 road=0,要改 road_value。你确认一下标注文件后缀,拿不准把 gtFine/val 下某个文件名发我,我帮你定。

哪些脚本跑什么
脚本	作用	现在能跑?
scripts/vision/check_eval_paths.py	路径体检,列出缺什么	✅ 现在
scripts/vision/03_test_pidnet_openvino.py	PIDNet 单图分割 demo + 测延迟	✅
scripts/vision/04_compare_segmenters.py	road-adas vs PIDNet 对比	✅
scripts/evaluate_accuracy.py(队友的)	旧版精度评测	⚠️ 能跑但口径有 bug,将被新脚本取代
comprehensive_eval / eval_bdd100k_speed_count 等	队友的统计脚本	✅ 改完路径能跑(部分将被取代)
检测精度评测(新)	障碍物 recall/mAP	❌ 下一步写
分割精度评测(新)	road IoU	❌ 下一步写
量化 + INT8 精度	FP32 vs INT8	❌ 下一步写
延迟基准	端侧延迟	❌ 下一步写(上板跑)
现在缺什么(体检脚本刚跑出来的)
缺的	谁来补
BDD 图片 + JSON 标注目录	你(填真实路径)
Cityscapes 图片 + 标注目录	你(填真实路径 + 确认 road_value)
检测 INT8 模型	后续量化步骤产出
PIDNet FP16 / INT8 模型	后续转换/量化产出
✅ 已就位:检测 PyTorch、检测 OpenVINO FP32、PIDNet FP32、road-adas —— 精度评测的模型基线都齐了,只差你把数据集路径填对。

怎么用

# 注意：conda run 对中文输出会崩，加 --no-capture-output，或直接用环境 python
conda run --no-capture-output -n intel python scripts/vision/check_eval_paths.py
填完路径反复跑这个体检,直到检测和分割的数据都显示 [OK],就可以让我开始写分割精度评测(不卡依赖,第一个出数)。







延迟脚本写好并验证通过了,而且意外收获:你这台开发机本身就有 Intel 核显(iGPU),所以脚本自动测出了 CPU + GPU 两组,逻辑完全跑通。

这轮做完的(都不碰数据集)
✅ FP16 转换 —— scripts/vision/convert_fp16.py,已产出 3 个 FP16 模型(yolo / pidnet / road-adas)
✅ 延迟基准脚本 —— scripts/vision/bench_latency.py,纯推理计时、p95、FPS、并发摆放,自动检测可用设备、自动跳过缺失的模型/设备

冒烟测试结果(⚠️ 仅验证逻辑,不是真实数)
这是开发机、且只跑了 warmup=3/iters=15(太少),数字别当真。重点是脚本通了。

模型	CPU	GPU(开发机弱核显)
yolo26n	~29ms / 34 FPS	~34ms / 30 FPS
PIDNet-S	~83ms / 12 FPS	~212ms / 4.7 FPS
road-adas	~35ms / 29 FPS	~32ms / 31 FPS
(开发机这颗核显比较弱,GPU 反而慢;DK-2500 的 Arc 核显 + NPU 会是另一回事——上板跑同一脚本即得真实三设备表。)

JSON 存到 runs/latency_eval/latency_results.json。

现在的局面
不碰数据集能写的主体已经写完了(FP16 + 延迟)。剩下的:

还能写(非数据集)	状态
PyTorch↔OpenVINO 一致性抽查(yolo,用样例图)	小活,可写
延迟脚本里清掉 OpenVINO 的 warning 刷屏	可顺手优化
真正卡数据的(先不写)	卡在哪
检测精度评测	BDD 数据+格式没定
分割精度评测	Cityscapes 数据没好
INT8 量化	要校准图(数据集)
