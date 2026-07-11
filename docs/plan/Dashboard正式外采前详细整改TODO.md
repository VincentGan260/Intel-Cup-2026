# Dashboard 正式外采前详细整改 TODO

> 目标：把当前“能实时显示并写出样本”的原型，整改为可直接用于 GT-MRFN 数据准备的稳定采集系统。
>
> 当前测试结论：相机、LD2451、GPS串口、目标检测、语义分割、Dashboard和统一Recorder已跑通；现有数据只作为管线测试集。完成本文 P0 并通过60秒短采验收后，才能开始正式长时间采集。

## 当前实现状态（本轮代码整改）

- [x] 相机改为单一后台采集线程，页面、视觉和Recorder共享最新帧、`camera_frame_id`与采集时间。
- [x] LD2451每次消费时排空完整帧，仅使用最新雷达帧，避免20 Hz雷达在低频视觉循环中持续积压。
- [x] 雷达/GPS同步差改用数据实际到达主机的单调时间，不再用`read_once()`调用结束时间冒充样本时间。
- [x] 超过同步阈值的雷达数据不再参与视觉—雷达关联。
- [x] 正式`--record`默认等待GPS有效定位；室内测试必须显式使用`--skip-gps-fix`。
- [x] Recorder关闭前等待状态写线程真正退出，避免关闭文件后的并发写入。
- [x] 增加磁盘余量门槛、Git commit、模型/配置SHA256和静态受控场景`--risk-label`。
- [x] GT-MRFN特征schema升至v2并删除`max_visual_risk`。
- [x] 新增`build_gt_mrfn_dataset.py`，可从通过质检且有标签的session生成N=5窗口。
- [x] mock录制→质量检查→窗口化端到端回归通过。
- [ ] 尚需DK2500真机完成60秒室内动态短采、60秒室外GPS短采、雷达—相机标定和异构设备基准；这些项目不能在无硬件开发机上代验收。

## 一、最终数据定义（先冻结，后改代码）

### 1.1 每个样本必须保存的内容

- [x] `sample_id`：session内连续递增，从0开始，禁止重复。
- [x] `frame`：原图路径、宽、高、是否保存成功。
- [x] `vision`：`valid`、障碍物总数、行人数、车辆数、可行驶路面占比、检测框列表、类别、置信度、是否落在可行驶区域。
- [x] `radar`：`valid`、目标列表、最近距离、最小TTC；每个目标保留距离、相对速度、方位角和置信度。
- [x] `gps`：`valid`、速度、经纬度、定位质量；不把卫星数作为训练特征。
- [x] `fusion`：视觉—雷达关联数量及对象列表；保留来源、检测框、距离、速度/角度、TTC和是否在路面，不保存视觉风险分数。
- [x] `timestamps`：各环节独立单调时间戳、同步差值和端到端处理延迟。
- [x] `label`：先允许为空，但必须预留场景/事件/风险标签字段，后续不能破坏schema。

### 1.2 综合风险网络的固定输入

- [x] 视觉：`obstacle_count`、`person_count`、`vehicle_count`、`drivable_area_ratio`、视觉模态`valid`。
- [x] 雷达：`nearest_distance_m`、最近目标`relative_speed_mps`、目标置信度、`min_ttc`、雷达模态`valid`。
- [x] GPS：`speed_kmh`、GPS模态`valid`。
- [x] 明确不使用：原始图像像素、预计算视觉风险、GPS卫星数、IMU字段。
- [x] 原图仍必须保留，用于视觉模型更新后离线重新推理和纠错。

## 二、P0：正式外采前必须完成

### 2.1 独立采集时间戳

涉及文件：

- `run_dashboard.py`
- `src/dashboard/real_sensor_state.py`
- `src/dashboard/dashboard_recorder.py`
- 必要时修改 `src/dashboard/frame_producer.py`

实现任务：

- [x] 相机成功获得一帧后立即记录 `frame_capture_monotonic_ns`，不能等视觉推理结束后再补时间。
- [x] 雷达调用 `read_once()` 前后记录 `radar_read_start_monotonic_ns`、`radar_read_end_monotonic_ns`。
- [x] GPS调用 `read_once()` 前后记录 `gps_read_start_monotonic_ns`、`gps_read_end_monotonic_ns`。
- [x] 视觉推理前后记录 `vision_start_monotonic_ns`、`vision_finish_monotonic_ns`。
- [x] JSON序列化开始前记录 `record_write_monotonic_ns`。
- [x] 保留 `wall_time_ns` 仅用于查看真实日期时间；传感器对齐只使用同一主机的 `monotonic_ns`。
- [x] `relative_ms`以session的`started_monotonic_ns`为原点，主时间使用相机采集时间而不是写盘时间。

建议样本结构：

```json
{
  "timestamps": {
    "frame_capture_monotonic_ns": 0,
    "radar_read_start_monotonic_ns": 0,
    "radar_read_end_monotonic_ns": 0,
    "radar_sample_monotonic_ns": 0,
    "gps_read_start_monotonic_ns": 0,
    "gps_read_end_monotonic_ns": 0,
    "gps_sample_monotonic_ns": 0,
    "vision_start_monotonic_ns": 0,
    "vision_finish_monotonic_ns": 0,
    "record_write_monotonic_ns": 0,
    "radar_delta_ms": 0.0,
    "gps_delta_ms": 0.0,
    "vision_latency_ms": 0.0,
    "end_to_end_latency_ms": 0.0
  }
}
```

验收：

- [ ] 所有单调时间戳随`sample_id`递增。
- [ ] `vision_finish - vision_start`与`vision_inference_ms`基本一致。
- [ ] `record_write - frame_capture`能反映整轮端到端延迟。
- [ ] 不再把推理完成后的写盘时刻解释为画面采集时刻。

### 2.2 同步策略与过期判定

- [x] 明确相机是样本主时间轴。
- [x] 计算雷达时间相对相机的绝对差 `radar_delta_ms`。
- [x] 计算GPS时间相对相机的绝对差 `gps_delta_ms`。
- [x] 第一版建议阈值：雷达不超过100 ms，GPS不超过1000 ms；阈值写入配置和`session.json`，禁止散落在代码中。
- [x] 超阈值时保留原数据用于排查，但该模态训练`valid=false`。
- [x] 不允许用未来很久或上一轮过旧数据静默填充当前相机帧。
- [ ] 后续离线同步脚本使用相同阈值与时间轴定义。

验收：

- [ ] 质量报告能输出雷达/GPS同步差的median、P95和max。
- [ ] 人为暂停某路传感器后，该路会及时变为invalid。

### 2.3 修正循环节拍与有效帧率

涉及文件：`run_dashboard.py` 的 `dashboard_state_loop()`。

- [x] 用`time.monotonic()`记录每轮开始和结束。
- [x] 将固定 `time.sleep(interval)` 改为只等待剩余周期：`remaining = interval - elapsed`。
- [x] `remaining <= 0`时立即进入下一轮，不额外等待。
- [x] 停止等待使用`stop_event.wait(remaining)`，保证`Ctrl+C`能快速退出。
- [x] 记录实际样本间隔，而不是只显示配置的`state_hz`。
- [x] 前端显示最近有效采样率和视觉P50/P95延迟。

验收：

- [ ] 当前约443 ms推理时，不再额外固定等待200 ms。
- [ ] 60秒样本数与真实推理能力一致，无明显周期性大间断。
- [ ] 目标先定稳定4～5 Hz；若DK2500实测只能约2.5 Hz，训练窗口和报告必须使用真实频率。

### 2.4 关闭重复写盘并切换FP16检测

涉及文件：`configs/vision/detection.yaml`。

- [x] 将`output.save`从`true`改为`false`，避免每次推理重复写`runs/detect/`。
- [x] 将检测模型切换为`models/yolo26n_v2_fp16_openvino_model`。
- [ ] 启动后确认日志不再每帧打印`Results saved to ...`。
- [ ] 记录FP32与FP16在同一批图片上的检测数量、置信度差异和平均/P95延迟。
- [ ] 若FP16模型无法加载，显式报错并停止正式录制，不静默切回不同模型。

验收：

- [ ] 统一Recorder是正式session唯一逐帧写盘者。
- [x] `session.json`保存实际检测模型路径和配置版本。

### 2.5 自动质量检查

建议新增：`scripts/check_dashboard_recording.py`。

- [x] 检查`session.json`和`samples.jsonl`存在且可解析。
- [x] 检查`sample_id`连续、时间戳单调、最终`sample_count`与JSONL行数一致。
- [x] 检查每条有效frame对应图片真实存在。
- [x] 输出frame、vision、radar、gps、fusion的总数与valid比例。
- [x] 输出实际采样率、median/P95/max样本间隔。
- [x] 输出视觉推理延迟P50/P95/max。
- [x] 输出雷达/GPS同步差P50/P95/max和超阈值数量。
- [x] 输出视觉空检测比例、可行驶路面占比范围及异常值数量。
- [x] 输出雷达有目标帧比例、视觉—雷达匹配帧比例。
- [x] 发现缺图、时间倒退、JSON损坏或全部视觉无效时返回非0退出码。
- [x] Dashboard停止录制后自动运行检查并打印报告路径。

验收：

```bash
python scripts/check_dashboard_recording.py data/recordings/具体session目录
```

- [ ] 60秒短采报告无结构性错误。

### 2.6 风险监督标签方案

- [x] 冻结标签：`0=low`、`1=mid`、`2=high`，同时允许保存人工备注。
- [x] 每个session记录`scene`、`operator`、`route`、`weather`、`road_condition`、`group_id`。
- [ ] 单独建立事件标注文件，记录`event_id`、开始/结束时间、风险等级、事件类型和复核人。
- [ ] 事件类型至少覆盖：正常行驶、静态障碍、目标靠近、横穿、模拟急停。
- [ ] 不直接把现有规则风险无审核地当真值；规则只能做初始建议，低/中/高边界必须人工复核。
- [ ] 数据划分按`group_id/session`进行，同一连续片段不能跨训练、验证、测试集。

验收：

- [ ] 任意训练窗口都能追溯到原session、原图、事件和标签依据。
- [ ] 不允许只有输入特征而没有`y`的“训练集”。

## 三、P1：首批60秒短采后优化

### 3.1 DK2500异构设备基准

- [ ] 使用项目已有设备测试脚本确认OpenVINO可用设备列表。
- [ ] 分别测试检测CPU/GPU/NPU/AUTO。
- [ ] 分别测试分割CPU/GPU/NPU/AUTO。
- [ ] 测试优先组合：检测NPU+分割GPU、检测GPU+分割GPU、检测CPU+分割GPU。
- [ ] 每种组合至少预热10次、测试50次，记录平均、P50、P95和失败情况。
- [ ] 根据真实结果固定设备，不凭设备名称假定NPU最快。
- [ ] 实际设备和回退路径写入`session.json`。

### 3.2 相机采集与推理解耦

- [x] `CameraFrameProducer`使用独立后台线程持续读取摄像头。
- [x] 缓存最新帧、frame_id和采集时间戳，消费者读取副本。
- [x] Dashboard视频、视觉推理和Recorder共享同一最新帧，不重复调用底层`cap.read()`争抢帧。
- [x] 推理慢时丢弃过时帧，只处理最新帧，不建立无界队列。
- [ ] 记录相机原始采集率、推理消费率和被跳过帧数。

验收：

- [ ] 页面视频持续流畅，训练样本不出现越来越大的处理延迟。
- [ ] 采集10分钟内存占用不持续增长。

### 3.3 雷达—相机标定和关联

- [ ] 测量相机水平视场角，替换默认估计值。
- [ ] 确认雷达前装后的左右角符号，记录是否需要`flip_radar_angle`。
- [ ] 测量雷达中心线相对相机中心的角度偏置。
- [ ] 记录安装高度、水平偏移和俯仰角。
- [ ] 用单目标在左/中/右、近/中/远位置采集标定片段。
- [ ] 调整`assoc_angle_tol_deg`，避免多目标时错误绑定。
- [ ] 在线关联用于现场预览；训练前保留按独立时间戳和最终标定参数离线重算能力。

验收：

- [ ] 随机抽查至少20个有关联目标的帧，人工确认检测框和雷达方位对应正确。

### 3.4 磁盘与备份

- [x] 确认`data/recordings/`被`.gitignore`忽略。
- [x] 启动前检查剩余磁盘空间，低于安全阈值拒绝长时间录制。
- [ ] 实测1分钟session大小，估算每小时空间需求。
- [ ] 每个正式session结束后复制到移动硬盘或另一台电脑。
- [ ] 复制后校验文件数量或哈希，再允许清理DK2500本地数据。
- [ ] 不使用`git add -f data/recordings`上传原始数据。

## 四、P2：训练数据生成

### 4.1 窗口化

- [x] 新增脚本把逐样本JSONL转换为GT-MRFN窗口。
- [x] 窗口长度先使用`N=5`，但跨度必须由真实时间戳决定。
- [ ] 若采样约5 Hz，5帧窗口约1秒；若约2.5 Hz，则约2秒，必须在报告中如实记录。
- [x] 模态缺失时数值填0且valid位为0，不能把旧数据伪装成当前有效数据。
- [x] 输出`X_vision`、`X_radar`、`X_gps`、各模态`valid_*`、`y`、`group_id`。

### 4.2 可复现性

- [x] `session.json`增加代码commit、模型路径、模型文件哈希、配置文件哈希。
- [ ] 保存采集程序版本、特征schema版本、标定版本和标签版本。
- [x] 训练集只读取通过质量检查的session。
- [ ] 保存训练/验证/测试的session清单，禁止按单帧随机打散。

## 五、正式外采放行检查表

### 5.1 室内检查

- [ ] 摄像头、雷达、GPS串口和视觉管线全部成功启动。
- [ ] 页面显示camera/vision/radar/gps为real或GPS未定位状态，不出现mock。
- [ ] 连续运行60秒没有`StateLoop`异常。
- [ ] 动态目标出现时雷达targets非空，视觉检测数量发生变化。
- [ ] `Ctrl+C`后传感器安全关闭、Recorder关闭并写回最终`sample_count`。

### 5.2 室外短采

- [ ] GPS变为`valid=true`且`fix_quality>0`后再计入正式数据。
- [ ] 录制60秒并运行自动质量检查。
- [ ] 图片数与JSONL样本数一致，无缺图。
- [ ] 时间戳单调，同步差在阈值内，视觉推理延迟无持续恶化。
- [ ] 人工查看至少20帧：原图、检测框、路面占比、雷达目标和关联结果合理。

### 5.3 放行标准

- [ ] P0全部完成。
- [ ] 60秒短采无结构性问题。
- [ ] 有明确风险标签流程和现场记录人。
- [ ] 磁盘空间与备份方案确认。
- [ ] 满足以上条件后，才开始正式长时间采集。

## 六、推荐执行顺序

1. 独立时间戳与同步差。
2. 循环节拍修正。
3. 关闭YOLO重复写盘并切FP16。
4. 自动质量检查脚本。
5. 60秒室内动态测试。
6. 室外GPS短采。
7. 相机—雷达标定。
8. 设备组合性能基准。
9. 风险事件标注与窗口化。
10. 正式采集与备份。
