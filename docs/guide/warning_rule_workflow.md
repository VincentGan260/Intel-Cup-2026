# 风险规则配置、验证与回放使用说明

本文说明比赛预警规则如何配置、运行、验证和标定。当前普通笔记本用于规则与时序测试；视觉 NPU/GPU 性能、真实外设周期和实车阈值必须在 DK2500 及车辆上验证。

## 1. 这套工具解决什么问题

风险规则工程化不改变“危险如何定义”，而是保证规则具备以下能力：

- 参数集中管理，避免同一阈值散落在多个 Python 文件中。
- 每次运行记录规则版本、配置哈希和实际生效参数。
- 同一段录制数据可以用不同参数重复计算，公平比较修改前后结果。
- 可以注入雷达、视觉或 GPS 单独失效，检查降级行为。
- 自动检查参数矛盾、时间戳错误和录制质量。

当前比赛预警结果不使用旧版加权模型 `src/fusion/risk_model.py`。旧模型仅为兼容测试保留。

## 2. 主要文件

| 文件 | 用途 |
|---|---|
| `configs/warning_rules.yaml` | 当前规则默认参数、版本和标定状态 |
| `src/fusion/warning_config.py` | 加载并校验配置，计算配置哈希 |
| `src/fusion/physical_risk_rule.py` | 雷达 TTC 与横向 point gate 规则 |
| `src/fusion/vision_warning_rule.py` | 视觉路径、接近程度和视觉 tau 规则 |
| `src/fusion/imu_warning_rule.py` | IMU转弯补偿、横倾残差、外倾预测与连续风险 |
| `src/fusion/warning_arbiter.py` | 雷达、视觉、IMU有效性检查与 max 仲裁 |
| `src/fusion/warning_system.py` | 连续分数、等级状态、下降保持和时间戳对齐 |
| `scripts/test_warning_scenario_matrix.py` | 无外设理论场景矩阵 |
| `scripts/replay_warning_session.py` | 录制回放与设备故障注入 |
| `scripts/check_dashboard_recording.py` | 录制完整性、同步和风险时间戳质检 |

## 3. 环境约定

Windows 普通笔记本统一使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe --version
```

不要用全局 Anaconda 的 `python` 判断项目依赖是否完整。普通笔记本没有 DK2500 的 NPU 时，正式视觉配置初始化失败并进入降级状态是预期行为，不代表部署配置错误。

## 4. 配置文件结构

打开 `configs/warning_rules.yaml` 可以看到以下分区：

| 分区 | 内容 |
|---|---|
| `score_contract` | 中风险和高风险连续分数边界，目前固定为 `0.35 / 0.70` |
| `radar` | 车体宽度、安装偏移、point gate、TTC参考和雷达范围 |
| `vision` | 视觉走廊、路径策略、远近位置、tau和时序关联 |
| `gps` | 视觉单帧接近风险的速度修正范围 |
| `imu` | 安装零偏、转弯补偿、横倾残差、外倾速度和预测时间 |
| `freshness` | 雷达、视觉、GPS、IMU过期时间和雷达通信看门狗 |
| `state` | 风险下降保持时间 |

配置顶部字段：

- `schema_version`：配置结构版本，不是规则效果版本。
- `version`：本次规则版本，修改候选参数时必须同步修改。
- `calibration_status`：标定状态。当前为 `provisional_pending_vehicle_test`，表示等待实车验证。

`configured_warning_range_m` 当前为 `100.0`，对应项目既有硬件记录中的
LD2451 V1.03 默认/协议最大探测距离。该值只过滤雷达已经上报的目标，不会扩大
雷达的实际探测能力。

## 5. 启动 Dashboard

正式入口会直接读取版本化配置，一般无需额外传入量程：

```powershell
.\.venv\Scripts\python.exe start_demo.py
```

正式入口默认启用视觉、雷达、GPS、IMU、比赛风险规则和真实DRV2605马达，
状态循环为`20 Hz`，并以`bike-001`向`http://124.70.108.34`上传状态和分段视频。
离线诊断使用`--disable-cloud`；不希望驱动马达时使用`--disable-motor`。
IMU安装方向或USB位置变化后，不得直接沿用当前
`roll_offset_deg`、`pitch_offset_deg`和`turn_sign`。

参数优先级为：

1. 命令行显式参数。
2. `--warning-config` 指定的 YAML。
3. 程序兼容默认值。

命令行覆盖适合临时排查。正式标定结果应写入新版本 YAML，以便回放和答辩追溯。Dashboard 状态和新录制会话会保存配置版本、SHA-256 哈希以及实际生效参数。

## 6. 没有外设时如何验证

先运行配置、分数契约、单模态规则、融合和场景测试：

```powershell
.\.venv\Scripts\python.exe scripts/test_warning_config.py
.\.venv\Scripts\python.exe scripts/test_risk_score_contract.py
.\.venv\Scripts\python.exe scripts/test_gps_risk_context.py
.\.venv\Scripts\python.exe scripts/test_imu_warning_rule.py
.\.venv\Scripts\python.exe scripts/test_competition_risk_rule.py
.\.venv\Scripts\python.exe scripts/test_multimodal_warning_system.py
.\.venv\Scripts\python.exe scripts/test_warning_scenario_matrix.py
.\.venv\Scripts\python.exe scripts/test_warning_session_replay.py
```

理论场景矩阵覆盖：

- 雷达恰好位于中风险、高风险边界。
- 静止目标、路径外目标和空旷道路。
- 视觉单独给出中风险或高风险。
- 雷达与视觉结论冲突。
- 雷达、视觉、GPS、IMU失效或过期。
- 风险立即升级和延迟下降。

这些测试证明程序行为符合当前规则，不代表临时阈值已经获得实车准确率。

## 7. 修改一个候选参数

不要直接覆盖基线。先复制候选配置：

```powershell
Copy-Item configs/warning_rules.yaml configs/warning_rules_margin_015.yaml
```

然后执行：

1. 修改候选文件中的 `version`。
2. 每轮只修改一个参数或一条判断。
3. 运行第 6 节全部相关测试。
4. 用同一段录制分别回放基线和候选配置。
5. 比较首次报警时刻、等级、连续分数、误报、漏报和等级变化次数。
6. 没有明确改善时回退候选，不顺带修改第二个参数。

禁止把 `urgent_reference_s` 描述为制动安全时间。它只是骑手需要立即采取行动的紧迫参考。

## 8. 录制完成后先做质量检查

```powershell
.\.venv\Scripts\python.exe scripts/check_dashboard_recording.py `
  data/recordings/SESSION
```

检查内容包括：

- 样本编号和相机主时间线是否严格递增。
- 雷达、GPS、视觉采集与处理时间是否完整。
- 图片、JSONL和会话元数据是否一致。
- 数据同步超时比例和视觉推理延迟。
- 风险决策时间是否严格递增。
- 每个风险证据是否满足 `采集时间 <= 完成时间 <= 决策时间`。
- 降级保持是否保留原风险证据时间。

报告写入会话目录下的 `quality_report.json`。质量检查失败的录制不能直接用于调参结论。

## 9. 回放同一段录制

基线回放：

```powershell
.\.venv\Scripts\python.exe scripts/replay_warning_session.py `
  data/recordings/SESSION `
  --warning-config configs/warning_rules.yaml `
  --warning-range-m 20 `
  --output outputs/replay_baseline.json
```

候选参数回放：

```powershell
.\.venv\Scripts\python.exe scripts/replay_warning_session.py `
  data/recordings/SESSION `
  --warning-config configs/warning_rules_margin_015.yaml `
  --warning-range-m 20 `
  --output outputs/replay_margin_015.json
```

新录制通常已经保存雷达范围，不需要 `--warning-range-m`。旧录制没有保存 APP 范围时必须显式提供。

回放报告包含：

- `level_counts`：低、中、高和 unknown 样本数量。
- `transition_count`：等级变化次数，可辅助观察抖动。
- 每个样本的决策时间、风险等级、连续分数和告警原因。
- 每个模态在该时刻的状态。

自动统计不能替代人工场景标签。最终仍要判断每个场景“是否应该报警”。

## 10. 注入设备失效

参数中的数字是从零开始的样本序号，表示从该样本起让设备失效：

```powershell
.\.venv\Scripts\python.exe scripts/replay_warning_session.py `
  data/recordings/SESSION --fail-radar-at 100

.\.venv\Scripts\python.exe scripts/replay_warning_session.py `
  data/recordings/SESSION --fail-vision-at 100

.\.venv\Scripts\python.exe scripts/replay_warning_session.py `
  data/recordings/SESSION --fail-gps-at 100

.\.venv\Scripts\python.exe scripts/replay_warning_session.py `
  data/recordings/SESSION --fail-imu-at 100
```

预期行为：

- 雷达失效：视觉仍可输出风险，系统状态为 degraded。
- 视觉失效：雷达 TTC 高风险通道继续立即响应。
- GPS失效：速度修正回到 `1.0`，不改变雷达 TTC。
- IMU失效：雷达和视觉继续独立预警，系统状态为 degraded。
- GPS不足以支持转弯补偿：IMU仍输出连续分数，但最高限制为中风险。
- 雷达和视觉均不可用：输出 unknown，不能伪装成低风险。

## 11. 如何理解时间戳字段

- `risk_decision_monotonic_ns`：本次仲裁作出结果的时刻。
- `risk_effective_updated_monotonic_ns`：当前展示风险真正生效的时刻。
- `risk_source_timing`：当前展示风险实际引用的模态证据时间。
- `raw_risk_source_timing`：本次原始仲裁使用的模态证据时间。
- `risk_timestamp_alignment=as_of_latest_fresh`：风险对应当前最新有效证据。
- `risk_timestamp_alignment=downgrade_held`：当前仍展示上一风险，下降保持尚未结束。
- `risk_timestamp_alignment=no_usable_warning_modality`：当前没有可用风险模态。

所有字段使用同一主机的单调时钟，只用于计算先后关系与延迟，不应当转换成日期时间展示。

## 12. 上车后的正式标定顺序

1. 核对 LD2451 APP 探测范围和雷达速度正负语义。
2. 测量各设备更新周期、排队延迟和 P95/P99。
3. 固定场景、路线和设备安装位置，建立基线录制。
4. 先验证IMU零偏、左右转弯符号和正常转弯补偿，再验证视觉路径误报。
5. 再调雷达 point gate、`urgent_reference_s`，最后调 `release_hold_ms`。
6. 每个场景重复 3至5 次，只保留稳定改善的候选版本。
7. 记录参数版本、期望等级、实际等级、首次报警时间、最小 TTC、最近距离、告警原因及人工备注。

需要实车确认的参数统一记录在 `docs/plan/risk_rule_calibration_checklist.md`，不得把当前近似值宣传为最终安全结论。
