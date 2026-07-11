# 多模态数据采集指南

## 0. 推荐：一键实地测试（相机 + LD2451 + GPS，无IMU）

只做设备预检：

```bash
python scripts/field_record.py --scene desk_test --profile dk2500 --dry-run
```

等待GPS定位后录制60秒，并自动同步、检查：

```bash
python scripts/field_record.py --scene low_clear --profile dk2500 --duration 60
```

桌面测试允许GPS未定位：

```bash
python scripts/field_record.py --scene desk_test --profile dk2500 --duration 30 --skip-gps-fix
```

一键脚本默认不启动马达且不加载IMU。结束后自动生成`fusion.jsonl`和`quality_report.json`；质量检查失败时返回非0退出码。

## 1. 正式采集前桌面测试

在项目根目录使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe scripts\record_multimodal.py --mode mock --duration 10 --scene desk_test
```

真实硬件（Windows 端口读取 `configs/sensor_ports.yaml`）：

```powershell
.\.venv\Scripts\python.exe scripts\record_multimodal.py --mode real --profile windows --duration 30 --scene desk_test
```

DK2500：

```bash
python scripts/record_multimodal.py --mode real --profile dk2500 --duration 30 --scene desk_test
```

每次运行在 `data/recordings/<时间_场景>/` 创建独立目录。正式采集前先检查端口配置和相机编号，不要覆盖或手工混合两个 session。

## 2. 保存内容

- `session.json`：场景、设备配置、起止时间和记录数量。
- `frames/` + `frames.jsonl`：逐帧 JPEG 及索引。
- `radar.jsonl`、`gps.jsonl`：各路原始解析数据（当前项目不使用IMU）。
- 每条记录同时保存 Unix 纳秒、单调时钟纳秒和相对毫秒；同步只使用 `monotonic_ns`。

## 3. 离线同步与检查

```powershell
.\.venv\Scripts\python.exe scripts\sync_recording.py data\recordings\<session目录>
.\.venv\Scripts\python.exe scripts\check_recording.py data\recordings\<session目录> --require-fusion
```

同步以相机帧为时间轴，默认匹配最近的雷达（100 ms）和GPS（1000 ms）。超过阈值的数据保留但标记`valid=false`，便于重新调整阈值而无需重新采集。

检查`quality_report.json`：三路记录数不能为零、时间戳不能倒退、图片不能缺失；重点查看`fusion.valid_ratio`和`max_abs_delta_ms`。

## 4. 场景命名

推荐：`low_clear`、`mid_approach`、`high_crossing`、`high_braking`。每类至少录制 5 个独立 session，不要把一次长视频切片后伪装成多个独立场景。高风险场景只能在空场低速、假人或纸箱条件下受控复现。

## 5. 现场停止条件

- 相机无法写帧；
- 任意关键数据文件持续为空；
- 时间戳倒退；
- 雷达或GPS连续大段无效；
- 存储空间不足。

发现问题立即停止该 session，保留目录用于排错，修复后新建 session 重录。
