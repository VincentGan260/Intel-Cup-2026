# 骑行数据云端上传功能开发文档

## 1. 文档目的

本文档说明骑行安全系统中“边缘端实时数据与骑行视频上传至华为云”的设计与实现，供开发、联调、验收和项目答辩使用。

系统目标如下：

1. DK2500 边缘设备持续采集 GPS、毫米波雷达、视觉和风险评估数据。
2. 以默认 1 Hz 频率将关键骑行数据上传至华为云。
3. 复用实时摄像头画面，每 60 秒生成一段原始 MP4 视频并上传。
4. openGauss 保存结构化骑行数据和视频元数据。
5. Nginx 提供 API 入口、云端数据页面以及支持拖动播放的视频访问。
6. 云端只保留最近一小时视频，避免磁盘持续增长。

## 2. 当前完成状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| openGauss 数据库与表结构 | 已完成 | 已完成约束、索引和写入验证 |
| FastAPI 数据上传与查询接口 | 已完成 | 公网 API 已验证 |
| 一分钟视频上传接口 | 已完成 | 元数据和文件同时留存 |
| Nginx 反向代理与视频访问 | 已完成 | Range 请求返回 206，可拖动播放 |
| 最近一小时视频清理 | 已完成 | systemd timer 每分钟执行 |
| 云端数据查看页面 | 已完成 | 可查看数据和最近视频 |
| 边缘端上传模块 | 已完成 | 本地语法与载荷测试通过 |
| 云端开机自动恢复 | 已完成 | openGauss、API、清理任务依赖已配置 |
| DK2500 真实硬件端到端验证 | 待完成 | 最后集中完成实机联调和稳定性测试 |

## 3. 系统总体架构

```text
GPS ───────────┐
毫米波雷达 ────┤
视觉感知 ──────┼─> DK2500 DashboardState
风险模型 ──────┘           │
                           ├─ 1 Hz JSON ─> POST /api/ride-samples
摄像头原始帧 ──────────────└─ 60 s MP4 ─> POST /api/video-segments
                                                   │
                                      华为云 Nginx :80
                                         │         │
                                      FastAPI    /videos/
                                         │         │
                                     openGauss   视频文件
                                         │         │
                                         └─ 云端数据查看页面
```

边缘端、云端服务与存储相互解耦。Dashboard 的主循环只负责产生状态，上传模块使用独立线程处理网络请求和视频编码，因此网络波动不会直接阻塞传感器读取和实时页面刷新。

## 4. 运行环境

### 4.1 边缘端

- 硬件：DK2500
- 操作系统：Ubuntu 24.04，x86_64
- Python：Conda `intel` 环境，Python 3.14.6
- OpenCV：5.0.0
- Requests：2.34.2
- 视频编码：优先 `avc1`，其次 `H264`，最后回退到 `mp4v`

### 4.2 云端

- 平台：华为云 ECS
- 操作系统：openEuler 20.03 LTS，aarch64
- 公网地址：`124.70.108.34`
- openGauss：5.0.1，端口 26000
- 数据库：`rider_dashboard`
- Web API：FastAPI + Uvicorn，监听 `127.0.0.1:8000`
- Web 入口：Nginx，监听 80 端口
- 视频目录：`/opt/rider-cloud/videos`

## 5. 数据库设计

### 5.1 骑行数据表 `ride_samples`

该表共 17 个字段，只保留 Dashboard 展示、风险分析和后续回放所需的关键数据。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | BIGSERIAL | 主键 |
| `device_id` | VARCHAR(32) | 边缘设备编号 |
| `collected_at` | TIMESTAMPTZ | 边缘端采集时间 |
| `received_at` | TIMESTAMPTZ | 云端接收时间 |
| `gps_valid` | BOOLEAN | GPS 数据是否有效 |
| `latitude` | DOUBLE PRECISION | 纬度，无效时为空 |
| `longitude` | DOUBLE PRECISION | 经度，无效时为空 |
| `speed_kmh` | REAL | 车速，单位 km/h |
| `radar_valid` | BOOLEAN | 雷达数据是否有效 |
| `target_count` | SMALLINT | 雷达目标数量 |
| `nearest_distance_m` | REAL | 最近目标距离，单位 m |
| `min_ttc_s` | REAL | 最小碰撞时间 TTC，单位 s |
| `vision_valid` | BOOLEAN | 视觉数据是否有效 |
| `obstacle_count` | SMALLINT | 视觉障碍物数量 |
| `drivable_area_ratio` | REAL | 可行驶区域占比，范围 0～1 |
| `risk_score` | REAL | 综合风险分数，范围 0～1 |
| `risk_level` | SMALLINT | 风险等级：0 低、1 中、2 高 |

主要约束包括经纬度范围、非负速度与数量、风险分数范围以及风险等级枚举。主要索引为采集时间索引，以及 `(device_id, collected_at DESC)` 组合索引，便于 Dashboard 快速查询某设备的最新数据和最近一小时曲线。

### 5.2 视频元数据表 `video_segments`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | BIGSERIAL | 主键 |
| `device_id` | VARCHAR(32) | 设备编号 |
| `started_at` | TIMESTAMPTZ | 视频片段开始时间 |
| `duration_s` | REAL | 视频时长，默认约 60 秒 |
| `file_path` | VARCHAR | 相对视频存储路径 |
| `file_size_bytes` | BIGINT | 文件大小 |
| `status` | VARCHAR | 文件状态，正常完成为 `ready` |
| `received_at` | TIMESTAMPTZ | 云端接收时间 |

数据库只保存视频元数据，MP4 文件存储在文件系统中。这样避免把大文件直接写入数据库，降低数据库备份、查询和维护成本。

## 6. 边缘端数据映射

核心实现位于：

```text
src/dashboard/cloud_sync.py
```

`build_ride_payload(state, device_id)` 将 Dashboard 的内部状态转换为云端紧凑 JSON：

```python
payload = {
    "device_id": "bike-001",
    "collected_at": "2026-07-11T15:05:00+00:00",
    "gps_valid": True,
    "latitude": 39.9042,
    "longitude": 116.4074,
    "speed_kmh": 18.6,
    "radar_valid": True,
    "target_count": 2,
    "nearest_distance_m": 4.25,
    "min_ttc_s": 3.1,
    "vision_valid": True,
    "obstacle_count": 3,
    "drivable_area_ratio": 0.63,
    "risk_score": 0.68,
    "risk_level": 1,
}
```

数据清洗规则：

- GPS、雷达或视觉无效时，相应可空测量值上传为 `null`。
- `NaN` 和正负无穷值转换为 `null`，避免生成非标准 JSON 或触发数据库异常。
- 数量和速度限制为非负值。
- `risk_score` 限制在 0～1。
- `risk_level` 限制在 0～2。
- `device_id` 只允许 1～32 位字母、数字、下划线和连字符。
- `collected_at` 使用带时区的 ISO 8601 时间，云端同时记录 `received_at`，便于分析网络延迟。

## 7. 非阻塞上传设计

`CloudSyncClient` 启动三个独立线程：

1. `cloud-state-upload`：上传结构化骑行数据。
2. `cloud-video-record`：从共享摄像头帧录制视频。
3. `cloud-video-upload`：上传已完成的 MP4 文件。

### 7.1 骑行状态队列

状态队列最大长度为 2。当网络上传速度低于状态产生速度时，系统丢弃旧的待传状态，只保留较新的状态。

这是有意设计：实时监控更关注“当前状态”，不能让历史积压拖慢感知主循环。openGauss 中仍会保存所有成功上传的采样点。

### 7.2 视频持久化队列

视频和状态采用不同策略。视频录制完成后保留在本地 `data/cloud_spool`，只有云端成功返回后才删除。上传失败时等待 10 秒并重新入队；程序重启后会扫描遗留 MP4 并继续上传。

因此，短时断网可能造成少量实时状态采样缺口，但已经完成的视频片段能够补传。

## 8. 一分钟视频分段

视频录制复用 `CameraFrameProducer` 的 BGR 帧，不重复打开摄像头，避免多进程或多线程争抢同一设备。

默认参数：

- 帧率：10 FPS；DK2500 优先使用负载较低的 `mp4v` 编码
- 分段长度：60 秒
- 容器格式：MP4
- 编码器尝试顺序：`avc1` → `H264` → `mp4v`

正在写入的文件使用：

```text
*.partial.mp4
```

片段完成并关闭 `VideoWriter` 后，原子重命名为：

```text
*.mp4
```

上传线程只处理最终 `.mp4`，不会上传尚未写完的文件。程序启动时清理上次异常退出遗留的 `.partial.mp4`；正常退出时会关闭当前编码器，并把最后一段视频加入上传队列。

## 9. 云端 API

API 基地址：

```text
http://124.70.108.34
```

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/api/health` | 检查 API 与数据库连接 |
| POST | `/api/ride-samples` | 写入一条骑行数据 |
| GET | `/api/ride-samples/latest` | 查询指定设备最新数据 |
| GET | `/api/ride-samples/recent` | 查询指定设备最近若干分钟数据 |
| POST | `/api/video-segments` | 上传视频文件和元数据 |
| GET | `/api/video-segments/recent` | 查询指定设备最近视频 |
| GET | `/videos/<path>` | 访问视频文件 |

### 9.1 健康检查

```bash
curl http://124.70.108.34/api/health
```

正常响应：

```json
{"status":"ok","database":"connected"}
```

### 9.2 上传骑行数据

```bash
curl -X POST http://124.70.108.34/api/ride-samples \
  -H 'Content-Type: application/json' \
  -d '{
    "device_id":"bike-001",
    "collected_at":"2026-07-11T23:05:00+08:00",
    "gps_valid":true,
    "latitude":39.9042,
    "longitude":116.4074,
    "speed_kmh":18.6,
    "radar_valid":true,
    "target_count":2,
    "nearest_distance_m":4.25,
    "min_ttc_s":3.1,
    "vision_valid":true,
    "obstacle_count":3,
    "drivable_area_ratio":0.63,
    "risk_score":0.68,
    "risk_level":1
  }'
```

### 9.3 查询最近一小时数据

```bash
curl 'http://124.70.108.34/api/ride-samples/recent?device_id=bike-001&minutes=60'
```

### 9.4 查询最近一小时视频

```bash
curl 'http://124.70.108.34/api/video-segments/recent?device_id=bike-001&minutes=60'
```

API 返回的 `video_url` 可直接交给浏览器 `<video>` 标签。Nginx 支持 HTTP Range 请求，浏览器可以拖动播放进度。

## 10. 前端访问

系统包含两个界面：

1. DK2500 浅色实时 Dashboard：显示实时画面、传感器和风险状态。
2. 云端数据查看页：查询 openGauss 中的历史数据和最近视频。

云端页面地址：

```text
http://124.70.108.34/cloud-data.html
```

浅色 Dashboard 右上角提供“云端数据”按钮，点击后打开该页面。云端页面与 API 同源，不需要额外处理浏览器跨域问题。

## 11. 云端视频保留策略

`rider-video-cleanup.timer` 每分钟执行一次清理脚本：

```text
/opt/rider-cloud/app/cleanup_videos.py
```

清理逻辑：

1. 查询 `started_at` 早于当前时间一小时的视频记录。
2. 删除对应 MP4 文件。
3. 删除对应 `video_segments` 数据库记录。
4. 输出过期数、成功删除数和失败数。

实际测试结果：

```text
cleanup complete: expired=1, deleted=1, failed=0
```

该策略使磁盘占用近似稳定在“最近一小时视频量”，而骑行结构化数据长期保存在 openGauss 中。

## 12. 服务启动与恢复

云端启动关系：

```text
opengauss.service
       ├─> rider-cloud.service
       └─> rider-video-cleanup.service / timer

nginx.service ─> 对外提供页面、API 代理和视频文件
```

FastAPI 和视频清理服务均声明依赖 `opengauss.service`，避免服务器重启后出现 API 已启动但数据库 socket 尚不存在的问题。受控重启后，健康接口、数据库和页面均已验证能够自动恢复。

DK2500 的 systemd 模板位于：

```text
deploy/edge/rider-dashboard.service
```

模板使用真实传感器模式、视觉推理、1 Hz 状态上传、10 FPS 视频和 60 秒分段。需在最终硬件验证通过后安装。

## 13. 边缘端启动参数

手动联调命令：

```bash
cd /home/intelcup/Intel-Cup-2026
conda activate intel
python run_dashboard.py \
  --host 0.0.0.0 \
  --port 8000 \
  --dashboard-mode real \
  --profile dk2500 \
  --camera-id 0 \
  --enable-vision \
  --cloud-enable \
  --cloud-url http://124.70.108.34 \
  --device-id bike-001 \
  --cloud-state-hz 1 \
  --cloud-video-fps 10 \
  --cloud-video-seconds 60 \
  --cloud-spool data/cloud_spool
```

关键参数可以按设备性能调整。提高视频 FPS 会改善流畅度，但会增加边缘端编码负载、网络带宽和云端磁盘占用。

## 14. 测试与验收

### 14.1 已完成测试

- openGauss 建表、约束和索引检查。
- 事务插入与回滚测试。
- Python 连接 openGauss 测试。
- 骑行数据 API 正常写入测试。
- 非法风险等级校验测试。
- 最新数据与最近一小时查询测试。
- 视频上传、元数据查询和文件访问测试。
- Nginx Range 请求 `206 Partial Content` 测试。
- 一小时视频过期清理测试。
- 云端服务重启恢复测试。
- 边缘端载荷字段映射、无效值和非有限浮点数测试。
- Python 语法检查。
- DK2500 到云端健康接口连通性测试。
- DK2500 OpenCV MP4 编码能力测试。

载荷单元测试：

```bash
python scripts/test_cloud_sync_payload.py
```

通过标志：

```text
cloud sync payload test passed
```

### 14.2 待完成实机验收

1. DK2500 拉取最新代码。
2. 核对摄像头、GPS 和雷达串口。
3. 运行 10 秒，确认连续骑行数据进入 openGauss。
4. 运行 70～90 秒，确认一分钟视频生成、上传、播放和拖动正常。
5. 核对 `risk_score`、`risk_level`、视觉障碍物数和可行驶区域占比来自真实算法输出。
6. 安装并验证 DK2500 systemd 服务。
7. 连续运行一小时，检查断网恢复、内存、线程、本地暂存目录和云端磁盘。

## 15. 故障排查

### 15.1 API 报数据库 socket 不存在

现象：

```text
No such file or directory: .s.PGSQL.26000
```

检查：

```bash
systemctl status opengauss.service --no-pager
sudo -iu omm bash -lc 'source ~/.bashrc && gs_ctl query -D /gaussdb/data/db1'
```

恢复后重启 API：

```bash
systemctl restart rider-cloud.service
```

### 15.2 视频返回 403

检查 Nginx 对路径各级目录是否具有读取和遍历权限：

```bash
namei -l /opt/rider-cloud/videos/<device>/<file>.mp4
```

目录应至少为 755，视频文件应至少为 644。

### 15.3 视频无法拖动

验证 Range：

```bash
curl -D - -o /dev/null -H 'Range: bytes=0-9' \
  http://127.0.0.1/videos/<device>/<file>.mp4
```

预期状态码为 `206 Partial Content`。

### 15.4 边缘端视频未上传

检查：

- 摄像头是否能提供 BGR 帧。
- `data/cloud_spool` 是否存在 `.partial.mp4` 或 `.mp4`。
- OpenCV 是否至少能打开一种编码器。
- 控制台是否出现 `no usable MP4 codec` 或网络超时。
- 云端 `/api/health` 是否可访问。

### 15.5 数据没有持续更新

检查运行参数是否包含 `--cloud-enable`，并确认 `--device-id` 与前端查询设备一致。状态上传失败只记录日志，不会使 Dashboard 主程序退出。

## 16. 设计亮点与答辩表述

### 16.1 边云解耦

边缘端负责低延迟感知和风险计算，云端负责长期留存、查询与回放。即使云端短时不可用，边缘端安全功能仍能继续运行。

### 16.2 不阻塞实时主链路

网络请求和视频编码不在状态主循环中执行。有限状态队列通过保留最新样本控制积压，适合实时监控场景。

### 16.3 数据有效性显式建模

系统没有用 `0` 冒充无效经纬度、距离或可行驶比例，而是同时上传 `gps_valid`、`radar_valid`、`vision_valid`，无效测量值存为 `null`，避免后续统计产生误判。

### 16.4 视频完整性保护

录制中的视频使用 `.partial.mp4`，只有编码器正常关闭后才成为正式 MP4。上传成功前文件始终留在边缘端暂存目录。

### 16.5 分层存储

结构化指标进入 openGauss，视频文件进入文件系统，数据库只记录视频索引。该方案兼顾查询性能、实现复杂度和演示阶段成本。

### 16.6 可恢复部署

云端服务通过 systemd 管理并声明启动依赖，解决了重启后数据库未启动导致 API 不可用的问题。

## 17. 当前限制与后续改进

当前版本以快速实现竞赛原型和答辩演示为目标，后续可继续改进：

- 将 HTTP 升级为 HTTPS。
- 增加设备令牌、签名认证和接口限流。
- 对骑行状态增加磁盘级缓冲，实现断网后的完整补传。
- 为同一条样本增加唯一 ID，支持幂等写入和去重。
- 视频规模扩大后迁移至华为云 OBS，只在 openGauss 保存对象地址。
- 增加数据库分区和历史数据归档策略。
- 增加上传成功率、延迟、队列长度和磁盘空间监控。
- 将公网 IP、设备编号和上传频率移入独立配置文件或环境变量。

这些限制不影响当前“实时数据云端留存、最近一小时视频回放和 Dashboard 查询”的核心闭环。

## 18. 答辩常见问题

### 为什么状态数据和视频采用不同重试策略？

状态是高频时序数据，实时性高于逐条完整性，网络拥塞时保留最新值可以避免拖慢主系统；视频价值更高且频率低，因此使用本地持久化暂存并在网络恢复后补传。

### 为什么不把视频直接存进 openGauss？

大二进制文件会增加数据库体积、备份成本和查询负担。文件系统或对象存储更适合视频，数据库负责保存可检索的时间、设备和路径元数据。

### 如何保证无效传感器数据不会污染分析？

每类传感器都有独立 `valid` 字段。无效传感器的可空测量值写为 `null`，数据库还具有范围和非负约束。

### 如何保证上传不影响骑行安全算法？

上传、录制和视频上传都运行在独立线程；状态队列有固定上限，网络阻塞不会无限占用内存，也不会在实时状态循环中同步等待网络。

### 为什么视频只保留一小时？

项目需求是提供过去一小时骑行画面。定时删除过期视频可以限制磁盘增长，同时结构化骑行数据继续长期保存，便于统计分析。

### 服务器重启后为什么仍能恢复？

openGauss、FastAPI、Nginx 和清理定时器均由 systemd 管理，FastAPI 与清理服务显式依赖 openGauss。已通过受控重启验证服务能够自动恢复。
