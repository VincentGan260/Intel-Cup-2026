# 待修复 Bug 清单（队友运行时模块）

> 来源：对新 push 代码的三路审查（评测/量化、融合/传感器、集成/面板）。
> **本文只列「我没动、需队友确认或需上硬件验证」的问题**——传感器解析、融合语义、面板长跑稳定性这些属队友模块且改完要在真实硬件上验证，不宜单方面改。
>
> **已由我修复的（不在本文）**：eval.yaml 的 ACDC 标签配置、量化脚本校准路径/分辨率/死代码（见 git 改动）。
>
> **不影响视觉评测结论**：下列问题都在"运行时/上车"链路，`docs/精度评测结果.md` 的分割/检测数字由独立脚本（eval_seg.py/eval_det.py）产出，不经过这些代码。

---

## 🔴 上车前建议必修（影响数据真实性 / 长跑稳定性）

### 1. IMU 帧解析 off-by-one —— 缓冲尾部的完整包永远读不到
- **位置**：`src/sensors/imu_reader.py`（帧扫描循环 `for i in range(len(self._buffer) - 11)`）
- **为什么错**：WT61C 一帧 11 字节。`range(N-11)` 最后一个 `i` 是 `N-12`，当完整包恰好从 `N-11` 开始（缓冲里最后一个完整包）时不会被匹配 → IMU 数据持续滞后一帧。
- **改法**：改为 `range(len(self._buffer) - 10)`，并保证删除时 `i+11 <= len`。

### 2. IMU 无校验和验证 —— 噪声/错位会解析出假姿态
- **位置**：`src/sensors/imu_reader.py`（解析角度/加速度包处）
- **为什么错**：WT61C 第 11 字节是 `(0x55+type+data[0..7]) & 0xFF` 校验和，现在完全不校验，串口噪声可能产生"看似合法"的 0x55 0x53 序列 → 错误 roll/pitch 直接喂风险模型。
- **改法**：解析前校验和不通过则跳过该候选位置继续找（不要 `del buffer[:i+11]` 误删后续）。

### 3. 雷达丢失 payload 长度校验 —— 坏帧被当好帧（valid=True）
- **位置**：`src/sensors/radar_reader.py`（`parse_radar_frame`）
- **为什么错**：相比原 `test_ld2451_radar.py` 删掉了 `expected_payload_len = 2 + target_count*5` 与长度检查，改成循环里 `if offset+5 > len: break` 静默截断。当 `target_count` 因错位虚高时，会凑出"看似有效"的 RadarData，垃圾距离/TTC 直接驱动报警。
- **改法**：恢复 `expected_payload_len` 校验，长度不符返回 `None`（让 `valid=False`）。

### 4. IMU 是「哑通道」—— R_pose 三个分量从未被填充，IMU 对风险零贡献
- **位置**：`src/sensors/imu_reader.py`（未计算 brake/bump/tilt score）+ `src/fusion/risk_model.py`（`R_pose = 0.4*brake + 0.3*bump + 0.3*tilt`）
- **为什么错**：reader（real 与 mock）都只填 roll/pitch/yaw/acc，三个 score 恒为 0 → `R_pose` 恒为 0；而 `modules_valid["pose"]=True` 又让 pose 占满权重，**反而稀释了其他真实风险项**。
- **改法**：在 reader 或独立特征步骤里由 acc/gyro/pitch 计算 brake/bump/tilt，写回 IMUData。

### 5. logger CSV `radar_target_count` 列写成了 vision_object_count（静默数据错位）
- **位置**：`src/utils/logger.py`（`write()` 第 ~92 行）
- **为什么错**：表头第 11 列是 `radar_target_count`，但该位置写的是 `state.vision_object_count`（且该值在另一列又写了一次）；雷达目标数从未被记录。离线分析会把视觉目标数误当雷达目标数。
- **改法**：`SystemState` 增 `radar_target_count` 字段并用 `len(radar.targets)` 填充，或删掉该列。

### 6. Dashboard MJPEG 流永不终止 —— 摄像头/线程泄漏
- **位置**：`src/dashboard/server.py`（`video_feed` / `video_annotated_feed` 的 `_generate()` `while True`）
- **为什么错**：客户端关页/刷新后生成器无取消检测，每开一次页面就新起一路 20-30FPS 解码循环，多客户端 + 残留连接争抢同一 `VideoCapture` 与锁，长跑线程累积、CPU 飙升。
- **改法**：循环里检测连接断开/捕获 `CancelledError`；或合并两个端点、共享单一抓帧循环。

---

## 🟡 建议修（标定偏差 / 部署脆弱性）

### 7. 多视频端点各自抢同一摄像头
- **位置**：`src/dashboard/server.py`（两个视频端点 + hybrid 后台都各自 `read()`）
- **影响**：单 `VideoCapture` 被多路并发读，帧被瓜分、标注框与显示帧错位。
- **改法**：单一抓帧线程维护"最新帧"，所有消费者读共享帧。

### 8. 同步器无过期数据剔除 —— 名为同步实为"最新值缓存"
- **位置**：`src/fusion/synchronizer.py`（`build_frame` 用 `now()` 而非源时间戳，无 max_age 判断）
- **影响**：某传感器线程卡死时，旧帧一直 `valid=True` 被融合使用。
- **改法**：`build_frame` 时按 `now()-data.timestamp > max_age` 把该源降级为 `valid=False`。

### 9. 相对配置路径 —— 非项目根启动会 FileNotFoundError
- **位置**：`src/fusion/risk_model.py`、`src/actuator/motor_controller.py`、`src/fusion/risk_level.py`、`run_dashboard.py` 等（`open("configs/...")`）
- **影响**：以 systemd 服务或从子目录启动时崩溃（DK-2500 部署易踩）。
- **改法**：下游模块按项目根解析相对路径，或入口统一传绝对路径。

### 10. R_speed 在电动车常速即饱和
- **位置**：`src/fusion/risk_model.py`（`R_speed = min(speed/25, 1.0)`）
- **影响**：电动车限速 25km/h，正常骑行即 R_speed≈1.0，把综合风险整体抬高、正常骑行频繁判中风险。
- **改法**：`max_speed_kmh` 设略高于巡航速度（如 35），或改非线性映射。

### 11. 高风险被自身冷却压制
- **位置**：`src/actuator/motor_controller.py`（冷却检查在打断分支之前）
- **影响**：持续高风险时报警按 `min_interval_sec` 间歇而非持续，削弱"高风险持续强震"意图。
- **改法**：level=2 不受同级冷却限制，或单独设更短间隔。

### 12. vision_adapter `max_visual_risk` 未 clamp
- **位置**：`src/fusion/vision_adapter.py`（透传未 `[0,1]` 截断）
- **影响**：越界值会写入日志 CSV（下游风险计算有兜底，但日志会脏）。
- **改法**：此处 clamp 到 [0,1]。

---

## 🟢 低优先（健壮性 / 清理）

- **run_dashboard.py**：mock 模式下传 `--enable-imu` 会触发 `NameError`（`imu_init_ok` 仅在 hybrid 分支定义）→ 函数顶部统一初始化 `imu_init_ok=False`。
- **radar_reader.py**：忽略了 `payload[1]` 硬件报警位；`confidence=SNR/255` 量纲存疑（几乎恒≈0）→ 核对 LD2451 SNR 量纲。
- **dashboard server.py**：无帧时仍 yield 空 MJPEG part → 改 `continue`。
- **motor_controller.py**：`_last_any_time` 写了从不读（死字段）。

---

## 确认无误的点（审查已核对，无需改）
- GPS NMEA 解析（knots→km/h ×1.852、经纬度拆分、S/W 取负）正确。
- 雷达帧提取 `_extract_frame`（find header→对齐→按长度取帧）正确。
- `state_store` 锁、VisionResult 缓存锁正确，并发读写安全。
- main_integrated 的安全确认、逆序关闭、视觉降级、`finally` 清理正确。
- preflight_dk2500 各 reader 的串口未开检测、超时、马达回退检测正确。

> 建议处理顺序：上车联调前先修 🔴 1/2/3/4/5/6（决定融合输出是否真实可信 + 面板能否长跑），🟡 可在联调中逐步修。
