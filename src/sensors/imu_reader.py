"""IMU 读取器封装（WT61C 六轴姿态传感器）。

从 read_wt61c.py 提取核心协议解析逻辑，不直接 import 原脚本。
支持 real / mock 两种模式。

real 模式：
  - 从串口读取 WT61C 数据帧
  - 统一解析三种数据包并做校验和验证：
      0x55 0x51 — 加速度包（acc_x/y/z）
      0x55 0x52 — 角速度包（gyro_x/y/z）
      0x55 0x53 — 角度包（roll/pitch/yaw）
  - 基于解析结果实时计算 brake_score / bump_score / tilt_score
  - 一次 read_once() 消耗缓冲区中全部完整帧，保留各类型最新值
  - 上电自动校准：静置采样 acc_z 基准，补偿安装误差
  - 防误触发：EMA 平滑抑制单帧尖峰，gyro 交叉验证刹车

mock 模式：
  - 返回模拟 IMU 数据，包含小幅随机变化及合理评分
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from src.fusion.data_types import IMUData, now
from src.sensors.sensor_base import BaseSensorReader

# — 评分阈值（经验值，后续可通过骑行数据标定） —
BRAKE_DECEL_THRESHOLD = 5.0  # m/s²，急刹判定阈值
BUMP_DEVIATION_THRESHOLD = 8.0  # m/s²，垂直颠簸判定阈值（原10.0过高，中等颠簸无感知）
TILT_ANGLE_THRESHOLD = 35.0  # °，侧倾判定阈值（双轮 ≤30° 为正常状态，≥35° 接近失控）
GYRO_CROSS_VALIDATE_PITCH_RATE = 15.0  # °/s，急刹时 nose-down pitch 角速度辅助阈值
EMA_ALPHA = 0.25  # EMA 平滑系数（越小越平滑，抑制单帧尖峰）


def _signed(high: int, low: int) -> int:
    """将 16 位无符号组合值转为有符号整数。"""
    v = ((high & 0xFF) << 8) | (low & 0xFF)
    return v - 65536 if v >= 32768 else v


def _compute_imu_scores(
    roll: float, pitch: float,
    acc_x: float, acc_y: float, acc_z: float,
    gyro_x: float, gyro_y: float, gyro_z: float,
) -> tuple:
    """由原始 IMU 数据计算三类姿态风险评分 [0, 1]。

    - brake_score：基于 x 轴负向加速度 + pitch 角速度交叉验证（急刹特征）
    - bump_score：基于 z 轴偏离重力加速度的程度（颠簸特征）
    - tilt_score：基于 roll/pitch 偏离水平面的程度 + 角速度辅助（侧倾特征）
    """
    # 急刹：x 轴负向加速度越大风险越高
    raw_brake = max(0.0, min(1.0, abs(min(acc_x, 0.0)) / BRAKE_DECEL_THRESHOLD))
    # 角速度交叉验证：急刹时车头下沉，gyro_y（pitch 角速度）应同向增大
    # 若 x 轴减速大但 pitch 角速度小（路面坑洞单体冲击），衰减 brake_score
    if raw_brake > 0.15 and abs(gyro_y) < 5.0:
        raw_brake *= 0.4  # 减速但 pitch 不动 → 可能是路面坑洞，非真急刹
    brake_score = raw_brake

    # 颠簸：z 轴偏离 1g 越远风险越高（取绝对值，方向无意义）
    bump_deviation = abs(acc_z - 9.8)
    bump_score = max(0.0, min(1.0, bump_deviation / BUMP_DEVIATION_THRESHOLD))

    # 侧倾：roll 或 pitch 绝对值越大风险越高，角速度放大快速倾斜
    max_tilt = max(abs(roll), abs(pitch))
    raw_tilt = max(0.0, min(1.0, max_tilt / TILT_ANGLE_THRESHOLD))
    # 角速度辅助：快速倾斜（如被撞）即使静态角度未达阈值也应提升风险
    if raw_tilt < 0.5 and abs(gyro_x) > GYRO_CROSS_VALIDATE_PITCH_RATE:
        raw_tilt = max(raw_tilt, 0.3)
    tilt_score = raw_tilt

    return brake_score, bump_score, tilt_score


class IMUReader(BaseSensorReader):
    """WT61C 六轴姿态读取器。"""

    def __init__(
        self,
        mode: str = "mock",
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(mode, config)
        self._serial: Optional["serial.Serial"] = None  # noqa: F821
        self._buffer = bytearray()
        self._component_values: dict[str, tuple[float, float, float]] = {}
        self._component_times: dict[str, float] = {}
        # Oldest component time is the causal capture time of the combined
        # acc/gyro/angle sample. Consumers must not replace it with read time.
        self.last_sample_monotonic_ns: int = 0
        self._max_data_age_sec = float(self.config.get("max_data_age_sec", 0.5))
        self._roll_offset_deg = float(self.config.get("roll_offset_deg", 0.0))
        self._pitch_offset_deg = float(self.config.get("pitch_offset_deg", 0.0))
        if not math.isfinite(self._roll_offset_deg) or not math.isfinite(self._pitch_offset_deg):
            raise ValueError("IMU installation offsets must be finite")
        self._packet_counts = {"acc": 0, "gyro": 0, "angle": 0}
        self._bad_checksum_count = 0
        self._discarded_byte_count = 0
        self.last_error = ""

        # mock 状态
        self._mock_roll = 0.0
        self._mock_pitch = 0.0
        self._mock_yaw = 0.0

        # 零偏校准值（上电静置采样 acc_z 偏移，补偿安装倾斜/温漂）
        self._calib_acc_z_offset: float = 0.0
        self._calibrated: bool = False

        # EMA 平滑状态（防误触发：抑制单帧尖峰）
        self._ema_brake: float = 0.0
        self._ema_bump: float = 0.0
        self._ema_tilt: float = 0.0

    def start(self) -> None:
        if self.is_real:
            self._serial_reconnect_enabled = True
            if self._connect_serial():
                # Only calibrate during startup. Recalibrating after a reconnect
                # could block the live pipeline while the bicycle is moving.
                self._run_calibration()
        else:
            print("[IMUReader] mock 模式启动")

    def _connect_serial(self) -> bool:
        connected = self._open_serial_with_retry(
            port=self.config.get("port", "/dev/ttyIMUWT61C"),
            baudrate=int(self.config.get("baudrate", 115200)),
            timeout=float(self.config.get("timeout", 0.1)),
            label="IMUReader",
        )
        if connected:
            self._buffer.clear()
            self._component_values.clear()
            self._component_times.clear()
            self.last_sample_monotonic_ns = 0
        return connected

    def _run_calibration(self) -> None:
        """上电静置校准：采集 N 帧 acc_z，取平均偏移作为零偏补偿。

        传感器应保持静止（平放），校准约需 1 秒。
        若采样不足（串口无数据），跳过校准使用默认值 0。
        """
        CALIB_FRAMES = 50
        acc_z_samples: list[float] = []
        attempts = 0
        max_attempts = CALIB_FRAMES * 3  # 最多等待 3 倍帧数

        print(f"[IMUReader]   校准中...（采集 {CALIB_FRAMES} 帧静置数据，约 1 秒，请保持传感器静止）")
        while len(acc_z_samples) < CALIB_FRAMES and attempts < max_attempts:
            try:
                if self._serial is None or not self._serial.is_open:
                    break
                if self._serial.in_waiting > 0:
                    chunk = self._serial.read(self._serial.in_waiting)
                    self._buffer.extend(chunk)
                parsed, consumed = self._parse_next_packet()
                if parsed is not None and consumed > 0 and parsed["type"] == "acc":
                    acc_z_samples.append(parsed["acc_z"])
                attempts += 1
                time.sleep(0.01)
            except Exception:
                break

        if len(acc_z_samples) >= 10:
            avg_acc_z = sum(acc_z_samples) / len(acc_z_samples)
            self._calib_acc_z_offset = avg_acc_z - 9.8
            self._calibrated = True
            print(f"[IMUReader]   校准完成：{len(acc_z_samples)} 帧，acc_z 平均={avg_acc_z:.3f}，"
                  f"零偏 offset={self._calib_acc_z_offset:.4f} m/s²")
        else:
            print(f"[IMUReader]   校准跳过：仅采集 {len(acc_z_samples)} 帧（传感器无数据或异常），"
                  f"使用默认 offset=0")
        self._buffer.clear()

    def stop(self) -> None:
        self._disable_serial_reconnect()
        print("[IMUReader] 已停止")

    def read_once(self) -> IMUData:
        ts = now()
        if self.is_real:
            if self._serial is None or not getattr(self._serial, "is_open", False):
                self._connect_serial()
            # real 模式下串口不可用必须明确返回 invalid，不得伪装成 mock 数据。
            result = (self._read_real(ts) if self._serial is not None
                      else IMUData(timestamp=ts, valid=False))
        else:
            result = self._read_mock(ts)
        self._latest = result
        return result

    # ---- real ----

    def _read_real(self, ts: float) -> IMUData:
        """从串口读取并解析 WT61C 数据。

        一次调用消费缓冲区中全部完整帧，
        保留各类型（acc / gyro / angle）最新有效值，
        最后基于组合结果计算评分。
        """
        imu = IMUData(timestamp=ts, valid=False)
        try:
            if self._serial is None or not self._serial.is_open:
                return imu

            # 读取串口数据
            if self._serial.in_waiting > 0:
                chunk = self._serial.read(self._serial.in_waiting)
                self._buffer.extend(chunk)

            # 限制缓冲区大小（溢出时对齐到最后一个 0x55 帧头，防止截断在帧中间）
            if len(self._buffer) > 2048:
                keep_from = max(0, len(self._buffer) - 1024)
                # 往后找第一个 0x55，确保从帧头开始
                while keep_from < len(self._buffer) and self._buffer[keep_from] != 0x55:
                    keep_from += 1
                if keep_from >= len(self._buffer):
                    keep_from = max(0, len(self._buffer) - 10)
                self._buffer = self._buffer[keep_from:]

            # 循环解析所有完整帧
            received_new_packet = False

            while True:
                parsed, consumed = self._parse_next_packet()
                if parsed is None or consumed <= 0:
                    break

                ptype = parsed["type"]
                packet_time = time.monotonic()
                if ptype == "acc":
                    self._component_values["acc"] = (
                        parsed["acc_x"], parsed["acc_y"], parsed["acc_z"],
                    )
                elif ptype == "gyro":
                    self._component_values["gyro"] = (
                        parsed["gyro_x"], parsed["gyro_y"], parsed["gyro_z"],
                    )
                elif ptype == "angle":
                    self._component_values["angle"] = (
                        parsed["roll"], parsed["pitch"], parsed["yaw"],
                    )
                self._component_times[ptype] = packet_time
                received_new_packet = True

            # WT61C 的 acc/gyro/angle 是独立帧。只有三类数据均已收到且
            # 都在时效窗口内，才输出一个完整、有效的 IMU 样本。
            sample_time = time.monotonic()
            required = ("acc", "gyro", "angle")
            components_fresh = all(
                name in self._component_values
                and sample_time - self._component_times.get(name, float("-inf"))
                <= self._max_data_age_sec
                for name in required
            )

            if components_fresh:
                self.last_sample_monotonic_ns = int(
                    min(self._component_times[name] for name in required)
                    * 1_000_000_000)
                imu.acc_x, imu.acc_y, imu.acc_z = self._component_values["acc"]
                imu.gyro_x, imu.gyro_y, imu.gyro_z = self._component_values["gyro"]
                imu.roll, imu.pitch, imu.yaw = self._component_values["angle"]
                imu.body_roll = imu.roll - self._roll_offset_deg
                imu.body_pitch = imu.pitch - self._pitch_offset_deg

                # 应用零偏校准：补偿安装倾斜 / 温漂
                if self._calibrated:
                    imu.acc_z -= self._calib_acc_z_offset

                # 没有新帧时只沿用最近的完整样本，不重复推进 EMA。
                if received_new_packet:
                    raw_brake, raw_bump, raw_tilt = _compute_imu_scores(
                        imu.body_roll, imu.body_pitch,
                        imu.acc_x, imu.acc_y, imu.acc_z,
                        imu.gyro_x, imu.gyro_y, imu.gyro_z,
                    )

                    self._ema_brake = EMA_ALPHA * raw_brake + (1.0 - EMA_ALPHA) * self._ema_brake
                    self._ema_bump = EMA_ALPHA * raw_bump + (1.0 - EMA_ALPHA) * self._ema_bump
                    self._ema_tilt = EMA_ALPHA * raw_tilt + (1.0 - EMA_ALPHA) * self._ema_tilt

                imu.valid = True
                imu.brake_score = round(self._ema_brake, 3)
                imu.bump_score = round(self._ema_bump, 3)
                imu.tilt_score = round(self._ema_tilt, 3)

        except Exception as e:
            self._mark_serial_disconnected(e, label="IMUReader")

        return imu

    def _parse_next_packet(self) -> tuple:
        """从缓冲区头部扫描下一个完整 WT61C 帧并做校验和验证。

        WT61C 帧结构（11 字节）：
          [0]     0x55 帧头
          [1]     类型（0x51=acc, 0x52=gyro, 0x53=angle）
          [2..9]  8 字节数据载荷
          [10]    校验和 = (0x55 + type + data[0..7]) & 0xFF

        Returns:
            (parsed_dict, consumed_bytes)
            parsed_dict=None 表示未找到有效帧，consumed 为已跳过的字节数。
        """
        buf_len = len(self._buffer)
        search_from = 0

        # 逐字节扫描帧头 0x55。坏校验帧只计数一次，
        # 扫描继续向后查找，以便同一批数据中的后续好帧仍能被解析。
        for i in range(buf_len - 10):
            if i < search_from:
                continue
            if self._buffer[i] != 0x55:
                continue

            ptype = self._buffer[i + 1]
            if ptype not in (0x51, 0x52, 0x53):
                continue

            frame = self._buffer[i : i + 11]

            # 校验和验证：前 10 字节累加取低 8 位
            expected_sum = (sum(frame[:10])) & 0xFF
            if expected_sum != frame[10]:
                self._bad_checksum_count += 1
                search_from = i + 1
                continue  # 校验和不匹配，继续搜索

            # 解析通过，删除已消费字节
            self._discarded_byte_count += i
            del self._buffer[: i + 11]

            if ptype == 0x53:  # 角度包
                roll_raw = _signed(frame[3], frame[2])
                pitch_raw = _signed(frame[5], frame[4])
                yaw_raw = _signed(frame[7], frame[6])
                roll = roll_raw / 32768.0 * 180.0
                pitch = pitch_raw / 32768.0 * 180.0
                yaw = yaw_raw / 32768.0 * 180.0
                self._packet_counts["angle"] += 1
                return {"type": "angle", "roll": roll, "pitch": pitch, "yaw": yaw}, i + 11

            elif ptype == 0x51:  # 加速度包
                ax = _signed(frame[3], frame[2])
                ay = _signed(frame[5], frame[4])
                az = _signed(frame[7], frame[6])
                acc_x = ax / 32768.0 * 16.0 * 9.8
                acc_y = ay / 32768.0 * 16.0 * 9.8
                acc_z = az / 32768.0 * 16.0 * 9.8
                self._packet_counts["acc"] += 1
                return {"type": "acc", "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z}, i + 11

            elif ptype == 0x52:  # 角速度包
                gx = _signed(frame[3], frame[2])
                gy = _signed(frame[5], frame[4])
                gz = _signed(frame[7], frame[6])
                gyro_x = gx / 32768.0 * 2000.0
                gyro_y = gy / 32768.0 * 2000.0
                gyro_z = gz / 32768.0 * 2000.0
                self._packet_counts["gyro"] += 1
                return {"type": "gyro", "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z}, i + 11

        # 丢弃已确认为坏校验帧的帧头，避免下次调用重复计数。
        if search_from > 0:
            self._discarded_byte_count += search_from
            del self._buffer[:search_from]
            buf_len = len(self._buffer)

        # 缓冲区中没有完整有效帧，清理残留垃圾（保留最后 10 字节用于拼接）
        if buf_len > 20:
            self._discarded_byte_count += buf_len - 10
            self._buffer = self._buffer[-10:]
        return None, 0

    def get_diagnostics(self) -> dict:
        """返回 IMU 串口、帧校验和三类分量同步诊断。"""
        current = time.monotonic()
        component_age_ms = {
            name: round((current - timestamp) * 1000.0, 3)
            for name, timestamp in self._component_times.items()
        }
        component_arrival_ns = {
            name: int(timestamp * 1_000_000_000)
            for name, timestamp in self._component_times.items()
        }
        component_skew_ms = None
        if all(name in self._component_times for name in ("acc", "gyro", "angle")):
            timestamps = [self._component_times[name] for name in ("acc", "gyro", "angle")]
            component_skew_ms = round((max(timestamps) - min(timestamps)) * 1000.0, 3)
        return {
            "port": self.config.get("port", "/dev/ttyIMUWT61C"),
            "baudrate": int(self.config.get("baudrate", 115200)),
            "roll_offset_deg": self._roll_offset_deg,
            "pitch_offset_deg": self._pitch_offset_deg,
            "connected": bool(self._serial is not None and getattr(self._serial, "is_open", False)),
            "packet_counts": dict(self._packet_counts),
            "bad_checksum_count": self._bad_checksum_count,
            "discarded_byte_count": self._discarded_byte_count,
            "component_arrival_monotonic_ns": component_arrival_ns,
            "sample_capture_monotonic_ns": self.last_sample_monotonic_ns,
            "component_age_ms": component_age_ms,
            "component_skew_ms": component_skew_ms,
            "last_error": self.last_error,
        }

    # ---- mock ----

    def _read_mock(self, ts: float) -> IMUData:
        """模拟 IMU 数据，小幅随机变化，并生成合理评分。"""
        self.last_sample_monotonic_ns = time.monotonic_ns()
        self._mock_roll += random.uniform(-0.5, 0.5)
        self._mock_pitch += random.uniform(-0.5, 0.5)
        self._mock_yaw += random.uniform(-1.0, 1.0)
        self._mock_roll = max(-30, min(30, self._mock_roll))
        self._mock_pitch = max(-30, min(30, self._mock_pitch))

        acc_x = random.uniform(-0.5, 0.5)
        acc_y = random.uniform(-0.3, 0.3)
        acc_z = 9.8 + random.uniform(-0.2, 0.2)
        gyro_x = random.uniform(-1, 1)
        gyro_y = random.uniform(-1, 1)
        gyro_z = random.uniform(-1, 1)

        roll = round(self._mock_roll, 1)
        pitch = round(self._mock_pitch, 1)
        yaw = round(self._mock_yaw, 1)
        acc_x = round(acc_x, 2)
        acc_y = round(acc_y, 2)
        acc_z = round(acc_z, 2)
        gyro_x = round(gyro_x, 2)
        gyro_y = round(gyro_y, 2)
        gyro_z = round(gyro_z, 2)

        brake_score, bump_score, tilt_score = _compute_imu_scores(
            roll, pitch, acc_x, acc_y, acc_z,
            gyro_x, gyro_y, gyro_z,
        )

        # EMA 平滑（mock 路径也做，行为与 real 一致）
        self._ema_brake = EMA_ALPHA * brake_score + (1.0 - EMA_ALPHA) * self._ema_brake
        self._ema_bump = EMA_ALPHA * bump_score + (1.0 - EMA_ALPHA) * self._ema_bump
        self._ema_tilt = EMA_ALPHA * tilt_score + (1.0 - EMA_ALPHA) * self._ema_tilt

        return IMUData(
            timestamp=ts,
            valid=True,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            body_roll=roll,
            body_pitch=pitch,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            gyro_x=gyro_x,
            gyro_y=gyro_y,
            gyro_z=gyro_z,
            brake_score=round(self._ema_brake, 3),
            bump_score=round(self._ema_bump, 3),
            tilt_score=round(self._ema_tilt, 3),
        )
