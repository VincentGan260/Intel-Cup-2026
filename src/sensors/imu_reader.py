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
            import serial as _serial

            port = self.config.get("port", "/dev/ttyIMUWT61C")
            baudrate = self.config.get("baudrate", 115200)
            timeout = self.config.get("timeout", 0.1)

            try:
                self._serial = _serial.Serial(port, baudrate, timeout=timeout)
                self._buffer = bytearray()
                print(f"[IMUReader] 已打开串口 {port} @ {baudrate}")

                # ── 上电自动校准：静置采集 acc_z 基准 ──
                self._run_calibration()
            except Exception as e:
                print(f"[IMUReader] 串口打开失败: {e}")
                self._serial = None
        else:
            print("[IMUReader] mock 模式启动")

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
        if self._serial is not None:
            try:
                self._serial.close()
                print("[IMUReader] 串口已关闭")
            except Exception:
                pass
            self._serial = None
        else:
            print("[IMUReader] 已停止")

    def read_once(self) -> IMUData:
        ts = now()
        if self.is_real and self._serial is not None:
            result = self._read_real(ts)
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
            has_acc = False
            has_gyro = False
            has_angle = False

            while True:
                parsed, consumed = self._parse_next_packet()
                if parsed is None or consumed <= 0:
                    break

                ptype = parsed["type"]
                if ptype == "acc":
                    imu.acc_x = parsed["acc_x"]
                    imu.acc_y = parsed["acc_y"]
                    imu.acc_z = parsed["acc_z"]
                    has_acc = True
                elif ptype == "gyro":
                    imu.gyro_x = parsed["gyro_x"]
                    imu.gyro_y = parsed["gyro_y"]
                    imu.gyro_z = parsed["gyro_z"]
                    has_gyro = True
                elif ptype == "angle":
                    imu.roll = parsed["roll"]
                    imu.pitch = parsed["pitch"]
                    imu.yaw = parsed["yaw"]
                    has_angle = True

            # 有任一类型数据即认为有效
            if has_acc or has_gyro or has_angle:
                # 应用零偏校准：补偿安装倾斜 / 温漂
                if self._calibrated:
                    imu.acc_z -= self._calib_acc_z_offset

                # 原始评分（含角速度交叉验证：急刹时 nose-down pitch 角速度应同向增大）
                raw_brake, raw_bump, raw_tilt = _compute_imu_scores(
                    imu.roll, imu.pitch,
                    imu.acc_x, imu.acc_y, imu.acc_z,
                    imu.gyro_x, imu.gyro_y, imu.gyro_z,
                )

                # EMA 平滑防误触发：alpha=0.25 意味约 4 帧响应大部分变化，尖峰被抑制
                self._ema_brake = EMA_ALPHA * raw_brake + (1.0 - EMA_ALPHA) * self._ema_brake
                self._ema_bump = EMA_ALPHA * raw_bump + (1.0 - EMA_ALPHA) * self._ema_bump
                self._ema_tilt = EMA_ALPHA * raw_tilt + (1.0 - EMA_ALPHA) * self._ema_tilt

                imu.valid = True
                imu.brake_score = round(self._ema_brake, 3)
                imu.bump_score = round(self._ema_bump, 3)
                imu.tilt_score = round(self._ema_tilt, 3)

        except Exception as e:
            print(f"[IMUReader] 读取异常: {e}")

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

        # 逐字节扫描帧头 0x55
        for i in range(buf_len - 10):
            if self._buffer[i] != 0x55:
                continue

            ptype = self._buffer[i + 1]
            if ptype not in (0x51, 0x52, 0x53):
                continue

            frame = self._buffer[i : i + 11]

            # 校验和验证：前 10 字节累加取低 8 位
            expected_sum = (sum(frame[:10])) & 0xFF
            if expected_sum != frame[10]:
                continue  # 校验和不匹配，继续搜索

            # 解析通过，删除已消费字节
            del self._buffer[: i + 11]

            if ptype == 0x53:  # 角度包
                roll_raw = _signed(frame[3], frame[2])
                pitch_raw = _signed(frame[5], frame[4])
                yaw_raw = _signed(frame[7], frame[6])
                roll = roll_raw / 32768.0 * 180.0
                pitch = pitch_raw / 32768.0 * 180.0
                yaw = yaw_raw / 32768.0 * 180.0
                return {"type": "angle", "roll": roll, "pitch": pitch, "yaw": yaw}, i + 11

            elif ptype == 0x51:  # 加速度包
                ax = _signed(frame[3], frame[2])
                ay = _signed(frame[5], frame[4])
                az = _signed(frame[7], frame[6])
                acc_x = ax / 32768.0 * 16.0 * 9.8
                acc_y = ay / 32768.0 * 16.0 * 9.8
                acc_z = az / 32768.0 * 16.0 * 9.8
                return {"type": "acc", "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z}, i + 11

            elif ptype == 0x52:  # 角速度包
                gx = _signed(frame[3], frame[2])
                gy = _signed(frame[5], frame[4])
                gz = _signed(frame[7], frame[6])
                gyro_x = gx / 32768.0 * 2000.0
                gyro_y = gy / 32768.0 * 2000.0
                gyro_z = gz / 32768.0 * 2000.0
                return {"type": "gyro", "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z}, i + 11

        # 缓冲区中没有完整有效帧，清理残留垃圾（保留最后 10 字节用于拼接）
        if buf_len > 20:
            self._buffer = self._buffer[-10:]
        return None, 0

    # ---- mock ----

    def _read_mock(self, ts: float) -> IMUData:
        """模拟 IMU 数据，小幅随机变化，并生成合理评分。"""
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
