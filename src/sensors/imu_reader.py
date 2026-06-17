"""IMU 读取器封装（WT61C 六轴姿态传感器）。

从 read_wt61c.py 提取核心协议解析逻辑（0x55 0x53 角度包），不直接 import 原脚本。
支持 real / mock 两种模式。

real 模式：
  - 从串口读取 WT61C 数据帧
  - 解析角度数据包 (0x55 0x53)：roll/pitch/yaw
  - 解析加速度数据包 (0x55 0x51)：acc_x/acc_y/acc_z

mock 模式：
  - 返回模拟 IMU 数据，包含小幅随机变化
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from src.fusion.data_types import IMUData, now
from src.sensors.sensor_base import BaseSensorReader


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

    def start(self) -> None:
        if self.is_real:
            import serial as _serial

            port = self.config.get("port", "/dev/ttyUSB0")
            baudrate = self.config.get("baudrate", 115200)
            timeout = self.config.get("timeout", 0.1)

            try:
                self._serial = _serial.Serial(port, baudrate, timeout=timeout)
                self._buffer = bytearray()
                print(f"[IMUReader] 已打开串口 {port} @ {baudrate}")
            except Exception as e:
                print(f"[IMUReader] 串口打开失败: {e}")
                self._serial = None
        else:
            print("[IMUReader] mock 模式启动")

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
        """从串口读取并解析 WT61C 数据。"""
        imu = IMUData(timestamp=ts, valid=False)
        try:
            if self._serial is None or not self._serial.is_open:
                return imu

            if self._serial.in_waiting > 0:
                chunk = self._serial.read(self._serial.in_waiting)
                self._buffer.extend(chunk)

            if len(self._buffer) > 2048:
                self._buffer = self._buffer[-1024:]

            angle_data = self._parse_angle_packet()
            acc_data = self._parse_acc_packet()

            if angle_data or acc_data:
                imu.valid = True
                if angle_data:
                    imu.roll = angle_data["roll"]
                    imu.pitch = angle_data["pitch"]
                    imu.yaw = angle_data["yaw"]
                if acc_data:
                    imu.acc_x = acc_data["acc_x"]
                    imu.acc_y = acc_data["acc_y"]
                    imu.acc_z = acc_data["acc_z"]

        except Exception as e:
            print(f"[IMUReader] 读取异常: {e}")

        return imu

    def _parse_angle_packet(self) -> Optional[dict]:
        """从缓冲区解析角度数据包 (0x55 0x53)。"""
        for i in range(len(self._buffer) - 11):
            if self._buffer[i] == 0x55 and self._buffer[i + 1] == 0x53:
                rollL = self._buffer[i + 2]
                rollH = self._buffer[i + 3]
                pitchL = self._buffer[i + 4]
                pitchH = self._buffer[i + 5]
                yawL = self._buffer[i + 6]
                yawH = self._buffer[i + 7]

                roll = ((rollH << 8) | rollL) / 32768.0 * 180
                pitch = ((pitchH << 8) | pitchL) / 32768.0 * 180
                yaw = ((yawH << 8) | yawL) / 32768.0 * 180

                if roll > 180:
                    roll -= 360
                if pitch > 180:
                    pitch -= 360
                if yaw > 180:
                    yaw -= 360

                del self._buffer[: i + 11]
                return {"roll": roll, "pitch": pitch, "yaw": yaw}
        return None

    def _parse_acc_packet(self) -> Optional[dict]:
        """从缓冲区解析加速度数据包 (0x55 0x51)。"""
        for i in range(len(self._buffer) - 11):
            if self._buffer[i] == 0x55 and self._buffer[i + 1] == 0x51:
                axL = self._buffer[i + 2]
                axH = self._buffer[i + 3]
                ayL = self._buffer[i + 4]
                ayH = self._buffer[i + 5]
                azL = self._buffer[i + 6]
                azH = self._buffer[i + 7]

                acc_x = ((axH << 8) | axL) / 32768.0 * 16  # ±16g
                acc_y = ((ayH << 8) | ayL) / 32768.0 * 16
                acc_z = ((azH << 8) | azL) / 32768.0 * 16

                acc_x *= 9.8
                acc_y *= 9.8
                acc_z *= 9.8

                del self._buffer[: i + 11]
                return {"acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z}
        return None

    # ---- mock ----

    def _read_mock(self, ts: float) -> IMUData:
        """模拟 IMU 数据，小幅随机变化。"""
        self._mock_roll += random.uniform(-0.5, 0.5)
        self._mock_pitch += random.uniform(-0.5, 0.5)
        self._mock_yaw += random.uniform(-1.0, 1.0)
        self._mock_roll = max(-30, min(30, self._mock_roll))
        self._mock_pitch = max(-30, min(30, self._mock_pitch))

        acc_x = random.uniform(-0.5, 0.5)
        acc_y = random.uniform(-0.3, 0.3)
        acc_z = 9.8 + random.uniform(-0.2, 0.2)

        return IMUData(
            timestamp=ts,
            valid=True,
            roll=round(self._mock_roll, 1),
            pitch=round(self._mock_pitch, 1),
            yaw=round(self._mock_yaw, 1),
            acc_x=round(acc_x, 2),
            acc_y=round(acc_y, 2),
            acc_z=round(acc_z, 2),
            gyro_x=round(random.uniform(-1, 1), 2),
            gyro_y=round(random.uniform(-1, 1), 2),
            gyro_z=round(random.uniform(-1, 1), 2),
        )
