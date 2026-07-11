"""毫米波雷达读取器封装（HLK-LD2451）。

从 test_ld2451_radar.py 提取核心协议解析逻辑（数据帧：F4 F3 F2 F1），不直接 import 原脚本。
支持 real / mock 两种模式。

real 模式：
  - 从串口读取 LD2451 数据帧
  - 帧头 F4 F3 F2 F1，帧尾 F8 F7 F6 F5
  - 解析目标信息（角度、距离、速度方向、速度值、信噪比）
  - 计算最近目标距离和 TTC

mock 模式：
  - 模拟雷达目标数据，包含 0~2 个随机目标
"""

from __future__ import annotations

import math
import random
import struct
import time
from typing import List, Optional

from src.fusion.data_types import RadarData, RadarTarget, now
from src.sensors.sensor_base import BaseSensorReader

# 协议常量
DATA_HEADER = b"\xF4\xF3\xF2\xF1"
DATA_END = b"\xF8\xF7\xF6\xF5"


def parse_radar_frame(
    data: bytes,
    approaching_direction_code: int = 1,
    angle_sign: int = -1,
) -> Optional[dict]:
    """解析一帧雷达数据。

    数据帧格式:
      F4 F3 F2 F1 | 帧内数据长度(2B, LE) | 目标数(1B) | 报警信息(1B) | 目标*N(5B) | F8 F7 F6 F5

    每个目标 5 字节:
      [角度(1B)] [距离(1B)] [速度方向(1B)] [速度值(1B)] [信噪比(1B)]
      实际角度 = 原始值 - 0x80
      速度方向: V1.03 表格写 01=靠近，但同页数据实例写 00=靠近。
      因此由 approaching_direction_code 配置；当前前向安装真机标定为 01=靠近。
      距离: 0-100 米
      速度: 0-120 km/h
    """
    if len(data) < 10:
        return None
    if not data.startswith(DATA_HEADER):
        return None
    if not data.endswith(DATA_END):
        return None

    payload_len = struct.unpack("<H", data[4:6])[0]

    if payload_len == 0:
        return {"targets": []}

    payload = data[6 : 6 + payload_len]
    if len(payload) < 2:
        return None

    target_count = payload[0]
    # 完整性校验：payload = 目标数(1) + 报警(1) + 目标×5。长度不符即坏帧（防错位把虚高
    # target_count 解析成"看似有效"的垃圾目标）。alarm_info = payload[1]（硬件报警位）。
    expected_len = 2 + target_count * 5
    if len(payload) < expected_len:
        return None
    alarm_info = payload[1]

    targets = []
    for i in range(target_count):
        offset = 2 + i * 5
        t = payload[offset : offset + 5]
        angle_raw = t[0]
        distance = t[1]
        speed_dir = t[2]
        speed_kmh = t[3]
        snr = t[4]

        angle = (angle_raw - 0x80) * angle_sign
        speed_mps = speed_kmh / 3.6
        is_approaching = speed_dir == approaching_direction_code
        if is_approaching:
            relative_speed_mps = -speed_mps
        else:
            relative_speed_mps = speed_mps

        targets.append(
            {
                "target_id": i,
                "distance_m": float(distance),
                "relative_speed_mps": relative_speed_mps,
                "angle_deg": float(angle),
                "snr": int(snr),
                "speed_direction_code": int(speed_dir),
                "is_approaching": is_approaching,
            }
        )

    return {"targets": targets, "alarm": int(alarm_info)}


def _calculate_ttc(distance_m: float, relative_speed_mps: float) -> float:
    """计算碰撞时间 TTC（秒）。

    只有目标在接近（相对速度为负）时才有意义的 TTC。
    如果目标远离或静止，返回 -1。
    """
    if relative_speed_mps >= 0:
        return -1.0
    approaching_speed = -relative_speed_mps
    if approaching_speed < 0.01:
        return -1.0
    return distance_m / approaching_speed


class RadarReader(BaseSensorReader):
    """HLK-LD2451 毫米波雷达读取器。"""

    def __init__(
        self,
        mode: str = "mock",
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(mode, config)
        self._serial: Optional["serial.Serial"] = None  # noqa: F821
        self._buffer = bytearray()
        self.last_sample_monotonic_ns: int = 0
        self.approaching_direction_code = int(self.config.get("approaching_direction_code", 1))
        if self.approaching_direction_code not in (0, 1):
            raise ValueError("radar approaching_direction_code must be 0 or 1")
        self.angle_sign = int(self.config.get("angle_sign", -1))
        if self.angle_sign not in (-1, 1):
            raise ValueError("radar angle_sign must be -1 or 1")

    def start(self) -> None:
        if self.is_real:
            import serial as _serial

            port = self.config.get("port", "/dev/ttyRadarLD2451")
            baudrate = self.config.get("baudrate", 256000)
            timeout = self.config.get("timeout", 0.5)

            try:
                self._serial = _serial.Serial(port, baudrate, timeout=timeout)
                self._buffer = bytearray()
                print(f"[RadarReader] 已打开串口 {port} @ {baudrate}")
            except Exception as e:
                print(f"[RadarReader] 串口打开失败: {e}")
                self._serial = None
        else:
            print("[RadarReader] mock 模式启动")

    def stop(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
                print("[RadarReader] 串口已关闭")
            except Exception:
                pass
            self._serial = None
        else:
            print("[RadarReader] 已停止")

    def read_once(self) -> RadarData:
        ts = now()
        if self.is_real:
            if self._serial is not None:
                result = self._read_real(ts)
            else:
                # real 模式但串口不可用（打开失败），返回异常状态
                result = RadarData(timestamp=ts, valid=False)
        else:
            result = self._read_mock(ts)
        self._latest = result
        return result

    # ---- real ----

    def _read_real(self, ts: float) -> RadarData:
        """从串口读取并解析雷达数据帧。"""
        radar = RadarData(timestamp=ts, valid=False)
        try:
            if self._serial is None or not self._serial.is_open:
                return radar

            if self._serial.in_waiting > 0:
                chunk = self._serial.read(self._serial.in_waiting)
                self._buffer.extend(chunk)
                # LD2451无硬件时间戳；以本批串口字节到达主机并读完的时刻作为保守到达时间。
                self.last_sample_monotonic_ns = time.monotonic_ns()

            if len(self._buffer) > 4096:
                self._buffer = self._buffer[-2048:]

            # Dashboard/训练消费频率低于雷达约20 Hz上报率时，必须丢弃积压旧帧，
            # 否则每轮只取最老一帧会让雷达时间越来越落后于相机。
            result = self._extract_latest_frame()
            if result is not None:
                targets = []
                nearest = -1.0
                min_ttc = -1.0

                for t in result["targets"]:
                    rt = RadarTarget(
                        target_id=t["target_id"],
                        distance_m=t["distance_m"],
                        relative_speed_mps=t["relative_speed_mps"],
                        angle_deg=t["angle_deg"],
                        confidence=min(1.0, t.get("snr", 0) / 255.0),
                    )
                    targets.append(rt)

                    if nearest < 0 or t["distance_m"] < nearest:
                        nearest = t["distance_m"]

                    ttc = _calculate_ttc(t["distance_m"], t["relative_speed_mps"])
                    if ttc > 0:
                        if min_ttc < 0 or ttc < min_ttc:
                            min_ttc = ttc

                radar.targets = targets
                radar.nearest_distance_m = nearest
                radar.min_ttc = min_ttc
                radar.valid = True

        except Exception as e:
            print(f"[RadarReader] 读取异常: {e}")

        return radar

    def _extract_latest_frame(self) -> Optional[dict]:
        """Drain every complete buffered frame and return only the newest valid one."""
        latest: Optional[dict] = None
        while True:
            result = self._extract_frame()
            if result is None:
                return latest
            latest = result

    def _extract_frame(self) -> Optional[dict]:
        """从缓冲区提取并解析一帧雷达数据。"""
        while True:
            idx = self._buffer.find(DATA_HEADER)
            if idx < 0:
                if len(self._buffer) > 4096:
                    self._buffer.clear()
                return None

            if idx > 0:
                del self._buffer[:idx]
                continue

            if len(self._buffer) < 10:
                return None

            payload_len = struct.unpack("<H", self._buffer[4:6])[0]
            frame_total = 4 + 2 + payload_len + 4

            if len(self._buffer) < frame_total:
                return None

            frame = bytes(self._buffer[:frame_total])
            del self._buffer[:frame_total]

            result = parse_radar_frame(
                frame,
                self.approaching_direction_code,
                self.angle_sign,
            )
            if result is not None:
                return result

    # ---- mock ----

    def _read_mock(self, ts: float) -> RadarData:
        """模拟雷达数据。

        mock 表示为正常工作的雷达，valid 始终为 True。
        有目标时 targets 非空、nearest/min_ttc 反映真实值；
        无目标时 targets=[]、nearest/min_ttc=-1.0。
        """
        target_count = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
        targets: List[RadarTarget] = []
        nearest = -1.0
        min_ttc = -1.0

        for i in range(target_count):
            dist = round(random.uniform(2.0, 30.0), 1)
            speed = round(random.uniform(-10.0, 10.0), 2)
            angle = round(random.uniform(-30.0, 30.0), 1)
            confidence = round(random.uniform(0.5, 0.99), 2)

            rt = RadarTarget(
                target_id=i,
                distance_m=dist,
                relative_speed_mps=speed,
                angle_deg=angle,
                confidence=confidence,
            )
            targets.append(rt)

            if nearest < 0 or dist < nearest:
                nearest = dist

            ttc = _calculate_ttc(dist, speed)
            if ttc > 0:
                if min_ttc < 0 or ttc < min_ttc:
                    min_ttc = round(ttc, 1)

        targets.sort(key=lambda t: t.distance_m)

        # mock 雷达始终正常工作，valid=True
        # 无目标时 targets=[], nearest/min_ttc 保持 -1.0
        return RadarData(
            timestamp=ts,
            valid=True,
            targets=targets,
            nearest_distance_m=nearest,
            min_ttc=min_ttc,
        )
