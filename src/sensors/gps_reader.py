"""WHEELTEC G60 GPS 读取器封装。

从早期GPS测试脚本提取核心 NMEA 解析逻辑，不直接 import 原脚本。
支持 real / mock 两种模式。

real 模式：
  - 自动识别 G60 USB 串口（VID:PID 1a86:55d4）
  - 从串口读取 NMEA 报文（GGA + RMC，兼容 GN/GP/BD/GL talker）
  - 使用手动 NMEA 解析（不依赖 pynmea2），减少外部依赖
  - 串口默认为 auto，波特率从 configs/sensor_ports.yaml 读取

mock 模式：
  - 返回模拟 GPS 数据，速度在 0~25 km/h 之间随机变化
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from src.fusion.data_types import GPSData, now, gps_kmh_to_mps
from src.sensors.sensor_base import BaseSensorReader

# — NMEA 简单解析器（不依赖 pynmea2） —

G60_USB_VID = 0x1A86
G60_USB_PID = 0x55D4


def _find_g60_ports() -> list[str]:
    """通过 USB VID/PID 查找 G60，兼容 Linux 和 Windows。"""
    try:
        from serial.tools import list_ports

        return sorted(
            port.device
            for port in list_ports.comports()
            if port.vid == G60_USB_VID and port.pid == G60_USB_PID
        )
    except (ImportError, OSError):
        return []


def _nmea_checksum_valid(sentence: str) -> bool:
    """校验 NMEA `$...*HH` XOR checksum。正式采集拒绝残缺/坏句。"""
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        return False
    body, supplied = sentence[1:].rsplit("*", 1)
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    try:
        return checksum == int(supplied[:2], 16)
    except ValueError:
        return False


def _parse_nmea_gga(sentence: str) -> Optional[dict]:
    """解析 GGA 报文：$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47

    返回: {"latitude": float, "longitude": float, "fix_quality": int, "satellites": int} 或 None
    """
    try:
        parts = sentence.split(",")
        if len(parts) < 10:
            return None

        if parts[0].lstrip("$")[-3:] != "GGA":
            return None

        # 纬度: ddmm.mmmm → dd.dddd
        lat_raw = parts[2]
        lat_dir = parts[3]
        if lat_raw:
            dd = float(lat_raw[:2])
            mm = float(lat_raw[2:]) if len(lat_raw) > 2 else 0.0
            latitude = dd + mm / 60.0
            if lat_dir == "S":
                latitude = -latitude
        else:
            latitude = 0.0

        # 经度: dddmm.mmmm → ddd.dddd
        lon_raw = parts[4]
        lon_dir = parts[5]
        if lon_raw:
            dd = float(lon_raw[:3])
            mm = float(lon_raw[3:]) if len(lon_raw) > 3 else 0.0
            longitude = dd + mm / 60.0
            if lon_dir == "W":
                longitude = -longitude
        else:
            longitude = 0.0

        fix_quality = int(parts[6]) if parts[6] else 0
        satellites = int(parts[7]) if parts[7] else 0

        return {
            "latitude": latitude,
            "longitude": longitude,
            "fix_quality": fix_quality,
            "satellites": satellites,
        }
    except (ValueError, IndexError):
        return None


def _parse_nmea_rmc(sentence: str) -> Optional[dict]:
    """解析 RMC 报文：$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A

    返回: {"speed_kn": float, "status": str} 或 None
    """
    try:
        parts = sentence.split(",")
        if len(parts) < 8:
            return None
        if parts[0].lstrip("$")[-3:] != "RMC":
            return None
        status = parts[2]
        speed_kn = float(parts[7]) if parts[7] else 0.0
        return {"speed_kn": speed_kn, "status": status}
    except (ValueError, IndexError):
        return None


class GPSReader(BaseSensorReader):
    """GPS 读取器。"""

    def __init__(
        self,
        mode: str = "mock",
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(mode, config)
        self._serial: Optional["serial.Serial"] = None  # noqa: F821
        self._active_port: str = ""
        self._rx_buffer = bytearray()
        self._mock_speed: float = 0.0
        self._mock_lat: float = 31.2304
        self._mock_lon: float = 121.4737
        self._latest_gga: Optional[dict] = None
        self._latest_rmc: Optional[dict] = None
        self._gga_received_mono: float = 0.0
        self._rmc_received_mono: float = 0.0
        self._gga_received_wall: float = 0.0
        self._rmc_received_wall: float = 0.0
        self._bad_nmea_count: int = 0
        self.bytes_received: int = 0
        self.valid_sentence_count: int = 0
        self.last_byte_monotonic_ns: int = 0
        self.last_valid_sentence_monotonic_ns: int = 0
        self.last_error: str = ""
        self._max_sentence_age_sec = float(self.config.get("max_sentence_age_sec", 2.5))
        self.last_sample_monotonic_ns: int = 0

    def start(self) -> None:
        if self.is_real:
            import serial as _serial

            configured_port = str(self.config.get("port", "auto")).strip()
            detected_ports = _find_g60_ports()

            if not detected_ports:
                self.last_error = "未找到 G60 USB 设备 (VID:PID 1a86:55d4)"
                print(f"[GPSReader] {self.last_error}")
                return
            if configured_port.lower() == "auto":
                if len(detected_ports) != 1:
                    self.last_error = (
                        "检测到多个 G60 串口: " + ", ".join(detected_ports)
                        + "；请在配置中指定 port"
                    )
                    print(f"[GPSReader] {self.last_error}")
                    return
                port = detected_ports[0]
            else:
                ports_by_name = {item.lower(): item for item in detected_ports}
                port = ports_by_name.get(configured_port.lower(), "")
                if not port:
                    self.last_error = (
                        f"配置端口 {configured_port} 不是已识别的 G60；"
                        f"当前 G60: {', '.join(detected_ports)}"
                    )
                    print(f"[GPSReader] {self.last_error}")
                    return

            baudrate = int(self.config.get("baudrate", 9600))

            try:
                # Dashboard 不能被串口超时阻塞；完整行由 _rx_buffer 拼接。
                self._serial = _serial.Serial(port, baudrate, timeout=0)
                self._serial.reset_input_buffer()
                self._active_port = port
                self._rx_buffer.clear()
                self._latest_gga = None
                self._latest_rmc = None
                self._gga_received_mono = 0.0
                self._rmc_received_mono = 0.0
                self._gga_received_wall = 0.0
                self._rmc_received_wall = 0.0
                self.bytes_received = 0
                self.valid_sentence_count = 0
                self._bad_nmea_count = 0
                self.last_byte_monotonic_ns = 0
                self.last_valid_sentence_monotonic_ns = 0
                self.last_error = ""
                print(f"[GPSReader] 已打开串口 {port} @ {baudrate}")
            except Exception as e:
                print(f"[GPSReader] 串口打开失败: {e}")
                self._serial = None
                self._active_port = ""
                self.last_error = str(e)
        else:
            self._mock_speed = 12.0  # 初始速度 12 km/h
            print("[GPSReader] mock 模式启动")

    def stop(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
                print("[GPSReader] 串口已关闭")
            except Exception:
                pass
            self._serial = None
            self._active_port = ""
            self._rx_buffer.clear()
        else:
            print("[GPSReader] 已停止")

    def read_once(self) -> GPSData:
        ts = now()
        if self.is_real:
            result = self._read_real(ts)
        else:
            result = self._read_mock(ts)
        self._latest = result
        return result

    def get_diagnostics(self) -> dict:
        current_mono = time.monotonic()
        return {
            "port": self._active_port,
            "bytes_received": int(self.bytes_received),
            "valid_sentence_count": int(self.valid_sentence_count),
            "bad_sentence_count": int(self._bad_nmea_count),
            "last_byte_monotonic_ns": int(self.last_byte_monotonic_ns),
            "last_valid_sentence_monotonic_ns": int(self.last_valid_sentence_monotonic_ns),
            "gga_fresh": bool(
                self._latest_gga is not None
                and current_mono - self._gga_received_mono <= self._max_sentence_age_sec
            ),
            "rmc_fresh": bool(
                self._latest_rmc is not None
                and current_mono - self._rmc_received_mono <= self._max_sentence_age_sec
            ),
            "last_error": self.last_error,
        }

    def _read_real(self, ts: float) -> GPSData:
        """真实串口读取 NMEA 数据。"""
        gps = GPSData(timestamp=ts, valid=False)
        try:
            if self._serial is None or not self._serial.is_open:
                return gps

            # 一次只读当前已到达的字节，永不等待串口 timeout。
            available = min(int(self._serial.in_waiting), 4096)
            if available > 0:
                raw = self._serial.read(available)
                self.bytes_received += len(raw)
                self.last_byte_monotonic_ns = time.monotonic_ns()
                self._rx_buffer.extend(raw)

            # 异常数据不得让缓冲区无限增长。
            if len(self._rx_buffer) > 16384:
                last_start = self._rx_buffer.rfind(b"$")
                self._rx_buffer = self._rx_buffer[last_start:] if last_start >= 0 else bytearray()

            while b"\n" in self._rx_buffer:
                raw_line, _, remainder = self._rx_buffer.partition(b"\n")
                self._rx_buffer = bytearray(remainder)
                line = raw_line.strip().decode("ascii", errors="ignore")

                if not _nmea_checksum_valid(line):
                    self._bad_nmea_count += 1
                    continue

                received_mono = time.monotonic()
                received_wall = now()
                self.valid_sentence_count += 1
                self.last_valid_sentence_monotonic_ns = time.monotonic_ns()

                message = line[1:].split(",", 1)[0]
                kind = message[-3:]
                if kind == "GGA":
                    parsed = _parse_nmea_gga(line)
                    if parsed:
                        self._latest_gga = parsed
                        self._gga_received_mono = received_mono
                        self._gga_received_wall = received_wall

                elif kind == "RMC":
                    parsed = _parse_nmea_rmc(line)
                    if parsed:
                        self._latest_rmc = parsed
                        self._rmc_received_mono = received_mono
                        self._rmc_received_wall = received_wall

            current_mono = time.monotonic()
            gga_fresh = (
                self._latest_gga is not None
                and current_mono - self._gga_received_mono <= self._max_sentence_age_sec
            )
            rmc_fresh = (
                self._latest_rmc is not None
                and current_mono - self._rmc_received_mono <= self._max_sentence_age_sec
            )
            if gga_fresh:
                gga = self._latest_gga or {}
                gps.latitude = float(gga.get("latitude") or 0.0)
                gps.longitude = float(gga.get("longitude") or 0.0)
                gps.fix_quality = int(gga.get("fix_quality", 0))
                gps.satellites = int(gga.get("satellites", 0))
            if rmc_fresh:
                rmc = self._latest_rmc or {}
                gps.speed_kmh = float(rmc.get("speed_kn", 0.0)) * 1.852
                gps.speed_mps = gps_kmh_to_mps(gps.speed_kmh)
            gps.valid = bool(
                gga_fresh and rmc_fresh
                and gps.fix_quality > 0
                and (self._latest_rmc or {}).get("status") == "A"
                and -90.0 <= gps.latitude <= 90.0
                and -180.0 <= gps.longitude <= 180.0
            )
            if gga_fresh or rmc_fresh:
                gps.timestamp = max(self._gga_received_wall, self._rmc_received_wall)
            # GPSData由GGA位置和RMC速度共同组成。两者均存在时使用较早到达者，
            # 这样任一组成报文过旧都会体现在与相机的同步差中，不会被较新的另一句掩盖。
            if gga_fresh and rmc_fresh:
                self.last_sample_monotonic_ns = int(
                    min(self._gga_received_mono, self._rmc_received_mono) * 1_000_000_000)
            elif gga_fresh:
                self.last_sample_monotonic_ns = int(self._gga_received_mono * 1_000_000_000)
            elif rmc_fresh:
                self.last_sample_monotonic_ns = int(self._rmc_received_mono * 1_000_000_000)

        except Exception as e:
            print(f"[GPSReader] 读取异常: {e}")
            self.last_error = str(e)

        return gps

    def _read_mock(self, ts: float) -> GPSData:
        """模拟 GPS 数据。"""
        self._mock_speed += random.uniform(-2.0, 2.0)
        self._mock_speed = max(0.0, min(35.0, self._mock_speed))

        self._mock_lat += random.uniform(-0.0001, 0.0001)
        self._mock_lon += random.uniform(-0.0001, 0.0001)

        speed_kmh = round(self._mock_speed, 1)
        gps = GPSData(
            timestamp=ts,
            valid=True,
            speed_kmh=speed_kmh,
            speed_mps=gps_kmh_to_mps(speed_kmh),
            latitude=round(self._mock_lat, 6),
            longitude=round(self._mock_lon, 6),
            fix_quality=1,
            satellites=random.randint(4, 12),
        )
        return gps
