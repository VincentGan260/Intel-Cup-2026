"""GPS 读取器封装。

从 gps_test1.py 提取核心 NMEA 解析逻辑，不直接 import 原脚本。
支持 real / mock 两种模式。

real 模式：
  - 从串口读取 NMEA 报文（GPGGA/GNGGA + GPRMC/GNRMC）
  - 使用手动 NMEA 解析（不依赖 pynmea2），减少外部依赖
  - 端口/波特率从 configs/sensor_ports.yaml 读取

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


def _parse_nmea_gga(sentence: str) -> Optional[dict]:
    """解析 GGA 报文：$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47

    返回: {"latitude": float, "longitude": float, "fix_quality": int, "satellites": int} 或 None
    """
    try:
        parts = sentence.split(",")
        if len(parts) < 10:
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
        status = parts[2]
        speed_kn = float(parts[6]) if parts[6] else 0.0
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
        self._mock_speed: float = 0.0
        self._mock_lat: float = 31.2304
        self._mock_lon: float = 121.4737

    def start(self) -> None:
        if self.is_real:
            import serial as _serial

            port = self.config.get("port", "/dev/ttyS5")
            baudrate = self.config.get("baudrate", 9600)
            timeout = self.config.get("timeout", 1.0)

            try:
                self._serial = _serial.Serial(port, baudrate, timeout=timeout)
                print(f"[GPSReader] 已打开串口 {port} @ {baudrate}")
            except Exception as e:
                print(f"[GPSReader] 串口打开失败: {e}")
                self._serial = None
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
        else:
            print("[GPSReader] 已停止")

    def read_once(self) -> GPSData:
        ts = now()
        if self.is_real and self._serial is not None:
            result = self._read_real(ts)
        else:
            result = self._read_mock(ts)
        self._latest = result
        return result

    def _read_real(self, ts: float) -> GPSData:
        """真实串口读取 NMEA 数据。"""
        gps = GPSData(timestamp=ts, valid=False)
        try:
            if self._serial is None or not self._serial.is_open:
                return gps

            speed_kmh = 0.0
            lat, lon = 0.0, 0.0
            fix_q, sats = 0, 0
            has_gga = has_rmc = False

            for _ in range(20):  # 读 20 行或直到超时
                raw = self._serial.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("ascii", errors="ignore").strip()
                except Exception:
                    continue

                if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    parsed = _parse_nmea_gga(line)
                    if parsed:
                        lat = parsed["latitude"]
                        lon = parsed["longitude"]
                        fix_q = parsed["fix_quality"]
                        sats = parsed["satellites"]
                        has_gga = True

                elif line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                    parsed = _parse_nmea_rmc(line)
                    if parsed and parsed["status"] == "A":
                        speed_kmh = parsed["speed_kn"] * 1.852
                        has_rmc = True

                if has_gga and has_rmc:
                    break

            if has_gga or has_rmc:
                gps.speed_kmh = speed_kmh
                gps.speed_mps = gps_kmh_to_mps(speed_kmh)
                gps.latitude = lat
                gps.longitude = lon
                gps.fix_quality = fix_q
                gps.satellites = sats
                gps.valid = has_rmc and fix_q > 0

        except Exception as e:
            print(f"[GPSReader] 读取异常: {e}")

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
