"""传感器统一基类，定义 Reader 标准接口。

所有传感器 Reader 必须实现以下方法：
  start()      — 打开硬件连接
  stop()       — 关闭硬件连接
  read_once()  — 从硬件读取一次数据，返回对应数据类
  get_latest() — 返回最近一次 read_once() 的结果（不阻塞）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Optional

from src.fusion.data_types import SensorBase


class BaseSensorReader(ABC):
    """传感器读取器基类。

    mode: "real" — 真实硬件 / "mock" — 模拟数据
    config: 来自 configs/sensor_ports.yaml 中对应传感器的配置字典
    """

    def __init__(self, mode: str = "mock", config: Optional[dict] = None) -> None:
        self.mode = mode
        self.config = config or {}
        self._latest: Optional[SensorBase] = None
        self._serial_reconnect_interval_sec = max(
            0.5, float(self.config.get("reconnect_interval_sec", 2.0)))
        self._serial_next_reconnect_at = 0.0
        self._serial_reconnect_enabled = False

    def _open_serial_with_retry(
        self,
        *,
        port: str,
        baudrate: int,
        timeout: float,
        label: str,
    ) -> bool:
        """Open a serial device now, or defer a failed retry without blocking callers."""
        if not self.is_real or not self._serial_reconnect_enabled:
            return False
        current = getattr(self, "_serial", None)
        if current is not None and getattr(current, "is_open", False):
            return False
        now_mono = time.monotonic()
        if now_mono < self._serial_next_reconnect_at:
            return False
        self._serial_next_reconnect_at = now_mono + self._serial_reconnect_interval_sec
        try:
            import serial

            self._serial = serial.Serial(port, baudrate, timeout=timeout)
            self._serial_next_reconnect_at = 0.0
            if hasattr(self, "last_error"):
                self.last_error = ""
            print(f"[{label}] 已连接串口 {port} @ {baudrate}")
            return True
        except Exception as exc:
            self._serial = None
            if hasattr(self, "last_error"):
                self.last_error = str(exc)
            print(f"[{label}] 串口暂不可用，{self._serial_reconnect_interval_sec:.1f}s 后重试: {exc}")
            return False

    def _mark_serial_disconnected(self, exc: Exception, *, label: str) -> None:
        """Close a failed handle and schedule a bounded reconnect attempt."""
        current = getattr(self, "_serial", None)
        if current is not None:
            try:
                current.close()
            except Exception:
                pass
        self._serial = None
        self._serial_next_reconnect_at = time.monotonic() + self._serial_reconnect_interval_sec
        if hasattr(self, "last_error"):
            self.last_error = str(exc)
        print(f"[{label}] 串口连接中断，将自动恢复: {exc}")

    def _disable_serial_reconnect(self) -> None:
        self._serial_reconnect_enabled = False
        current = getattr(self, "_serial", None)
        if current is not None:
            try:
                current.close()
            except Exception:
                pass
        self._serial = None

    @abstractmethod
    def start(self) -> None:
        """打开硬件连接。mock 模式下通常无需操作。"""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """关闭硬件连接。"""
        raise NotImplementedError

    @abstractmethod
    def read_once(self) -> SensorBase:
        """从硬件（或模拟数据）读取一次，返回对应的数据类实例。

        返回的数据类实例中 timestamp 由调用者设置或由读取方法内部设置。
        """
        raise NotImplementedError

    def get_latest(self) -> Optional[SensorBase]:
        """返回最近一次 read_once() 的结果。

        未调用过 read_once() 时返回 None。
        """
        return self._latest

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    @property
    def is_real(self) -> bool:
        return self.mode == "real"
