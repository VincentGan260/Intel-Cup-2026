"""传感器统一基类，定义 Reader 标准接口。

所有传感器 Reader 必须实现以下方法：
  start()      — 打开硬件连接
  stop()       — 关闭硬件连接
  read_once()  — 从硬件读取一次数据，返回对应数据类
  get_latest() — 返回最近一次 read_once() 的结果（不阻塞）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
