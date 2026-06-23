"""线程安全的状态池。

DashboardStateStore 是前端状态的唯一数据源。
提供 set_state / get_state，由后台更新线程写入，由 FastAPI 路由读取。
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Dict


class DashboardStateStore:
    """线程安全的系统状态存储器。

    内部使用 threading.Lock 保护读写操作。
    get_state() 返回深拷贝，避免并发修改。
    _version 递增计数器，供 WebSocket 端点检测状态变更。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "timestamp": 0.0,
            "risk_score": 0.0,
            "risk_level": 0,
            "risk_label": "低风险",
            "risk_items": {
                "obs": 0.0,
                "dist": 0.0,
                "pose": 0.0,
                "speed": 0.0,
            },
            "weights": {
                "obs": 0.35,
                "dist": 0.35,
                "pose": 0.15,
                "speed": 0.15,
            },
            "sensors": {
                "camera": False,
                "vision": "mock",
                "radar": "mock",
                "imu": "mock",
                "gps": "mock",
            },
            "mode": "mock",
            "message": "dashboard initialized",
        }
        self._version = 0

    def set_state(self, state: dict) -> None:
        """写入完整状态（线程安全），版本号递增。"""
        with self._lock:
            self._state = state
            self._version += 1

    def get_state(self) -> dict:
        """读取状态副本（线程安全）。

        返回深拷贝，调用方不能通过返回值修改内部状态。
        """
        with self._lock:
            return copy.deepcopy(self._state)

    def get_version(self) -> int:
        """读取当前版本号，用于 WebSocket 推送判断。"""
        with self._lock:
            return self._version
