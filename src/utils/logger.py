"""CSV 日志记录器。

将 SystemState 按行写入 CSV 文件，供离线分析。
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import List, Optional

from src.fusion.data_types import SystemState

# CSV 表头（与 SystemState 字段一一对应）
CSV_HEADER = [
    "timestamp",
    "gps_valid",
    "gps_speed_kmh",
    "imu_valid",
    "roll",
    "pitch",
    "brake_score",
    "bump_score",
    "tilt_score",
    "radar_valid",
    "radar_target_count",
    "nearest_distance_m",
    "min_ttc",
    "vision_valid",
    "vision_object_count",
    "vision_person_count",
    "vision_vehicle_count",
    "R_obs",
    "R_dist",
    "R_pose",
    "R_speed",
    "risk_score",
    "risk_level",
    "motor_pattern",
]


class CsvLogger:
    """CSV 日志记录器。

    使用方式：
      logger = CsvLogger("logs/closed_loop.csv")
      logger.write(state)
      logger.close()
    """

    def __init__(self, file_path: str, header: Optional[List[str]] = None) -> None:
        self.file_path = file_path
        self.header = header or CSV_HEADER
        self._file = None
        self._writer = None
        self._row_count = 0

        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> None:
        """打开文件并写入表头。"""
        self._file = open(self.file_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)
        self._file.flush()
        print(f"[CsvLogger] 日志文件: {self.file_path}")

    def write(self, state: SystemState, risk_items: Optional[dict] = None) -> None:
        """写入一行状态数据。

        Args:
            state: 系统综合状态
            risk_items: 可选的原始风险项（含 R_obs, R_dist, R_pose, R_speed）
        """
        if self._writer is None:
            self.open()

        row = [
            f"{state.timestamp:.3f}",
            str(state.gps_valid),
            f"{state.gps_speed_kmh:.2f}",
            str(state.imu_valid),
            f"{state.imu_roll:.2f}",
            f"{state.imu_pitch:.2f}",
            f"{state.imu_brake_score:.3f}",
            f"{state.imu_bump_score:.3f}",
            f"{state.imu_tilt_score:.3f}",
            str(state.radar_valid),
            str(state.radar_target_count),
            f"{state.radar_nearest_m:.2f}",
            f"{state.radar_min_ttc:.2f}",
            str(state.vision_valid),
            str(state.vision_object_count),
            str(state.vision_person_count),
            str(state.vision_vehicle_count),
            f"{state.risk_obs:.3f}",
            f"{state.risk_dist:.3f}",
            f"{state.risk_pose:.3f}",
            f"{state.risk_speed:.3f}",
            f"{state.risk_total:.3f}",
            str(state.risk_level),
            state.motor_pattern,
        ]
        self._writer.writerow(row)
        self._row_count += 1

        # 每 5 行 flush 一次
        if self._row_count % 5 == 0:
            self._file.flush()

    def close(self) -> None:
        """关闭日志文件。"""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
            print(f"[CsvLogger] 已写入 {self._row_count} 行 → {self.file_path}")

    @property
    def row_count(self) -> int:
        return self._row_count
