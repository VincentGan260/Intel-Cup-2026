"""多传感器数据同步器。

将 GPS、IMU、雷达、视觉四路独立读取的传感器数据，
按时间戳对齐后封装为统一的 FusionInput，供风险融合模块消费。

当前阶段视觉尚未接入，VisionData 由调用者提供或使用默认空数据。
"""

from __future__ import annotations

from typing import Optional

from src.fusion.data_types import (
    FusionInput,
    GPSData,
    IMUData,
    RadarData,
    VisionData,
    now,
)


class Synchronizer:
    """传感器数据同步器。

    维护每个传感器「最新有效帧」，按需组装 FusionInput。
    """

    def __init__(self, vision_enabled: bool = False) -> None:
        self.vision_enabled = vision_enabled
        self._latest_gps: Optional[GPSData] = None
        self._latest_imu: Optional[IMUData] = None
        self._latest_radar: Optional[RadarData] = None
        self._latest_vision: Optional[VisionData] = None

    def update_gps(self, data: GPSData) -> None:
        """更新最新 GPS 数据。"""
        self._latest_gps = data

    def update_imu(self, data: IMUData) -> None:
        """更新最新 IMU 数据。"""
        self._latest_imu = data

    def update_radar(self, data: RadarData) -> None:
        """更新最新雷达数据。"""
        self._latest_radar = data

    def update_vision(self, data: VisionData) -> None:
        """更新最新视觉数据。"""
        self._latest_vision = data

    def build_frame(self) -> FusionInput:
        """基于各传感器最新数据组装当前融合帧。

        如果某传感器从未更新过数据，将使用该数据类的默认无效值。
        """
        ts = now()
        gps = self._latest_gps if self._latest_gps is not None else GPSData(timestamp=ts)
        imu = self._latest_imu if self._latest_imu is not None else IMUData(timestamp=ts)
        radar = self._latest_radar if self._latest_radar is not None else RadarData(timestamp=ts)
        vision = self._latest_vision if self._latest_vision is not None else VisionData(timestamp=ts)

        return FusionInput(
            timestamp=ts,
            gps=gps,
            imu=imu,
            radar=radar,
            vision=vision,
            vision_enabled=self.vision_enabled,
        )

    def has_any_data(self) -> bool:
        """至少有一个传感器更新过数据。"""
        return any(
            x is not None
            for x in [self._latest_gps, self._latest_imu, self._latest_radar]
        )
