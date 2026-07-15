"""系统级统一数据结构。

所有跨模块数据类定义在此，供 sensors / fusion / actuator / debug / logging 消费。
每个数据类都包含 timestamp（Unix 时间戳，秒）和 valid（是否有效）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
#  传感器数据基类
# ============================================================


@dataclass
class SensorBase:
    """所有传感器数据的基类字段。"""

    timestamp: float = 0.0  # Unix 时间戳，秒
    valid: bool = False  # 当前数据是否有效
    source: str = ""  # 数据来源，如 "gps" / "imu" / "radar" / "camera"


# ============================================================
#  GPS 数据
# ============================================================


@dataclass
class GPSData(SensorBase):
    """GPS 定位与速度数据。

    字段说明：
      speed_kmh / speed_mps — 对地速度
      latitude / longitude  — WGS84 经纬度
      fix_quality           — 定位质量（0=无效, 1=单点, 2=差分）
      satellites            — 搜星数
    """

    speed_kmh: float = 0.0
    speed_mps: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    fix_quality: int = 0
    satellites: int = 0

    def __post_init__(self) -> None:
        self.source = "gps"


# ============================================================
#  IMU 数据
# ============================================================


@dataclass
class IMUData(SensorBase):
    """WT61C 六轴姿态数据。

    字段说明：
      roll / pitch / yaw  — 欧拉角（度）
      acc_x/y/z           — 三轴加速度（m/s²）
      gyro_x/y/z          — 三轴角速度（°/s）
      brake_score          — 急刹特征评分 [0, 1]
      bump_score           — 颠簸特征评分 [0, 1]
      tilt_score           — 侧倾特征评分 [0, 1]
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0

    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    brake_score: float = 0.0
    bump_score: float = 0.0
    tilt_score: float = 0.0

    def __post_init__(self) -> None:
        self.source = "imu"


# ============================================================
#  毫米波雷达数据
# ============================================================


@dataclass
class RadarTarget:
    """单个雷达目标。"""

    target_id: int = 0
    distance_m: float = 0.0  # 米
    relative_speed_mps: float = 0.0  # 相对速度（m/s，负值=接近）
    angle_deg: float = 0.0  # 角度（度，正=右侧）
    confidence: float = 0.0  # 置信度 [0, 1]


@dataclass
class RadarData(SensorBase):
    """毫米波雷达（HLK-LD2451）探测结果。

    字段说明：
      targets              — 当前帧所有目标列表
      nearest_distance_m   — 最近目标距离（米），无目标时为 -1
      min_ttc              — 最小碰撞时间（秒），无接近目标时为 -1
    """

    targets: List[RadarTarget] = field(default_factory=list)
    nearest_distance_m: float = -1.0
    min_ttc: float = -1.0

    def __post_init__(self) -> None:
        self.source = "radar"


# ============================================================
#  视觉数据
# ============================================================


@dataclass
class VisionObject:
    """单个视觉检测目标，供融合模块消费。"""

    class_name: str = ""
    risk_class: str = ""
    confidence: float = 0.0
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)  # x1, y1, x2, y2
    distance_m: float = -1.0  # 视觉估计距离，未启用时为 -1
    in_drivable_area: Optional[bool] = None
    visual_risk: float = 0.0  # R_obs 单项风险 [0, 1]


@dataclass
class VisionData(SensorBase):
    """视觉管线（检测 + 分割）输出。"""

    objects: List[VisionObject] = field(default_factory=list)
    person_count: int = 0  # 行人数量
    vehicle_count: int = 0  # 车辆（机动车+非机动车）数量
    max_confidence: float = 0.0  # 最高置信度
    drivable_area_ratio: float = 0.0  # 可行驶区域占画面比例
    center_drivable_score: float = 0.0  # 画面中心区域可行驶评分
    max_visual_risk: float = 0.0  # 单帧最大视觉风险 R_obs [0, 1]
    detection_inference_ms: float = 0.0
    segmentation_inference_ms: float = 0.0
    pipeline_inference_ms: float = 0.0

    def __post_init__(self) -> None:
        self.source = "camera"


# ============================================================
#  马达控制指令
# ============================================================


@dataclass
class MotorCommand:
    """震动马达控制指令。

    risk_level: 0=低/无风险, 1=中风险, 2=高风险
    pattern: "silent" / "short_pulse" / "strong_continuous"
    duration_ms: 震动持续时间（毫秒）
    """

    timestamp: float = 0.0
    risk_level: int = 0
    risk_score: float = 0.0
    intensity: float = 0.0
    pattern: str = "silent"
    duration_ms: int = 0


# ============================================================
#  系统综合状态（融合 + 日志输出）
# ============================================================


@dataclass
class SystemState:
    """全系统综合状态，每条日志记录一条。

    这是最终写入 CSV 日志的结构，包含所有关键字段。
    """

    # — 时间 —
    timestamp: float = 0.0

    # — GPS —
    gps_valid: bool = False
    gps_speed_kmh: float = 0.0
    gps_latitude: float = 0.0
    gps_longitude: float = 0.0
    gps_fix_quality: int = 0
    gps_satellites: int = 0

    # — IMU —
    imu_valid: bool = False
    imu_roll: float = 0.0
    imu_pitch: float = 0.0
    imu_brake_score: float = 0.0
    imu_bump_score: float = 0.0
    imu_tilt_score: float = 0.0

    # — 雷达 —
    radar_valid: bool = False
    radar_target_count: int = 0
    radar_nearest_m: float = -1.0
    radar_min_ttc: float = -1.0

    # — 视觉 —
    vision_valid: bool = False
    vision_object_count: int = 0
    vision_person_count: int = 0
    vision_vehicle_count: int = 0
    vision_max_risk: float = 0.0
    vision_drivable_ratio: float = 0.0

    # — 综合风险 —
    risk_obs: float = 0.0  # R_obs
    risk_dist: float = 0.0  # R_dist
    risk_pose: float = 0.0  # R_pose
    risk_speed: float = 0.0  # R_speed
    risk_total: float = 0.0  # 综合风险 R
    risk_level: int = 0  # 0=低, 1=中, 2=高

    # — 马达 —
    motor_pattern: str = "silent"
    motor_duration_ms: int = 0


# ============================================================
#  融合模块输入（Synchronizer 输出，RiskModel 输入）
# ============================================================


@dataclass
class FusionInput:
    """统一融合帧输入，携带当前时刻所有传感器数据。

    由 synchronizer 组装，由 risk_model 消费。
    当前时刻某模块无数据时，对应字段使用默认无效值。
    """

    timestamp: float = 0.0
    gps: GPSData = field(default_factory=GPSData)
    imu: IMUData = field(default_factory=IMUData)
    radar: RadarData = field(default_factory=RadarData)
    vision: VisionData = field(default_factory=VisionData)

    # 当前是否已接入视觉模块
    vision_enabled: bool = False


def now() -> float:
    """获取当前 Unix 时间戳（秒）。"""
    return time.time()


def gps_kmh_to_mps(kmh: float) -> float:
    """公里/小时 → 米/秒。"""
    return kmh / 3.6


def gps_mps_to_kmh(mps: float) -> float:
    """米/秒 → 公里/小时。"""
    return mps * 3.6
