"""命令行实时系统状态显示器。

接收 SystemState 数据，在终端打印格式化的状态表格。
"""
from __future__ import annotations

from src.fusion.data_types import SystemState


class ConsoleViewer:
    """控制台状态显示器。

    每一帧调用 display(state) 刷新终端显示。
    """

    def __init__(self) -> None:
        self._frame_count = 0

    def display(self, state: SystemState) -> None:
        """显示一帧系统状态。"""
        self._frame_count += 1
        ts_str = f"{state.timestamp:.3f}"
        ts_local = time_str_from_ts(state.timestamp)

        lines = [
            f"\n{'=' * 65}",
            f"  [帧 {self._frame_count:3d}] 时间: {ts_str} ({ts_local})",
            f"{'=' * 65}",
            # GPS
            f"  GPS    | 有效={state.gps_valid}  速度={state.gps_speed_kmh:6.1f} km/h",
            # IMU
            f"  IMU    | 有效={state.imu_valid}  "
            f"roll={state.imu_roll:+6.1f}  pitch={state.imu_pitch:+6.1f}  "
            f"brake={state.imu_brake_score:.2f}  bump={state.imu_bump_score:.2f}  "
            f"tilt={state.imu_tilt_score:.2f}",
            # 雷达
            f"  Radar  | 有效={state.radar_valid}  最近={state.radar_nearest_m:5.1f}m  "
            f"TTC={state.radar_min_ttc:5.1f}s",
            # 视觉
            f"  Vision | 有效={state.vision_valid}  目标={state.vision_object_count}  "
            f"行人={state.vision_person_count} 车={state.vision_vehicle_count}  "
            f"max_risk={state.vision_max_risk:.2f}",
            # 风险项
            f"  Risk   | R_obs={state.risk_obs:.3f}  R_dist={state.risk_dist:.3f}  "
            f"R_pose={state.risk_pose:.3f}  R_speed={state.risk_speed:.3f}",
            # 综合
            f"  Fusion | score={state.risk_total:.3f}  等级={state.risk_level}  "
            f"马达={state.motor_pattern}",
            f"{'=' * 65}",
        ]
        print("\n".join(lines))


def time_str_from_ts(ts: float) -> str:
    """将 Unix 时间戳转为本地时间字符串。"""
    import datetime

    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:12]
