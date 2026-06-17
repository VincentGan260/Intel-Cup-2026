"""无视觉闭环 mock 整合测试。

完整闭环（mock 模式）：
  GPSReader + IMUReader + RadarReader
    → Synchronizer
    → RiskModel + RiskLevelClassifier
    → MotorController
    → CsvLogger + ConsoleViewer

运行方式：
  python scripts/test_closed_loop_mock.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.data_types import SystemState, now
from src.fusion.risk_level import RiskLevelClassifier
from src.fusion.risk_model import RiskModel
from src.fusion.synchronizer import Synchronizer
from src.sensors.gps_reader import GPSReader
from src.sensors.imu_reader import IMUReader
from src.sensors.radar_reader import RadarReader
from src.actuator.motor_controller import MotorController
from src.debug.console_viewer import ConsoleViewer
from src.utils.logger import CsvLogger


# ============================================================
#  主循环
# ============================================================


def main():
    # 参数
    loop_count = 20
    loop_interval = 0.2  # 5Hz
    log_path = "logs/closed_loop_mock.csv"

    # === 初始化模块 ===
    print("=" * 65)
    print("  无视觉闭环整合测试 (mock 模式)")
    print(f"  运行次数: {loop_count} 次 @ {1/loop_interval:.0f}Hz")
    print("=" * 65)

    gps_reader = GPSReader(mode="mock")
    imu_reader = IMUReader(mode="mock")
    radar_reader = RadarReader(mode="mock")
    motor = MotorController(mode="mock")
    sync = Synchronizer(vision_enabled=False)
    risk_model = RiskModel()
    classifier = RiskLevelClassifier()
    viewer = ConsoleViewer()
    logger = CsvLogger(log_path)

    # === 启动 ===
    try:
        gps_reader.start()
        imu_reader.start()
        radar_reader.start()
        motor.start()
        # 日志文件在第一次 write 时自动创建

        for i in range(loop_count):
            frame_start = time.time()

            # ── 数据采集 ──
            gps = gps_reader.read_once()
            imu = imu_reader.read_once()
            radar = radar_reader.read_once()

            # ── 同步 ──
            sync.update_gps(gps)
            sync.update_imu(imu)
            sync.update_radar(radar)
            fusion = sync.build_frame()

            # ── 风险融合 ──
            risk_items, weights = risk_model.compute(fusion)
            level, label = classifier.classify(risk_items["risk_score"])
            risk_score = risk_items["risk_score"]

            # ── 马达控制 ──
            if level == 0:
                motor.alert_low()
            elif level == 1:
                motor.alert_medium(risk_score)
            else:
                motor.alert_high(risk_score)
            motor_cmd = motor.get_latest()

            # ── 组装 SystemState ──
            ts = now()
            state = SystemState(
                timestamp=ts,
                gps_valid=gps.valid,
                gps_speed_kmh=gps.speed_kmh,
                imu_valid=imu.valid,
                imu_roll=imu.roll,
                imu_pitch=imu.pitch,
                imu_brake_score=imu.brake_score,
                imu_bump_score=imu.bump_score,
                imu_tilt_score=imu.tilt_score,
                radar_valid=radar.valid,
                radar_nearest_m=radar.nearest_distance_m,
                radar_min_ttc=radar.min_ttc,
                vision_valid=False,
                vision_object_count=len(radar.targets),
                vision_max_risk=0.0,
                vision_drivable_ratio=0.0,
                risk_obs=risk_items["R_obs"],
                risk_dist=risk_items["R_dist"],
                risk_pose=risk_items["R_pose"],
                risk_speed=risk_items["R_speed"],
                risk_total=risk_score,
                risk_level=level,
                motor_pattern=motor_cmd.pattern if motor_cmd else "silent",
                motor_duration_ms=motor_cmd.duration_ms if motor_cmd else 0,
            )

            # ── 显示 + 日志 ──
            viewer.display(state)
            logger.write(state)

            # ── 维持循环频率 ──
            elapsed = time.time() - frame_start
            sleep_time = max(0, loop_interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[主循环] 用户手动终止")

    except Exception as e:
        print(f"\n[主循环] 异常: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # === 安全关闭 ===
        print("\n" + "=" * 65)
        print("  正在关闭所有模块...")
        print("=" * 65)

        gps_reader.stop()
        imu_reader.stop()
        radar_reader.stop()
        motor.stop()
        logger.close()

        print("\n[完成] 无视觉闭环测试结束")
        print(f"        日志: {os.path.abspath(logger.file_path)}")


if __name__ == "__main__":
    main()
