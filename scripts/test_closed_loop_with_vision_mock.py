"""带视觉的闭环整合测试（mock 传感器 + 真实视觉）。

数据流：
  真实/样例图片 → VisionAdapter → VisionData
  GPSReader(mock) + IMUReader(mock) + RadarReader(mock) → GPSData/IMUData/RadarData
    → Synchronizer(vision_enabled=True)
    → RiskModel (含 R_obs)
    → MotorController(mock)
    → ConsoleViewer + CsvLogger

运行方式：
  python scripts/test_closed_loop_with_vision_mock.py
  python scripts/test_closed_loop_with_vision_mock.py --source 图片路径
  python scripts/test_closed_loop_with_vision_mock.py --camera
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.data_types import SystemState, now
from src.fusion.risk_level import RiskLevelClassifier
from src.fusion.risk_model import RiskModel
from src.fusion.synchronizer import Synchronizer
from src.fusion.vision_adapter import VisionAdapter
from src.sensors.gps_reader import GPSReader
from src.sensors.imu_reader import IMUReader
from src.sensors.radar_reader import RadarReader
from src.actuator.motor_controller import MotorController
from src.debug.console_viewer import ConsoleViewer
from src.utils.logger import CsvLogger


def main():
    parser = argparse.ArgumentParser(description="带视觉的闭环整合测试")
    parser.add_argument("--source", type=str, default=None, help="图片路径")
    parser.add_argument("--camera", action="store_true", help="摄像头模式")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头编号")
    parser.add_argument("--loops", type=int, default=20, help="运行次数")
    args = parser.parse_args()

    loop_count = args.loops
    loop_interval = 0.2  # 5Hz
    log_path = "logs/closed_loop_with_vision_mock.csv"

    print("=" * 65)
    print("  带视觉闭环整合测试 (传感器 mock + 视觉真实)")
    print(f"  运行次数: {loop_count} 次 @ {1/loop_interval:.0f}Hz")
    print(f"  视觉模式: {'摄像头' if args.camera else '图片'}")
    print("=" * 65)

    # === 初始化 ===
    gps_reader = GPSReader(mode="mock")
    imu_reader = IMUReader(mode="mock")
    radar_reader = RadarReader(mode="mock")
    motor = MotorController(mode="mock")
    sync = Synchronizer(vision_enabled=True)
    risk_model = RiskModel()
    classifier = RiskLevelClassifier()
    viewer = ConsoleViewer()
    logger = CsvLogger(log_path)

    vision = VisionAdapter(
        pipeline_config_path="configs/vision/vision_pipeline.yaml",
        vision_enabled=True,
        use_camera=args.camera,
        camera_id=args.camera_id,
    )

    # === 启动 ===
    success = False
    try:
        gps_reader.start()
        imu_reader.start()
        radar_reader.start()
        motor.start()
        vision.start()

        # 加载静态图片（若指定）
        static_frame = None
        if args.source and not args.camera:
            from src.vision.common.preprocess import read_image_bgr

            static_frame = read_image_bgr(args.source)
            h, w = static_frame.shape[:2]
            print(f"  加载图片: {args.source} ({w}x{h})")
        elif not args.camera:
            # 默认用 bus 图
            try:
                from src.vision.common.preprocess import load_image_bgr_from_source
                from pathlib import Path

                static_frame, src_name = load_image_bgr_from_source(
                    "https://ultralytics.com/images/bus.jpg",
                    Path(__file__).resolve().parent.parent,
                )
                print(f"  加载默认图片: {src_name}")
            except Exception as e:
                print(f"  无法加载默认图片: {e}")

        for i in range(loop_count):
            frame_start = time.time()

            # ── 视觉（真实图片或摄像头）──
            if args.camera:
                vision_data = vision.process()
            else:
                vision_data = vision.process(static_frame)

            # ── 传感器（mock）──
            gps = gps_reader.read_once()
            imu = imu_reader.read_once()
            radar = radar_reader.read_once()

            # ── 同步 ──
            sync.update_gps(gps)
            sync.update_imu(imu)
            sync.update_radar(radar)
            sync.update_vision(vision_data)
            fusion = sync.build_frame()

            # ── 风险融合 ──
            risk_items, weights = risk_model.compute(fusion)
            level, label = classifier.classify(risk_items["risk_score"])
            risk_score = risk_items["risk_score"]

            # ── 马达 ──
            if level == 0:
                motor.alert_low()
            elif level == 1:
                motor.alert_medium(risk_score)
            else:
                motor.alert_high(risk_score)
            motor_cmd = motor.get_latest()

            # ── SystemState ──
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
                vision_valid=vision_data.valid,
                vision_object_count=len(vision_data.objects),
                vision_person_count=vision_data.person_count,
                vision_vehicle_count=vision_data.vehicle_count,
                vision_max_risk=vision_data.max_visual_risk,
                vision_drivable_ratio=vision_data.drivable_area_ratio,
                risk_obs=risk_items["R_obs"],
                risk_dist=risk_items["R_dist"],
                risk_pose=risk_items["R_pose"],
                risk_speed=risk_items["R_speed"],
                risk_total=risk_score,
                risk_level=level,
                motor_pattern=motor_cmd.pattern if motor_cmd else "silent",
                motor_duration_ms=motor_cmd.duration_ms if motor_cmd else 0,
            )

            viewer.display(state)
            logger.write(state)

            elapsed = time.time() - frame_start
            time.sleep(max(0, loop_interval - elapsed))

        success = True

    except KeyboardInterrupt:
        print("\n[主循环] 用户手动终止")
        success = True

    except Exception as e:
        print(f"\n[主循环] 异常: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "=" * 65)
        print("  正在关闭所有模块...")
        print("=" * 65)

        gps_reader.stop()
        imu_reader.stop()
        radar_reader.stop()
        motor.stop()
        vision.stop()
        logger.close()

        print(f"\n[完成] 带视觉闭环测试结束")
        print(f"        日志: {os.path.abspath(logger.file_path)}")
        print(f"        视觉: vision_enabled=True, 视觉有效帧={sum(1 for v in [success] if v)}")


if __name__ == "__main__":
    main()
