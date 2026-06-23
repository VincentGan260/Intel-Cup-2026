"""系统整合主入口。

完整闭环调度：
  VisionAdapter + GPSReader + IMUReader + RadarReader
    → Synchronizer
    → RiskModel + RiskLevel
    → MotorController
    → ConsoleViewer + CsvLogger

支持分级联调：马达可独立于传感器模式，方便真实传感器 + 马达 mock 安全验证。

用法：
  python main_integrated.py --mode mock   --vision false --loops 20
  python main_integrated.py --mode real   --vision false --motor mock --profile dk2500 --loops 50
  python main_integrated.py --mode real   --vision false --motor real --profile dk2500 --loops 50
  python main_integrated.py --mode real   --vision true  --motor mock --profile dk2500 --loops 50
  python main_integrated.py --mode real   --vision true  --motor real
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.data_types import SystemState, now
from src.fusion.risk_level import RiskLevelClassifier
from src.fusion.risk_model import RiskModel
from src.fusion.synchronizer import Synchronizer
from src.fusion.vision_radar_fusion import VisionRadarFusion
from src.debug.console_viewer import ConsoleViewer
from src.utils.logger import CsvLogger


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_sensor_ports(platform: str) -> dict:
    cfg = _load_config("configs/sensor_ports.yaml")
    ports = cfg.get(platform, {})
    camera = cfg.get("camera", {})
    if camera:
        ports["camera"] = camera
    return ports


# ============================================================
#  模块工厂（各模块独立构建，支持分级模式）
# ============================================================


def _build_gps_reader(sensor_mode: str, ports: dict):
    """构建 GPS 读取器。"""
    from src.sensors.gps_reader import GPSReader

    if sensor_mode == "mock":
        return GPSReader(mode="mock")

    cfg = ports.get("gps", {})
    return GPSReader(mode="real", config=cfg)


def _build_imu_reader(sensor_mode: str, ports: dict):
    """构建 IMU 读取器。"""
    from src.sensors.imu_reader import IMUReader

    if sensor_mode == "mock":
        return IMUReader(mode="mock")

    cfg = ports.get("imu", {})
    return IMUReader(mode="real", config=cfg)


def _build_radar_reader(sensor_mode: str, ports: dict):
    """构建雷达读取器。"""
    from src.sensors.radar_reader import RadarReader

    if sensor_mode == "mock":
        return RadarReader(mode="mock")

    cfg = ports.get("radar", {})
    return RadarReader(mode="real", config=cfg)


def _build_motor(motor_mode: str, ports: dict):
    """构建马达控制器。

    motor_mode=mock → MotorController(mode="mock")，**绝对不会调用 I2C**。
    motor_mode=real → MotorController(mode="real")，会初始化 I2C 并可能触发震动。
    """
    from src.actuator.motor_controller import MotorController

    if motor_mode == "mock":
        return MotorController(mode="mock")

    cfg = ports.get("motor", {})
    return MotorController(
        mode="real",
        i2c_bus=cfg.get("i2c_bus", 1),
        i2c_addr=int(cfg.get("driver_address", "0x5A"), 16),
    )


def _build_vision(vision_enabled: bool, use_camera: bool):
    """构建视觉适配器（不启动）。返回 (adapter_or_None, status_str)。"""
    if not vision_enabled:
        return None, "DISABLED"

    try:
        from src.fusion.vision_adapter import VisionAdapter

        adapter = VisionAdapter(
            pipeline_config_path="configs/vision/vision_pipeline.yaml",
            vision_enabled=True,
            use_camera=use_camera,
            camera_id=0,
        )
        return adapter, "INIT_OK"
    except ImportError as e:
        return None, f"DEGRADED (import: {e})"


# ============================================================
#  启动辅助 — 每个模块单独启动并记录详细状态
# ============================================================


def _try_start(mod, label: str, detail: str = "") -> tuple[bool, str]:
    """尝试启动一个模块。返回 (成功?, 状态字符串)。"""
    try:
        mod.start()
        return True, "OK"
    except Exception as e:
        info = f"FAILED ({e})" + (f" | {detail}" if detail else "")
        return False, info


def _print_startup_summary(
    sensor_mode: str,
    motor_mode: str,
    profile: str,
    loop_hz: int,
    log_file: str,
    loop_hint: str,
    statuses: dict,
    port_infos: dict,
) -> None:
    """打印带有详细模块状态的启动摘要。"""
    mode_map = {"gps": sensor_mode, "imu": sensor_mode, "radar": sensor_mode, "motor": motor_mode}

    print("=" * 65)
    print("  [启动摘要]")
    print(f"    传感器模式:     {sensor_mode}")
    print(f"    马达模式:       {motor_mode}")
    print(f"    端口配置:       {profile}")
    print(f"    循环频率:       {loop_hz} Hz")
    print(f"    运行轮数:       {loop_hint}")
    print(f"    日志文件:       {log_file}")
    print("-" * 65)

    for label in ("GPS", "IMU", "Radar", "Motor", "Vision"):
        st = statuses.get(label, "N/A")
        key = label.lower()
        actual_mode = mode_map.get(key, "N/A").upper()
        port_str = ""
        if key in port_infos:
            port_str = ", ".join(f"{k}={v}" for k, v in port_infos[key].items())
        if port_str:
            print(f"    {label + ':':8s}  {actual_mode:6s}  {st:20s}  ({port_str})")
        else:
            print(f"    {label + ':':8s}  {actual_mode:6s}  {st}")

    print("=" * 65)


def _make_state(
    ts: float, gps, imu, radar, vision_data, vision_enabled: bool,
    risk_items: dict, level: int, motor_cmd,
) -> SystemState:
    return SystemState(
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
        radar_target_count=len(radar.targets),
        radar_nearest_m=radar.nearest_distance_m,
        radar_min_ttc=radar.min_ttc,
        vision_valid=vision_data.valid if vision_data else False,
        vision_object_count=len(vision_data.objects) if vision_data else 0,
        vision_person_count=vision_data.person_count if vision_data else 0,
        vision_vehicle_count=vision_data.vehicle_count if vision_data else 0,
        vision_max_risk=vision_data.max_visual_risk if vision_data else 0.0,
        vision_drivable_ratio=vision_data.drivable_area_ratio if vision_data else 0.0,
        risk_obs=risk_items["R_obs"],
        risk_dist=risk_items["R_dist"],
        risk_pose=risk_items["R_pose"],
        risk_speed=risk_items["R_speed"],
        risk_total=risk_items["risk_score"],
        risk_level=level,
        motor_pattern=motor_cmd.pattern if motor_cmd else "silent",
        motor_duration_ms=motor_cmd.duration_ms if motor_cmd else 0,
    )


# ============================================================
#  主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="系统整合主入口 — 支持分级联调")
    parser.add_argument(
        "--mode", type=str, default="mock", choices=["mock", "real"],
        help="GPS / IMU / Radar 传感器模式（mock=模拟数据, real=真实硬件）",
    )
    parser.add_argument(
        "--vision", type=str, default="false", choices=["true", "false"],
        help="是否启用视觉模块",
    )
    parser.add_argument(
        "--motor", type=str, default=None,
        choices=["mock", "real"],
        help="马达模式（默认与 --mode 一致；设为 mock 可安全验证传感器而不触发震动）",
    )
    parser.add_argument(
        "--loops", type=int, default=None,
        help="运行 N 轮后自动退出；不传则持续运行",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        choices=["windows", "dk2500"],
        help="端口配置 profile（默认从 configs/config.yaml 读取）",
    )
    parser.add_argument(
        "--camera", action="store_true",
        help="视觉使用摄像头输入（否则使用默认图片）",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="系统配置文件路径（默认 configs/config.yaml）",
    )
    parser.add_argument(
        "--confirm-motor-real", action="store_true",
        help="确认马达使用 REAL 模式。不加此参数时 --motor real 将被拒绝，防止误触发硬件震动。",
    )
    args = parser.parse_args()

    vision_enabled = args.vision.lower() == "true"
    sensor_mode = args.mode
    motor_mode = args.motor if args.motor is not None else sensor_mode  # 默认与 --mode 一致

    # ── Motor real 二次确认 ──
    if motor_mode == "real" and not args.confirm_motor_real:
        print("=" * 65)
        print("  [安全保护] --motor real 需要二次确认")
        print()
        print("  您指定了 --motor real，这将导致 DRV2605 震动马达真实触发。")
        print("  为防止上车调试时误触发，请附加 --confirm-motor-real 参数：")
        print()
        print("    python main_integrated.py --mode real --motor real --confirm-motor-real")
        print()
        print("  如果不需要马达真实震动，建议使用：")
        print()
        print("    python main_integrated.py --mode real --motor mock")
        print()
        print("  程序已安全退出。")
        print("=" * 65)
        sys.exit(0)

    # ── 加载系统配置 ──
    sys_cfg = _load_config(args.config)
    system = sys_cfg.get("system", {})
    main_loop_hz = system.get("main_loop_hz", 10)
    profile = args.profile if args.profile is not None else system.get("platform", "windows")
    loop_interval = 1.0 / main_loop_hz
    loop_hint = str(args.loops) if args.loops else "无限"

    # ── 确定日志文件名 ──
    if sensor_mode == "mock":
        log_file = "logs/main_integrated_mock.csv"
    elif motor_mode == "mock" and not vision_enabled:
        log_file = "logs/main_integrated_real_sensors_motor_mock.csv"
    elif motor_mode == "real" and not vision_enabled:
        log_file = "logs/main_integrated_real_sensors_motor_real.csv"
    elif motor_mode == "mock" and vision_enabled:
        log_file = "logs/main_integrated_real_sensors_motor_mock_vision.csv"
    elif motor_mode == "real" and vision_enabled:
        log_file = "logs/main_integrated_all_real.csv"
    else:
        log_file = "logs/main_integrated.csv"

    # ── 构建各模块（不启动） ──
    ports = _load_sensor_ports(profile) if sensor_mode == "real" or motor_mode == "real" else {}

    gps_reader = _build_gps_reader(sensor_mode, ports)
    imu_reader = _build_imu_reader(sensor_mode, ports)
    radar_reader = _build_radar_reader(sensor_mode, ports)
    motor = _build_motor(motor_mode, ports)

    vision_adapter, vision_init_status = _build_vision(vision_enabled, args.camera)

    # ── 启动模块，收集详细状态 ──
    module_list = [gps_reader, imu_reader, radar_reader, motor]
    if vision_adapter:
        module_list.append(vision_adapter)

    statuses: dict[str, str] = {}
    port_infos: dict[str, dict] = {}

    # GPS
    gps_ports = ports.get("gps", {})
    port_infos["gps"] = {"port": gps_ports.get("port", "N/A"), "baud": str(gps_ports.get("baudrate", 9600))}
    ok, st = _try_start(gps_reader, "GPS", f"port={gps_ports.get('port', 'N/A')}")
    statuses["GPS"] = st

    # IMU
    imu_ports = ports.get("imu", {})
    port_infos["imu"] = {"port": imu_ports.get("port", "N/A"), "baud": str(imu_ports.get("baudrate", 115200))}
    ok, st = _try_start(imu_reader, "IMU", f"port={imu_ports.get('port', 'N/A')}")
    statuses["IMU"] = st

    # Radar
    radar_ports = ports.get("radar", {})
    port_infos["radar"] = {"port": radar_ports.get("port", "N/A"), "baud": str(radar_ports.get("baudrate", 115200))}
    ok, st = _try_start(radar_reader, "Radar", f"port={radar_ports.get('port', 'N/A')}")
    statuses["Radar"] = st

    # Motor
    if motor_mode == "real":
        print("")
        print("  [WARNING] Motor is running in REAL mode. The vibration motor may be triggered.")
        print("")
    motor_ports = ports.get("motor", {})
    port_infos["motor"] = {
        "bus": str(motor_ports.get("i2c_bus", "N/A")),
        "addr": motor_ports.get("driver_address", "N/A"),
    }
    ok, st = _try_start(motor, "Motor", f"i2c_bus={motor_ports.get('i2c_bus', 'N/A')}")
    statuses["Motor"] = st

    # Vision
    statuses["Vision"] = _try_start_vision(vision_adapter)
    vision_is_active = vision_adapter is not None and vision_adapter.vision_enabled

    # ── 打印启动摘要 ──
    _print_startup_summary(
        sensor_mode, motor_mode, profile, main_loop_hz,
        log_file, loop_hint, statuses, port_infos,
    )

    if sensor_mode == "real":
        for label in ("GPS", "IMU", "Radar", "Motor"):
            st = statuses.get(label, "")
            if "FAILED" in st:
                print(f"  [注意] {label} 启动失败，系统将继续运行（该模块数据不可用）")

    # ── 依赖模块 ──
    sync = Synchronizer(vision_enabled=vision_is_active)
    risk_model = RiskModel()
    vr_fusion = VisionRadarFusion()   # 视觉-雷达目标级融合（持续性跨帧，故在循环外持有）
    classifier = RiskLevelClassifier()
    viewer = ConsoleViewer()
    logger = CsvLogger(log_file)

    # ── 主循环 ──
    try:
        loop_count = 0
        vision_frame = None

        if vision_adapter and not args.camera:
            try:
                from src.vision.common.preprocess import load_image_bgr_from_source

                vision_frame, _ = load_image_bgr_from_source(
                    "https://ultralytics.com/images/bus.jpg",
                    PROJECT_ROOT,
                )
                print(f"  已加载默认视觉图片 ({vision_frame.shape[1]}x{vision_frame.shape[0]})")
            except Exception as e:
                print(f"  无法加载默认图片: {e}")

        while True:
            loop_count += 1
            frame_start = time.time()

            # 1. 采集传感器数据
            gps = gps_reader.read_once()
            imu = imu_reader.read_once()
            radar = radar_reader.read_once()

            # 2. 视觉（可选）
            vision_data = None
            if vision_adapter:
                vision_data = vision_adapter.process(vision_frame) if not args.camera else vision_adapter.process()

            # 3. 同步
            sync.update_gps(gps)
            sync.update_imu(imu)
            sync.update_radar(radar)
            if vision_data is not None:
                sync.update_vision(vision_data)
            fusion = sync.build_frame()

            # 3.5 视觉-雷达目标级融合：用融合后的 R_obs（雷达真实距离/TTC + 仅雷达未知障碍）
            #     覆盖 R_obs。雷达有效即运行——视觉失效/未启用时以"纯雷达"兜底（B 方案），
            #     摄像头/模型挂掉雷达仍报警。异常时安全退回原值。
            if radar.valid:
                try:
                    vres = vision_adapter.get_latest_vision_result() if (
                        vision_is_active and vision_adapter) else None
                    fused_out = vr_fusion.fuse_vision_result(vres, radar)
                    if fused_out is not None:
                        # R_obs 现为"视觉+雷达"统一障碍物风险；置 enabled/valid 让 risk_model 采用
                        fusion.vision.max_visual_risk = fused_out.max_risk
                        fusion.vision_enabled = True
                        fusion.vision.valid = True
                        if fused_out.n_radar_only > 0:
                            print(f"  [融合] 雷达兜底未知障碍 ×{fused_out.n_radar_only}"
                                  f"（视觉未识别/失效，已计入风险）")
                except Exception as e:
                    print(f"  [融合] 跳过(异常)，退回原 R_obs: {e}")

            # 4. 风险融合
            risk_items, weights = risk_model.compute(fusion)
            level, label = classifier.classify(risk_items["risk_score"])
            risk_score = risk_items["risk_score"]

            # 5. 马达控制
            if level == 0:
                motor.alert_low()
            elif level == 1:
                motor.alert_medium(risk_score)
            else:
                motor.alert_high(risk_score)
            motor_cmd = motor.get_latest()

            # 6. 日志 + 显示
            ts = now()
            state = _make_state(
                ts, gps, imu, radar, vision_data, vision_is_active,
                risk_items, level, motor_cmd,
            )
            viewer.display(state)
            logger.write(state)

            # 7. 退出条件
            if args.loops and loop_count >= args.loops:
                print(f"\n[主循环] 已完成 {loop_count} 轮，自动退出")
                break

            elapsed = time.time() - frame_start
            time.sleep(max(0, loop_interval - elapsed))

    except KeyboardInterrupt:
        print("\n[主循环] 用户手动终止")

    except Exception as e:
        print(f"\n[主循环] 异常: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "=" * 65)
        print("  正在关闭所有模块...")
        print("=" * 65)

        for mod in reversed(module_list):
            try:
                mod.stop()
            except Exception as e:
                print(f"  [关闭异常] {mod.__class__.__name__}: {e}")

        logger.close()
        print(f"  日志: {os.path.abspath(logger.file_path)}")


def _try_start_vision(adapter) -> str:
    """启动视觉适配器，返回状态字符串。"""
    if adapter is None:
        return "DISABLED"
    try:
        adapter.start()
        return "OK" if adapter.vision_enabled else "DEGRADED (模型加载失败)"
    except Exception as e:
        return f"DEGRADED ({e})"


if __name__ == "__main__":
    main()
