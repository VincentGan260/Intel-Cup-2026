"""Web Dashboard 启动入口。

零侵入：不修改任何已有业务代码。

状态架构：
  run_dashboard.py（总入口）
    ├── CameraFrameProducer         — 摄像头帧读取 (含 get_bgr_frame)
    ├── DashboardStateStore          — 线程安全状态池
    ├── mock 模式: build_mock_state()  — sin 波风险（纯 mock）
    ├── hybrid 模式: build_hybrid_state() — 真实 RiskModel + mock 传感器
    │     ├── RiskModel              — 真实风险模型
    │     ├── RiskLevelClassifier    — 真实等级分类
    │     ├── Synchronizer           — 传感器同步
    │     └── VisionAdapter (可选)   — 真实视觉推理
    ├── 后台线程 dashboard_state_loop — 按 state_hz 更新状态池
    └── FastAPI 服务                  — 提供 / /api/state /video_feed /api/health

用法：
  python run_dashboard.py
  python run_dashboard.py --dashboard-mode mock --camera-id 0
  python run_dashboard.py --dashboard-mode hybrid
  python run_dashboard.py --dashboard-mode hybrid --enable-vision
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import time
import threading
from pathlib import Path

# 确保项目根目录在 sys.path 中（顶层脚本自身就在根目录）
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def attach_runtime_info(
    state: dict,
    *,
    start_time: float,
    state_hz: int,
    camera_available: bool,
    enable_vision: bool,
    state_loop_ms: float = 0.0,
    actual_sample_hz: float = 0.0,
    vision_p50_ms: float = 0.0,
    vision_p95_ms: float = 0.0,
) -> dict:
    """在状态 dict 上附加运行时信息字段。

    由 dashboard_state_loop 在每次更新状态后调用。
    不接触核心算法，只修改 Dashboard 状态 dict。
    """
    from datetime import datetime

    now_ts = time.time()
    uptime = now_ts - start_time
    vision_status = "off"

    sensors = state.get("sensors", {})
    sensor_vision = sensors.get("vision", "off")
    if sensor_vision == "real":
        vision_status = "real"
    elif sensor_vision == "invalid":
        vision_status = "invalid"
    elif state.get("mode") == "mock":
        vision_status = "mock"
    else:
        vision_status = sensor_vision if isinstance(sensor_vision, str) else "off"

    vd = state.get("vision_details", {})
    object_count = vd.get("object_count", 0)

    # 从 state["performance"] 读取 vision_infer_ms
    perf = state.get("performance", {})
    vision_infer_ms = round(float(perf.get("vision_infer_ms", 0.0)), 2)

    # 近似 FPS：1000 / state_loop_ms，最小显示 0.01
    approx_state_fps = round(1000.0 / state_loop_ms, 2) if state_loop_ms > 0 else 0.0

    state["runtime"] = {
        "backend_uptime_sec": round(uptime, 2),
        "state_hz": state_hz,
        "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_available": camera_available,
        "vision_enabled": enable_vision,
        "vision_status": vision_status,
        "object_count": object_count,
        "state_loop_ms": round(state_loop_ms, 2),
        "vision_infer_ms": vision_infer_ms,
        "approx_state_fps": approx_state_fps,
        "actual_sample_hz": round(actual_sample_hz, 2),
        "vision_p50_ms": round(vision_p50_ms, 2),
        "vision_p95_ms": round(vision_p95_ms, 2),
    }
    return state


def _cue_vision_result_cache(vision_adapter) -> None:
    """将 vision_adapter 内最新的 VisionResult 写入 server.py 缓存。

    由 dashboard_state_loop 在每次状态更新后调用。
    不重复推理，仅读取 vision_adapter.get_latest_vision_result()。
    """
    try:
        from src.dashboard.server import update_vision_result_cache

        vr = vision_adapter.get_latest_vision_result() if vision_adapter else None
        update_vision_result_cache(vr)
    except Exception:
        pass


def dashboard_state_loop(
    state_store,
    camera_producer,
    stop_event: threading.Event,
    *,
    mode: str = "mock",
    risk_model=None,
    classifier=None,
    synchronizer=None,
    vision_adapter=None,
    enable_vision: bool = False,
    imu_reader=None,
    enable_imu: bool = False,
    radar_reader=None,
    gps_reader=None,
    recorder=None,
    cloud_sync=None,
    sync_thresholds=None,
    risk_rule=None,
    motor=None,
    target_stale_ms: float = 500.0,
    radar_communication_watchdog_ms: float = 2000.0,
    interval: float = 0.2,
    start_time: float = 0.0,
    state_hz: int = 5,
) -> None:
    """后台状态更新线程。

    mock 模式：调用 build_mock_state()
    hybrid 模式：调用 build_hybrid_state()（真实 RiskModel + mock 传感器）

    Args:
        state_store: DashboardStateStore 实例
        camera_producer: CameraFrameProducer 实例
        stop_event: 用于优雅关闭的 Event
        mode: "mock" 或 "hybrid"
        risk_model: RiskModel 实例（hybrid 模式必需）
        classifier: RiskLevelClassifier 实例（hybrid 模式必需）
        synchronizer: Synchronizer 实例（hybrid 模式必需）
        vision_adapter: VisionAdapter 或 None
        enable_vision: 是否启用视觉
        interval: 循环间隔（秒）
    """
    from src.dashboard.mock_state import build_mock_state

    if mode == "hybrid":
        from src.dashboard.hybrid_state import build_hybrid_state
    elif mode == "real":
        from src.dashboard.real_sensor_state import build_real_sensor_state

    print(f"[StateLoop] 后台状态更新线程已启动 (mode={mode}, interval={interval:.2f}s)")
    sample_times = []
    vision_latencies = []
    while not stop_event.is_set():
        try:
            loop_start = time.monotonic()
            if mode == "mock":
                state = build_mock_state(camera_available=camera_producer.is_available)
            elif mode == "hybrid":
                bgr_frame = camera_producer.get_bgr_frame() if enable_vision else None
                state = build_hybrid_state(
                    risk_model=risk_model,
                    classifier=classifier,
                    synchronizer=synchronizer,
                    vision_adapter=vision_adapter,
                    bgr_frame=bgr_frame,
                    camera_available=camera_producer.is_available,
                    imu_reader=imu_reader,
                    enable_imu=enable_imu,
                )

                # 将最新 VisionResult 写入缓存（视频流每帧读取绘制，不重复推理）
                if bgr_frame is not None:
                    _cue_vision_result_cache(vision_adapter)
            elif mode == "real":
                if enable_vision or recorder is not None:
                    bgr_frame, frame_capture_ns, camera_frame_id = camera_producer.get_bgr_frame_with_timestamp()
                else:
                    bgr_frame, frame_capture_ns, camera_frame_id = None, 0, -1
                state = build_real_sensor_state(
                    camera_available=camera_producer.is_available,
                    radar_reader=radar_reader,
                    gps_reader=gps_reader,
                    frame=bgr_frame,
                    frame_capture_monotonic_ns=frame_capture_ns,
                    camera_frame_id=camera_frame_id,
                    vision_adapter=vision_adapter,
                    fusion_engine=synchronizer,
                    recorder=recorder,
                    sync_thresholds=sync_thresholds,
                    risk_rule=risk_rule,
                    risk_model=risk_model,
                    classifier=classifier,
                    motor=motor,
                    target_stale_ms=target_stale_ms,
                    radar_communication_watchdog_ms=radar_communication_watchdog_ms,
                )
                if vision_adapter is not None:
                    _cue_vision_result_cache(vision_adapter)
            else:
                state = build_mock_state(camera_available=camera_producer.is_available)

            # 统计循环耗时（毫秒）
            state_loop_ms = round((time.monotonic() - loop_start) * 1000.0, 2)
            sample_times.append(loop_start)
            sample_times = sample_times[-300:]
            infer_ms = float(state.get("performance", {}).get("vision_infer_ms", 0.0))
            if infer_ms > 0:
                vision_latencies.append(infer_ms)
                vision_latencies = vision_latencies[-300:]
            actual_hz = ((len(sample_times) - 1) / (sample_times[-1] - sample_times[0])
                         if len(sample_times) > 1 and sample_times[-1] > sample_times[0] else 0.0)
            p50 = statistics.median(vision_latencies) if vision_latencies else 0.0
            p95 = (sorted(vision_latencies)[min(len(vision_latencies) - 1,
                   int(0.95 * len(vision_latencies)))]) if vision_latencies else 0.0

            # 附加运行时信息
            attach_runtime_info(
                state,
                start_time=start_time,
                state_hz=state_hz,
                camera_available=camera_producer.is_available,
                enable_vision=enable_vision,
                state_loop_ms=state_loop_ms,
                actual_sample_hz=actual_hz,
                vision_p50_ms=p50,
                vision_p95_ms=p95,
            )
            state_store.set_state(state)
            if cloud_sync is not None:
                cloud_sync.publish_state(state)
        except Exception as e:
            print(f"[StateLoop] 状态更新异常: {e}")
        remaining = interval - (time.monotonic() - loop_start)
        if remaining > 0:
            stop_event.wait(remaining)
    print("[StateLoop] 后台状态更新线程已退出")


def main() -> None:
    parser = argparse.ArgumentParser(description="骑手前向安全预警 Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头设备编号")
    parser.add_argument("--reload", action="store_true", help="开启 uvicorn 热重载（开发模式）")
    parser.add_argument(
        "--dashboard-mode", type=str, default="real", choices=["real", "mock", "hybrid"],
        help="Dashboard 运行模式: mock（纯模拟）/ hybrid（真实 RiskModel + mock 传感器）",
    )
    parser.add_argument(
        "--profile", type=str, default="dk2500", choices=["windows", "dk2500"],
        help="sensor_ports.yaml profile used in real mode",
    )
    parser.add_argument("--record", action="store_true", help="record aligned real sensor and vision samples")
    parser.add_argument("--scene", type=str, default="dashboard_test", help="recording scene label")
    parser.add_argument("--record-output", type=str, default="data/recordings")
    parser.add_argument("--operator", type=str, default="unknown")
    parser.add_argument("--route", type=str, default="unknown")
    parser.add_argument("--weather", type=str, default="unknown")
    parser.add_argument("--road-condition", type=str, default="unknown")
    parser.add_argument("--group-id", type=str, default="")
    parser.add_argument("--risk-label", choices=["low", "mid", "high"], default=None,
                        help="整段受控场景的初始风险标签，仍需人工复核")
    parser.add_argument("--skip-gps-fix", action="store_true",
                        help="兼容旧命令；GPS已降级为可选传感器，默认不等待定位")
    parser.add_argument("--wait-gps", action="store_true",
                        help="显式要求录制前等待GPS定位；GPS失效时不要使用")
    parser.add_argument("--gps-timeout", type=int, default=90)
    parser.add_argument("--cloud-enable", action="store_true",
                        help="upload compact ride data and raw minute videos to the cloud")
    parser.add_argument("--cloud-url", default="http://124.70.108.34",
                        help="cloud API base URL")
    parser.add_argument("--device-id", default="bike-001",
                        help="cloud device identifier")
    parser.add_argument("--cloud-state-hz", type=float, default=1.0,
                        help="ride sample upload frequency")
    parser.add_argument("--cloud-video-fps", type=float, default=10.0,
                        help="raw cloud video recording frame rate")
    parser.add_argument("--cloud-video-seconds", type=float, default=60.0,
                        help="cloud video segment duration")
    parser.add_argument("--cloud-spool", default="data/cloud_spool",
                        help="local directory for videos awaiting upload")
    parser.add_argument(
        "--enable-vision", action="store_true",
        help="hybrid 模式下启用视觉推理（需要 openvino 环境）",
    )
    parser.add_argument(
        "--enable-imu", action="store_true",
        help="hybrid 模式下启用真实 IMU（WT61C 串口传感器）",
    )
    parser.add_argument(
        "--risk-config", type=str, default="configs/risk_params.yaml",
        help="风险参数配置文件路径",
    )
    parser.add_argument(
        "--vision-config", type=str, default="configs/vision/vision_pipeline.yaml",
        help="视觉管线配置文件路径",
    )
    parser.add_argument(
        "--state-hz", type=int, default=5,
        help="状态更新频率（Hz），默认 5",
    )
    parser.add_argument("--enable-risk-rule", action="store_true",
                        help="real模式启用比赛版雷达TTC紧迫性规则")
    parser.add_argument("--motor-mode", choices=["off", "mock", "real"], default="off",
                        help="比赛版风险规则的马达输出模式")
    parser.add_argument("--confirm-motor-real", action="store_true",
                        help="确认真实驱动DRV2605；--motor-mode real时必须提供")
    parser.add_argument("--configured-warning-range-m", type=float, default=None,
                        help="与LD2451 APP最远检测距离一致的比赛工作范围")
    parser.add_argument("--radar-to-motor-p95-ms", type=float, default=0.469,
                        help="DK-2500实测雷达到DRV2605 GO的P95延迟，默认0.469 ms")
    parser.add_argument("--target-stale-ms", type=float, default=500.0,
                        help="有目标100ms周期的5倍工程容错值")
    parser.add_argument("--radar-communication-watchdog-ms", type=float, default=2000.0,
                        help="无目标约1s周期的2倍工程通信看门狗")
    args = parser.parse_args()

    if args.enable_risk_rule:
        required_values = {
            "--configured-warning-range-m": args.configured_warning_range_m,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            parser.error("--enable-risk-rule requires configuration values: " + ", ".join(missing))
        if args.configured_warning_range_m <= 0:
            parser.error("--configured-warning-range-m must be positive")
        if args.radar_to_motor_p95_ms < 0:
            parser.error("radar-to-motor P95 must be non-negative")
        if args.target_stale_ms <= 0 or args.radar_communication_watchdog_ms <= 0:
            parser.error("radar watchdog values must be positive")
    if args.motor_mode == "real" and not args.confirm_motor_real:
        parser.error("--motor-mode real requires --confirm-motor-real")
    if args.motor_mode != "off" and not args.enable_risk_rule:
        parser.error("--motor-mode requires --enable-risk-rule")

    import yaml
    recording_cfg = yaml.safe_load(
        (_project_root / "configs" / "dashboard_recording.yaml").read_text(encoding="utf-8")
    )
    sync_thresholds = recording_cfg["sync"]

    # ── 1. 创建摄像头 ──
    from src.dashboard.frame_producer import CameraFrameProducer

    camera = CameraFrameProducer(camera_id=args.camera_id)

    # ── 2. 创建状态池 ──
    from src.dashboard.state_store import DashboardStateStore

    state_store = DashboardStateStore()

    # 写入一次初始状态
    from src.dashboard.mock_state import build_mock_state

    state_store.set_state(build_mock_state(camera_available=camera.is_available))

    # ── 3. hybrid 模式初始化 ──
    risk_model = None
    classifier = None
    synchronizer = None
    vision_adapter = None
    imu_reader = None
    vision_init_ok = False
    imu_init_ok = False
    radar_reader = None
    gps_reader = None
    recorder = None
    cloud_sync = None
    competition_risk_rule = None
    motor_controller = None

    if args.dashboard_mode == "real":
        from src.sensors.gps_reader import GPSReader
        from src.sensors.radar_reader import RadarReader

        ports_cfg = yaml.safe_load(
            (_project_root / "configs" / "sensor_ports.yaml").read_text(encoding="utf-8")
        )
        profile_cfg = ports_cfg[args.profile]
        radar_reader = RadarReader(mode="real", config=profile_cfg["radar"])
        gps_reader = GPSReader(mode="real", config=profile_cfg["gps"])
        radar_reader.start()
        gps_reader.start()
        print(f"[RealSensors] profile={args.profile}; IMU and models disabled")

        from src.fusion.risk_model import RiskModel
        from src.fusion.risk_level import RiskLevelClassifier
        risk_model = RiskModel(config_path=args.risk_config)
        classifier = RiskLevelClassifier(config_path=args.risk_config)
        print("[RealSensors] adaptive RiskModel enabled (IMU invalid and automatically down-weighted)")

        if args.enable_risk_rule:
            from src.fusion.physical_risk_rule import PhysicalRiskRule

            competition_risk_rule = PhysicalRiskRule(
                body_width_m=0.66,
                point_gate_lateral_margin_m=0.025,
                mounting_offset_m=-0.055,
                mounting_uncertainty_m=0.005,
                configured_warning_range_m=args.configured_warning_range_m,
                radar_to_motor_p95_s=args.radar_to_motor_p95_ms / 1000.0,
                urgent_reference_s=2.5,
                max_abs_angle_deg=15.0,
            )
            print("[CompetitionRisk] radar TTC urgency rule enabled")
            print(f"[CompetitionRisk] configured_range={args.configured_warning_range_m:.2f}m, "
                  f"urgent_ttc={competition_risk_rule.urgent_ttc_s:.3f}s")

            if args.motor_mode != "off":
                from src.actuator.timed_motor_controller import TimedMotorController

                motor_cfg = profile_cfg["motor"]
                motor_addr = motor_cfg.get("driver_address", "0x5A")
                motor_addr = int(motor_addr, 0) if isinstance(motor_addr, str) else int(motor_addr)
                motor_controller = TimedMotorController(
                    mode=args.motor_mode,
                    i2c_bus=int(motor_cfg["i2c_bus"]),
                    i2c_addr=motor_addr,
                )
                motor_controller.start()

        if args.enable_vision:
            from src.fusion.vision_adapter import VisionAdapter
            from src.fusion.vision_radar_fusion import VisionRadarFusion
            vision_adapter = VisionAdapter(
                pipeline_config_path=args.vision_config, vision_enabled=True, use_camera=False)
            vision_adapter.start()
            vision_init_ok = bool(vision_adapter.vision_enabled)
            if vision_init_ok:
                synchronizer = VisionRadarFusion()
            else:
                vision_adapter = None
        if args.record:
            if not vision_init_ok:
                raise RuntimeError(
                    "--record requires a working vision pipeline; fix model/OpenVINO setup "
                    "and start with --enable-vision"
                )
            from src.dashboard.dashboard_recorder import DashboardRecorder
            if args.wait_gps and not args.skip_gps_fix:
                print(f"[GPS] 等待有效定位（最多 {args.gps_timeout}s）...")
                deadline = time.monotonic() + args.gps_timeout
                gps_fixed = False
                while time.monotonic() < deadline:
                    if gps_reader.read_once().valid:
                        gps_fixed = True
                        break
                if not gps_fixed:
                    raise RuntimeError(
                        "GPS未在超时内获得有效定位；当前GPS失效时请不要使用 --wait-gps")
            record_root = _project_root / args.record_output
            record_root.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(record_root).free / (1024 ** 3)
            min_free_gb = float(recording_cfg.get("storage", {}).get("min_free_gb", 2.0))
            if free_gb < min_free_gb:
                raise RuntimeError(f"录制目录可用空间仅 {free_gb:.2f} GiB，低于 {min_free_gb:.2f} GiB")
            detection_cfg = yaml.safe_load(
                (_project_root / "configs" / "vision" / "detection.yaml").read_text(encoding="utf-8")
            )
            session_fields = {"operator": args.operator, "route": args.route,
                              "weather": args.weather, "road_condition": args.road_condition,
                              "group_id": args.group_id or args.scene}
            recorder = DashboardRecorder(
                record_root, args.scene, args.profile,
                recording_config=recording_cfg, session_fields=session_fields,
                model_path=detection_cfg["detector"]["model_path"],
                vision_config=args.vision_config,
                vision_runtime=vision_adapter.get_runtime_info(),
                risk_label=args.risk_label)
            print(f"[Recorder] session={recorder.session_dir}")

    if args.dashboard_mode == "hybrid":
        print("-" * 55)
        print("  [Hybrid 模式初始化]")
        print(f"    风险配置: {args.risk_config}")
        print(f"    视觉:     {'启用' if args.enable_vision else '关闭'}")
        print(f"    IMU:      {'启用' if args.enable_imu else '关闭'}")

        # ── RiskModel ──
        try:
            from src.fusion.risk_model import RiskModel

            risk_model = RiskModel(config_path=args.risk_config)
            print("    RiskModel:         OK")
        except Exception as e:
            print(f"    RiskModel:         FAILED ({e})")
            print("    无法初始化 RiskModel，降级到 mock 模式")
            args.dashboard_mode = "mock"

        # ── RiskLevelClassifier ──
        if args.dashboard_mode == "hybrid":
            try:
                from src.fusion.risk_level import RiskLevelClassifier

                classifier = RiskLevelClassifier(config_path=args.risk_config)
                print("    RiskLevelClassifier: OK")
            except Exception as e:
                print(f"    RiskLevelClassifier: FAILED ({e})")
                args.dashboard_mode = "mock"

        # ── Synchronizer ──
        if args.dashboard_mode == "hybrid":
            try:
                from src.fusion.synchronizer import Synchronizer

                synchronizer = Synchronizer(vision_enabled=args.enable_vision)
                print("    Synchronizer:       OK")
            except Exception as e:
                print(f"    Synchronizer:       FAILED ({e})")
                args.dashboard_mode = "mock"

        # ── VisionAdapter（可选） ──
        if args.dashboard_mode == "hybrid" and args.enable_vision:
            try:
                from src.fusion.vision_adapter import VisionAdapter

                vision_adapter = VisionAdapter(
                    pipeline_config_path=args.vision_config,
                    vision_enabled=True,
                    use_camera=False,  # 由 Dashboard 摄像头统一管理
                    camera_id=args.camera_id,
                )
                vision_adapter.start()
                if vision_adapter.vision_enabled:
                    vision_init_ok = True
                    print(f"    VisionAdapter:      OK (config={args.vision_config})")
                else:
                    print("    VisionAdapter:      DEGRADED (模型加载失败，vision=off)")
                    vision_adapter = None
            except ImportError as e:
                print(f"    VisionAdapter:      SKIPPED (缺少依赖: {e})")
                vision_adapter = None
            except Exception as e:
                print(f"    VisionAdapter:      SKIPPED ({e})")
                vision_adapter = None

        # ── IMUReader（可选） ──
        imu_reader = None
        imu_init_ok = False
        if args.dashboard_mode == "hybrid" and args.enable_imu:
            try:
                from src.sensors.imu_reader import IMUReader

                # 加载 IMU 串口配置
                import yaml
                ports_path = Path("configs/sensor_ports.yaml")
                with open(ports_path, "r", encoding="utf-8") as f:
                    ports_cfg = yaml.safe_load(f)
                # 按平台选择配置：优先 windows → dk2500
                platform_cfg = ports_cfg.get("windows", ports_cfg.get("dk2500", {}))
                imu_cfg = platform_cfg.get("imu", {})
                imu_reader = IMUReader(mode="real", config=imu_cfg)
                imu_reader.start()
                if imu_reader._serial is not None:
                    imu_init_ok = True
                    print(f"    IMUReader:           OK (port={imu_cfg.get('port', '?')})")
                else:
                    print(f"    IMUReader:           DEGRADED (串口打开失败，imu=mock)")
                    imu_reader = None
            except Exception as e:
                print(f"    IMUReader:           SKIPPED ({e})")
                imu_reader = None

        if args.dashboard_mode == "mock":
            print("    → 已降级到 mock 模式")
        print("-" * 55)

    # ── 4. 启动后台状态更新线程 ──
    if args.cloud_enable:
        from src.dashboard.cloud_sync import CloudSyncClient

        cloud_sync = CloudSyncClient(
            base_url=args.cloud_url,
            device_id=args.device_id,
            camera=camera,
            spool_dir=_project_root / args.cloud_spool,
            state_hz=args.cloud_state_hz,
            video_fps=args.cloud_video_fps,
            segment_seconds=args.cloud_video_seconds,
        )
        cloud_sync.start()

    interval = 1.0 / max(args.state_hz, 1)
    stop_event = threading.Event()
    _start_time = time.time()
    state_thread = threading.Thread(
        target=dashboard_state_loop,
        kwargs={
            "state_store": state_store,
            "camera_producer": camera,
            "stop_event": stop_event,
            "mode": args.dashboard_mode,
            "risk_model": risk_model,
            "classifier": classifier,
            "synchronizer": synchronizer,
            "vision_adapter": vision_adapter,
            "enable_vision": args.enable_vision and vision_init_ok,
            "imu_reader": imu_reader,
            "enable_imu": args.enable_imu and imu_init_ok,
            "radar_reader": radar_reader,
            "gps_reader": gps_reader,
            "recorder": recorder,
            "cloud_sync": cloud_sync,
            "sync_thresholds": sync_thresholds,
            "risk_rule": competition_risk_rule,
            "motor": motor_controller,
            "target_stale_ms": args.target_stale_ms,
            "radar_communication_watchdog_ms": args.radar_communication_watchdog_ms,
            "interval": interval,
            "start_time": _start_time,
            "state_hz": args.state_hz,
        },
        daemon=True,
        name="dashboard-state-loop",
    )
    state_thread.start()

    # ── 5. 注入到 FastAPI 服务 ──
    from src.dashboard import server

    server.inject_camera(camera)
    server.inject_state_store(state_store)

    # ── 6. 启动 uvicorn ──
    import uvicorn

    print("=" * 55)
    print("  骑手前向安全预警 Dashboard")
    print(f"  模式:     {args.dashboard_mode}")
    print(f"  地址:     http://{args.host}:{args.port}")
    print(f"  摄像头:   {'已就绪' if camera.is_available else '不可用（使用回退图）'}")
    print(f"  状态更新: 后台线程 ({args.state_hz} Hz)")
    if args.dashboard_mode == "hybrid":
        print(f"  RiskModel: 真实计算")
        print(f"  GPS/IMU/Radar: {'IMU=real, ' if (args.enable_imu and imu_init_ok) else ''}其余 mock（不打开串口）")
        print(f"  Vision: {'已启用' if (args.enable_vision and vision_init_ok) else '关闭'}")
    elif args.dashboard_mode == "real":
        print(f"  GPS/Radar: real ({args.profile})")
        print(f"  Vision: {'enabled' if vision_init_ok else 'disabled'}")
        print(f"  Recording: {recorder.session_dir if recorder else 'disabled'}")
        print(f"  Competition risk rule: {'enabled' if competition_risk_rule else 'disabled'}")
        print(f"  Motor: {args.motor_mode if motor_controller else 'off'}")
        print("  IMU: disabled")
    print("=" * 55)

    try:
        uvicorn.run(
            "src.dashboard.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nDashboard 已停止")
    finally:
        stop_event.set()
        # 串口读取和视觉推理都有有界超时；必须等写线程真正退出后再关闭Recorder，
        # 否则可能出现后台线程向已关闭文件写入的竞争。
        state_thread.join()
        if cloud_sync is not None:
            cloud_sync.close()

        # 释放 VisionAdapter
        if vision_adapter is not None:
            try:
                if hasattr(vision_adapter, "stop"):
                    vision_adapter.stop()
            except Exception as e:
                print(f"[清理] VisionAdapter.stop() 异常: {e}")

        # 释放 IMUReader
        if imu_reader is not None:
            try:
                imu_reader.stop()
            except Exception as e:
                print(f"[清理] IMUReader.stop() 异常: {e}")

        if radar_reader is not None:
            radar_reader.stop()
        if gps_reader is not None:
            gps_reader.stop()
        if motor_controller is not None:
            motor_controller.stop()
        if recorder is not None:
            recorder.close()
            result = subprocess.run(
                [sys.executable, str(_project_root / "scripts" / "check_dashboard_recording.py"),
                 str(recorder.session_dir)], check=False)
            if result.returncode != 0:
                print(f"[QualityCheck] 录制质量检查未通过 (exit={result.returncode})")

        camera.release()


if __name__ == "__main__":
    main()
