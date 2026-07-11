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
    while not stop_event.is_set():
        try:
            loop_start = time.time()
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
                bgr_frame = camera_producer.get_bgr_frame() if (enable_vision or recorder is not None) else None
                state = build_real_sensor_state(
                    camera_available=camera_producer.is_available,
                    radar_reader=radar_reader,
                    gps_reader=gps_reader,
                    frame=bgr_frame,
                    vision_adapter=vision_adapter,
                    fusion_engine=synchronizer,
                    recorder=recorder,
                )
                if vision_adapter is not None:
                    _cue_vision_result_cache(vision_adapter)
            else:
                state = build_mock_state(camera_available=camera_producer.is_available)

            # 统计循环耗时（毫秒）
            state_loop_ms = round((time.time() - loop_start) * 1000.0, 2)

            # 附加运行时信息
            attach_runtime_info(
                state,
                start_time=start_time,
                state_hz=state_hz,
                camera_available=camera_producer.is_available,
                enable_vision=enable_vision,
                state_loop_ms=state_loop_ms,
            )
            state_store.set_state(state)
        except Exception as e:
            print(f"[StateLoop] 状态更新异常: {e}")
        time.sleep(interval)
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
    args = parser.parse_args()

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

    if args.dashboard_mode == "real":
        import yaml
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
            recorder = DashboardRecorder(_project_root / args.record_output, args.scene, args.profile)
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
        print("  IMU/Risk model: disabled")
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
        state_thread.join(timeout=2.0)

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
        if recorder is not None:
            recorder.close()

        camera.release()


if __name__ == "__main__":
    main()
