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
import signal
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


def vision_inference_loop(camera_producer, vision_adapter, snapshot_store,
                          stop_event: threading.Event,
                          warning_system=None, vision_rule=None) -> None:
    """Run slow vision inference independently from state and cloud upload."""
    print("[VisionWorker] asynchronous inference thread started")
    last_frame_id = -1
    while not stop_event.is_set():
        frame, capture_ns, frame_id = camera_producer.get_bgr_frame_with_timestamp()
        if frame is None:
            stop_event.wait(0.1)
            continue
        if frame_id <= last_frame_id:
            stop_event.wait(0.01)
            continue
        last_frame_id = frame_id
        started_ns = time.monotonic_ns()
        vision_data = vision_adapter.process(frame)
        completed_ns = time.monotonic_ns()
        vision_result = vision_adapter.get_latest_vision_result()
        from src.dashboard.vision_snapshot import VisionSnapshot

        snapshot_store.publish(VisionSnapshot(
            source_frame_id=frame_id,
            source_frame=frame,
            frame_capture_monotonic_ns=capture_ns,
            vision_start_monotonic_ns=started_ns,
            vision_finish_monotonic_ns=completed_ns,
            vision_data=vision_data,
            vision_result=vision_result,
        ))
        if warning_system is not None and vision_rule is not None:
            event = vision_rule.evaluate(
                (vision_result if bool(getattr(vision_data, "valid", False)) else None),
                source_frame_id=frame_id,
                capture_monotonic_ns=capture_ns,
                completed_monotonic_ns=completed_ns,
                sequence=frame_id + 1,
            )
            warning_system.publish_vision(event)
        _cue_vision_result_cache(vision_adapter)
        elapsed = (completed_ns - started_ns) / 1_000_000_000.0
        if elapsed >= 5.0:
            print(f"[VisionWorker] inference completed in {elapsed:.2f}s")
        stop_event.wait(0.05)
    print("[VisionWorker] asynchronous inference thread stopped")


def radar_warning_loop(radar_reader, risk_rule, warning_system,
                       stop_event: threading.Event,
                       target_stale_ms: float = 500.0,
                       radar_communication_watchdog_ms: float = 2000.0) -> None:
    """Read radar independently; urgent events never wait for vision or Dashboard."""
    print("[RadarWorker] independent warning thread started")
    sequence = 0
    while not stop_event.is_set():
        radar = radar_reader.read_once()
        completed_ns = time.monotonic_ns()
        packet_ns = int(getattr(radar_reader, "last_sample_monotonic_ns", 0) or 0)
        age_ms = ((completed_ns - packet_ns) / 1_000_000.0 if packet_ns else float("inf"))
        stale_limit_ms = (target_stale_ms if getattr(radar, "targets", [])
                          else radar_communication_watchdog_ms)
        fresh = 0.0 <= age_ms <= stale_limit_ms
        sequence += 1
        event = risk_rule.evaluate_event(
            radar, radar_fresh=fresh, sequence=sequence,
            packet_monotonic_ns=packet_ns,
            completed_monotonic_ns=completed_ns,
        )
        warning_system.publish_radar(event, fast=True)
        stop_event.wait(0.01)
    print("[RadarWorker] independent warning thread stopped")


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
    warning_system=None,
    imu_warning_rule=None,
    vision_snapshot_store=None,
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
    last_recorded_vision_version = 0
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
                vision_snapshot = None
                vision_snapshot_version = 0
                if vision_snapshot_store is not None:
                    vision_snapshot, vision_snapshot_version = vision_snapshot_store.get_snapshot()
                if vision_snapshot is not None:
                    bgr_frame = vision_snapshot.source_frame
                    frame_capture_ns = vision_snapshot.frame_capture_monotonic_ns
                    camera_frame_id = vision_snapshot.source_frame_id
                else:
                    bgr_frame, frame_capture_ns, camera_frame_id = None, 0, -1
                record_sample = bool(
                    recorder is not None
                    and vision_snapshot is not None
                    and vision_snapshot_version > last_recorded_vision_version
                )
                state = build_real_sensor_state(
                    camera_available=camera_producer.is_available,
                    radar_reader=radar_reader,
                    gps_reader=gps_reader,
                    imu_reader=imu_reader if enable_imu else None,
                    frame=bgr_frame,
                    frame_capture_monotonic_ns=frame_capture_ns,
                    camera_frame_id=camera_frame_id,
                    vision_adapter=vision_adapter,
                    vision_snapshot=vision_snapshot,
                    process_vision=False,
                    fusion_engine=synchronizer,
                    recorder=recorder,
                    sync_thresholds=sync_thresholds,
                    risk_rule=risk_rule,
                    risk_model=risk_model,
                    classifier=classifier,
                    motor=motor,
                    warning_system=warning_system,
                    imu_warning_rule=imu_warning_rule,
                    target_stale_ms=target_stale_ms,
                    radar_communication_watchdog_ms=radar_communication_watchdog_ms,
                    record_sample=record_sample,
                )
                if record_sample:
                    last_recorded_vision_version = vision_snapshot_version
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
    from src.fusion.warning_config import load_warning_rule_config

    default_warning_config = _project_root / "configs" / "warning_rules.yaml"
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--warning-config", default=str(default_warning_config))
    config_args, _ = config_parser.parse_known_args()
    warning_config_path = Path(config_args.warning_config)
    if not warning_config_path.is_absolute():
        warning_config_path = _project_root / warning_config_path
    warning_config = load_warning_rule_config(warning_config_path)
    radar_defaults = warning_config.section("radar")
    vision_defaults = warning_config.section("vision")
    gps_defaults = warning_config.section("gps")
    imu_defaults = warning_config.section("imu")
    freshness_defaults = warning_config.section("freshness")
    state_defaults = warning_config.section("state")

    parser = argparse.ArgumentParser(description="骑手前向安全预警 Dashboard")
    parser.add_argument("--warning-config", default=str(warning_config_path),
                        help="versioned competition warning-rule configuration")
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
    parser.add_argument("--cloud-video-queue-size", type=int, default=8,
                        help="maximum cloud video segments held in memory")
    parser.add_argument("--cloud-spool-max-gb", type=float, default=2.0,
                        help="pause new cloud video recording when the spool reaches this size")
    parser.add_argument(
        "--enable-vision", action="store_true",
        help="hybrid 模式下启用视觉推理（需要 openvino 环境）",
    )
    parser.add_argument(
        "--enable-imu", action="store_true",
        help="real / hybrid 模式下启用真实 IMU（WT61C 串口传感器）",
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
    parser.add_argument("--configured-warning-range-m", type=float,
                        default=radar_defaults.get("configured_warning_range_m"),
                        help="与LD2451 APP最远检测距离一致的比赛工作范围")
    parser.add_argument("--radar-parsed-to-motor-go-p95-ms", "--radar-to-motor-p95-ms",
                        dest="radar_parsed_to_motor_go_p95_ms", type=float,
                        default=float(radar_defaults["radar_parsed_to_motor_go_p95_ms"]),
                        help="雷达帧解析完成到DRV2605 GO写入的P95，默认0.469 ms")
    parser.add_argument("--point-gate-lateral-margin-m", type=float,
                        choices=[0.015, 0.025, 0.035],
                        default=float(radar_defaults["point_gate_lateral_margin_m"]),
                        help="competition point-gate lateral margin candidate")
    parser.add_argument("--urgent-reference-s", type=float, choices=[2.0, 2.5, 3.0],
                        default=float(radar_defaults["urgent_reference_s"]),
                        help="reaction-time urgency reference candidate; not a stopping threshold")
    parser.add_argument("--attention-reference-s", type=float,
                        default=float(radar_defaults["attention_reference_s"]),
                        help="attention-time reference; not a stopping threshold")
    parser.add_argument("--vision-path-policy", choices=["any", "center", "two_of_three"],
                        default=str(vision_defaults["path_policy"]),
                        help="visual path obstacle rule: any, center or two_of_three bottom points")
    parser.add_argument("--vision-corridor-top-y-ratio", type=float,
                        default=float(vision_defaults["corridor_top_y_ratio"]))
    parser.add_argument("--vision-corridor-top-width-ratio", type=float,
                        default=float(vision_defaults["corridor_top_width_ratio"]))
    parser.add_argument("--vision-corridor-bottom-width-ratio", type=float,
                        default=float(vision_defaults["corridor_bottom_width_ratio"]))
    parser.add_argument("--vision-near-bottom-ratio", type=float,
                        default=float(vision_defaults["near_bottom_ratio"]))
    parser.add_argument("--vision-very-near-bottom-ratio", type=float,
                        default=float(vision_defaults["very_near_bottom_ratio"]))
    parser.add_argument("--vision-attention-tau-s", type=float,
                        default=float(vision_defaults["attention_tau_s"]))
    parser.add_argument("--vision-urgent-tau-s", type=float,
                        default=float(vision_defaults["urgent_tau_s"]))
    parser.add_argument("--vision-temporal-window-s", type=float,
                        default=float(vision_defaults["temporal_window_s"]))
    parser.add_argument("--vision-min-history-s", type=float,
                        default=float(vision_defaults["min_history_s"]))
    parser.add_argument("--vision-min-observations", type=int,
                        default=int(vision_defaults["min_observations"]))
    parser.add_argument("--vision-track-iou-threshold", type=float,
                        default=float(vision_defaults["track_iou_threshold"]))
    parser.add_argument("--target-stale-ms", type=float,
                        default=float(freshness_defaults["target_stale_ms"]),
                        help="有目标100ms周期的5倍工程容错值")
    parser.add_argument("--radar-communication-watchdog-ms", type=float,
                        default=float(freshness_defaults["radar_communication_watchdog_ms"]),
                        help="无目标约1s周期的2倍工程通信看门狗")
    parser.add_argument("--vision-stale-ms", type=float,
                        default=float(freshness_defaults["vision_stale_ms"]),
                        help="视觉事件工程时效窗口")
    parser.add_argument("--release-hold-ms", type=float,
                        default=float(state_defaults["release_hold_ms"]),
                        help="风险降级去抖窗口，只延迟降级")
    args = parser.parse_args()

    if (args.dashboard_mode == "real" and args.enable_risk_rule
            and args.enable_imu and args.state_hz < 20):
        print("[IMURisk] state loop raised to 20 Hz for three-sample urgent confirmation")
        args.state_hz = 20

    effective_warning_parameters = {
        "radar": {**radar_defaults,
                  "configured_warning_range_m": args.configured_warning_range_m,
                  "radar_parsed_to_motor_go_p95_ms": args.radar_parsed_to_motor_go_p95_ms,
                  "point_gate_lateral_margin_m": args.point_gate_lateral_margin_m,
                  "attention_reference_s": args.attention_reference_s,
                  "urgent_reference_s": args.urgent_reference_s},
        "vision": {"path_policy": args.vision_path_policy,
                   "corridor_top_y_ratio": args.vision_corridor_top_y_ratio,
                   "corridor_top_width_ratio": args.vision_corridor_top_width_ratio,
                   "corridor_bottom_width_ratio": args.vision_corridor_bottom_width_ratio,
                   "near_bottom_ratio": args.vision_near_bottom_ratio,
                   "very_near_bottom_ratio": args.vision_very_near_bottom_ratio,
                   "attention_tau_s": args.vision_attention_tau_s,
                   "urgent_tau_s": args.vision_urgent_tau_s,
                   "temporal_window_s": args.vision_temporal_window_s,
                   "min_history_s": args.vision_min_history_s,
                   "min_observations": args.vision_min_observations,
                   "track_iou_threshold": args.vision_track_iou_threshold},
        "gps": gps_defaults,
        "imu": imu_defaults,
        "freshness": {"target_stale_ms": args.target_stale_ms,
                      "vision_stale_ms": args.vision_stale_ms,
                      "gps_stale_ms": float(freshness_defaults["gps_stale_ms"]),
                      "imu_stale_ms": float(freshness_defaults["imu_stale_ms"]),
                      "radar_communication_watchdog_ms":
                          args.radar_communication_watchdog_ms},
        "state": {"release_hold_ms": args.release_hold_ms,
                  "score_variation": state_defaults["score_variation"]},
    }
    warning_rule_metadata = {
        **warning_config.metadata,
        "effective_parameters": effective_warning_parameters,
    }

    if args.enable_risk_rule:
        required_values = {
            "--configured-warning-range-m": args.configured_warning_range_m,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            parser.error("--enable-risk-rule requires configuration values: " + ", ".join(missing))
        if args.configured_warning_range_m <= 0:
            parser.error("--configured-warning-range-m must be positive")
        if args.radar_parsed_to_motor_go_p95_ms < 0:
            parser.error("parsed-to-motor GO P95 must be non-negative")
        if (args.target_stale_ms <= 0 or args.radar_communication_watchdog_ms <= 0
                or args.vision_stale_ms <= 0 or args.release_hold_ms <= 0):
            parser.error("radar watchdog values must be positive")
        if args.attention_reference_s <= args.urgent_reference_s:
            parser.error("attention reference must exceed urgent reference")
    if args.motor_mode == "real" and not args.confirm_motor_real:
        parser.error("--motor-mode real requires --confirm-motor-real")
    if args.motor_mode != "off" and not args.enable_risk_rule:
        parser.error("--motor-mode requires --enable-risk-rule")

    import yaml
    recording_cfg = yaml.safe_load(
        (_project_root / "configs" / "dashboard_recording.yaml").read_text(encoding="utf-8")
    )
    sync_thresholds = recording_cfg["sync"]
    if float(sync_thresholds["gps_max_delta_ms"]) != float(
            freshness_defaults["gps_stale_ms"]):
        parser.error(
            "warning GPS stale threshold must match dashboard recording sync threshold")

    # ── 1. 创建摄像头 ──
    from src.dashboard.frame_producer import CameraFrameProducer

    camera = CameraFrameProducer(camera_id=args.camera_id)

    # ── 2. 创建状态池 ──
    from src.dashboard.risk_score_variation import (
        RiskScoreVariation,
        RiskScoreVariationConfig,
    )
    from src.dashboard.state_store import DashboardStateStore
    from src.dashboard.vision_snapshot import VisionSnapshotStore

    score_variation_defaults = state_defaults["score_variation"]
    score_variation = RiskScoreVariation(RiskScoreVariationConfig(
        enabled=bool(score_variation_defaults["enabled"]),
        max_amplitude=float(score_variation_defaults["max_amplitude"]),
        time_constant_s=float(score_variation_defaults["time_constant_s"]),
        seed=score_variation_defaults.get("seed"),
    ))
    state_store = DashboardStateStore(score_variation=score_variation)
    vision_snapshot_store = VisionSnapshotStore()

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
    warning_system = None
    vision_warning_rule = None
    imu_warning_rule = None

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
        if args.enable_imu:
            try:
                from src.sensors.imu_reader import IMUReader

                profile_imu_cfg = profile_cfg.get("imu", {})
                if not profile_imu_cfg:
                    raise ValueError(f"profile={args.profile} 缺少 IMU 串口配置")
                imu_cfg = {
                    **profile_imu_cfg,
                    "roll_offset_deg": float(imu_defaults["roll_offset_deg"]),
                    "pitch_offset_deg": float(imu_defaults["pitch_offset_deg"]),
                }
                imu_reader = IMUReader(mode="real", config=imu_cfg)
                imu_reader.start()
                imu_init_ok = imu_reader._serial is not None
                if not imu_init_ok:
                    imu_reader = None
            except Exception as e:
                print(f"[RealSensors] IMU initialization failed: {e}")
                imu_reader = None
                imu_init_ok = False
        print(f"[RealSensors] profile={args.profile}; IMU={'enabled' if imu_init_ok else 'disabled'}")

        from src.fusion.risk_model import RiskModel
        from src.fusion.risk_level import RiskLevelClassifier
        risk_model = RiskModel(config_path=args.risk_config)
        classifier = RiskLevelClassifier(config_path=args.risk_config)
        print("[RealSensors] adaptive RiskModel enabled (IMU invalid and automatically down-weighted)")

        if args.enable_risk_rule:
            from src.fusion.physical_risk_rule import PhysicalRiskRule

            competition_risk_rule = PhysicalRiskRule(
                body_width_m=float(radar_defaults["body_width_m"]),
                point_gate_lateral_margin_m=args.point_gate_lateral_margin_m,
                mounting_offset_m=float(radar_defaults["mounting_offset_m"]),
                mounting_uncertainty_m=float(radar_defaults["mounting_uncertainty_m"]),
                configured_warning_range_m=args.configured_warning_range_m,
                radar_parsed_to_motor_go_p95_s=(
                    args.radar_parsed_to_motor_go_p95_ms / 1000.0),
                attention_reference_s=args.attention_reference_s,
                urgent_reference_s=args.urgent_reference_s,
                max_abs_angle_deg=float(radar_defaults["max_abs_angle_deg"]),
            )
            print("[CompetitionRisk] radar TTC urgency rule enabled")
            print(f"[CompetitionRisk] configured_range={args.configured_warning_range_m:.2f}m, "
                  f"point_gate_half_width={competition_risk_rule.point_gate_half_width_m:.3f}m, "
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

            from src.fusion.warning_system import MultimodalWarningSystem
            from src.fusion.gps_risk_context import GpsSpeedModifierConfig
            if imu_init_ok:
                from src.fusion.imu_warning_rule import (
                    ImuWarningRule,
                    ImuWarningRuleConfig,
                )
                imu_warning_rule = ImuWarningRule(ImuWarningRuleConfig(
                    calibration_status=str(imu_defaults["calibration_status"]),
                    roll_offset_deg=float(imu_defaults["roll_offset_deg"]),
                    pitch_offset_deg=float(imu_defaults["pitch_offset_deg"]),
                    turn_sign=float(imu_defaults["turn_sign"]),
                    gravity_mps2=float(imu_defaults["gravity_mps2"]),
                    min_turn_compensation_speed_kmh=float(
                        imu_defaults["min_turn_compensation_speed_kmh"]),
                    attention_error_deg=float(imu_defaults["attention_error_deg"]),
                    critical_error_deg=float(imu_defaults["critical_error_deg"]),
                    attention_outward_rate_deg_s=float(
                        imu_defaults["attention_outward_rate_deg_s"]),
                    urgent_outward_rate_deg_s=float(
                        imu_defaults["urgent_outward_rate_deg_s"]),
                    attention_persistence_ms=float(
                        imu_defaults["attention_persistence_ms"]),
                    prediction_horizon_s=float(imu_defaults["prediction_horizon_s"]),
                    urgent_min_error_deg=float(imu_defaults["urgent_min_error_deg"]),
                    urgent_consistent_samples=int(
                        imu_defaults["urgent_consistent_samples"]),
                    max_sample_gap_ms=float(imu_defaults["max_sample_gap_ms"]),
                ))
            warning_system = MultimodalWarningSystem(
                motor=motor_controller,
                target_stale_ms=args.target_stale_ms,
                vision_stale_ms=args.vision_stale_ms,
                gps_stale_ms=float(freshness_defaults["gps_stale_ms"]),
                imu_stale_ms=float(freshness_defaults["imu_stale_ms"]),
                imu_enabled=bool(imu_init_ok),
                gps_modifier_config=GpsSpeedModifierConfig(
                    neutral_below_kmh=float(gps_defaults["neutral_below_kmh"]),
                    full_effect_kmh=float(gps_defaults["full_effect_kmh"]),
                    max_factor=float(gps_defaults["max_factor"]),
                ),
                release_hold_ms=args.release_hold_ms,
                radar_communication_watchdog_ms=args.radar_communication_watchdog_ms,
                rule_config_metadata=warning_rule_metadata,
            )

        if args.enable_vision:
            from src.fusion.vision_adapter import VisionAdapter
            from src.fusion.vision_radar_fusion import VisionRadarFusion
            vision_adapter = VisionAdapter(
                pipeline_config_path=args.vision_config, vision_enabled=True, use_camera=False)
            vision_adapter.start()
            vision_init_ok = bool(vision_adapter.vision_enabled)
            if vision_init_ok:
                synchronizer = VisionRadarFusion()
                from src.fusion.vision_warning_rule import VisionWarningRule
                vision_warning_rule = VisionWarningRule(
                    path_policy=args.vision_path_policy,
                    corridor_top_y_ratio=args.vision_corridor_top_y_ratio,
                    corridor_top_width_ratio=args.vision_corridor_top_width_ratio,
                    corridor_bottom_width_ratio=args.vision_corridor_bottom_width_ratio,
                    near_bottom_ratio=args.vision_near_bottom_ratio,
                    very_near_bottom_ratio=args.vision_very_near_bottom_ratio,
                    attention_tau_s=args.vision_attention_tau_s,
                    urgent_tau_s=args.vision_urgent_tau_s,
                    temporal_window_s=args.vision_temporal_window_s,
                    min_history_s=args.vision_min_history_s,
                    min_observations=args.vision_min_observations,
                    track_iou_threshold=args.vision_track_iou_threshold,
                )
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
                              "group_id": args.group_id or args.scene,
                              "warning_rule_config": warning_rule_metadata}
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
                ports_path = _project_root / "configs" / "sensor_ports.yaml"
                with open(ports_path, "r", encoding="utf-8") as f:
                    ports_cfg = yaml.safe_load(f)
                platform_cfg = ports_cfg.get(args.profile, {})
                profile_imu_cfg = platform_cfg.get("imu", {})
                if not profile_imu_cfg:
                    raise ValueError(f"profile={args.profile} 缺少 IMU 串口配置")
                imu_cfg = {
                    **profile_imu_cfg,
                    "roll_offset_deg": float(imu_defaults["roll_offset_deg"]),
                    "pitch_offset_deg": float(imu_defaults["pitch_offset_deg"]),
                }
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
            video_queue_size=args.cloud_video_queue_size,
            spool_max_gb=args.cloud_spool_max_gb,
        )
        cloud_sync.start()

    interval = 1.0 / max(args.state_hz, 1)
    stop_event = threading.Event()
    vision_thread = None
    radar_thread = None
    if args.dashboard_mode == "real" and vision_adapter is not None:
        vision_thread = threading.Thread(
            target=vision_inference_loop,
            args=(camera, vision_adapter, vision_snapshot_store, stop_event,
                  warning_system, vision_warning_rule),
            daemon=True,
            name="dashboard-vision-inference",
        )
        vision_thread.start()
    if (args.dashboard_mode == "real" and radar_reader is not None
            and competition_risk_rule is not None and warning_system is not None):
        radar_thread = threading.Thread(
            target=radar_warning_loop,
            args=(radar_reader, competition_risk_rule, warning_system, stop_event,
                  args.target_stale_ms, args.radar_communication_watchdog_ms),
            daemon=True,
            name="dashboard-radar-warning",
        )
        radar_thread.start()
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
            "warning_system": warning_system,
            "imu_warning_rule": imu_warning_rule,
            "vision_snapshot_store": vision_snapshot_store,
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
        print(f"  IMU: {'real' if imu_init_ok else 'disabled'}")
        print(f"  Vision: {'enabled' if vision_init_ok else 'disabled'}")
        print(f"  Recording: {recorder.session_dir if recorder else 'disabled'}")
        print(f"  Competition risk rule: {'enabled' if competition_risk_rule else 'disabled'}")
        print(f"  Motor: {args.motor_mode if motor_controller else 'off'}")
    print("=" * 55)

    # Uvicorn 0.51 re-raises captured signals after its own graceful shutdown.
    # Installing non-terminating application handlers first makes that final
    # signal return control here so the outer finally block always seals data.
    received_signal = threading.Event()

    def _request_shutdown(signum, _frame):
        received_signal.set()
        print(f"[Shutdown] received signal {signum}; finalizing resources")

    previous_handlers = {}
    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[shutdown_signal] = signal.getsignal(shutdown_signal)
        signal.signal(shutdown_signal, _request_shutdown)

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
        # Release the exclusive V4L2 handle first. Video/vision workers only
        # consume copied cached frames, so keeping /dev/video0 open during the
        # potentially slow upload shutdown prevents an immediate restart.
        camera.release()
        # 串口读取和视觉推理都有有界超时；必须等写线程真正退出后再关闭Recorder，
        # 否则可能出现后台线程向已关闭文件写入的竞争。
        state_thread.join(timeout=5.0)
        if state_thread.is_alive():
            print("[清理] 状态线程未在5秒内退出，将随主进程结束")
        if vision_thread is not None:
            vision_thread.join(timeout=10.0)
        if radar_thread is not None:
            radar_thread.join(timeout=5.0)
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
        # Recorder and sensors are now safely sealed. Cloud shutdown may wait
        # on a network request, so it must never delay session finalization.
        if cloud_sync is not None:
            cloud_sync.close()
        for shutdown_signal, previous_handler in previous_handlers.items():
            signal.signal(shutdown_signal, previous_handler)



if __name__ == "__main__":
    main()
