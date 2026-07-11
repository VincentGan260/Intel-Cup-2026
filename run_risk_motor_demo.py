"""DK-2500 live chain: camera/radar -> vision -> physical rule -> DRV2605.

The primary latency metric starts at the host receive timestamp of the LD2451
sample and ends immediately after the DRV2605 GO register is written. It does
not include the motor's mechanical rise time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.actuator.timed_motor_controller import TimedMotorController
from src.dashboard.frame_producer import CameraFrameProducer
from src.diagnostics.latency_tracker import LatencyTracker
from src.fusion.physical_risk_rule import PhysicalRiskRule
from src.fusion.vision_adapter import VisionAdapter
from src.fusion.vision_radar_fusion import VisionRadarFusion
from src.sensors.radar_reader import RadarReader

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DK-2500 risk-to-vibration demo and P95 logger")
    p.add_argument("--profile", default="dk2500", choices=["dk2500", "windows"])
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--vision-config", default="configs/vision/vision_pipeline.yaml")
    p.add_argument("--motor", choices=["mock", "real"], default="mock")
    p.add_argument("--confirm-motor-real", action="store_true")
    p.add_argument("--body-width-m", type=float, required=True,
                   help="measured maximum bicycle+rider width")
    p.add_argument("--point-gate-lateral-margin-m", "--radar-lateral-error-m",
                   dest="point_gate_lateral_margin_m", type=float, required=True,
                   help="prototype point-gate engineering margin (not datasheet accuracy)")
    p.add_argument("--mounting-offset-m", type=float, default=0.0,
                   help="signed radar offset from bicycle axis; right is positive")
    p.add_argument("--mounting-uncertainty-m", type=float, default=0.06,
                   help="uncertainty when the observed offset is about 5-6 cm")
    p.add_argument("--configured-warning-range-m", type=float, default=10.0,
                   help="LD2451 APP maximum detection-distance configuration")
    p.add_argument("--target-stale-ms", type=float, default=500.0)
    p.add_argument("--radar-communication-watchdog-ms", type=float, default=2000.0)
    p.add_argument("--force-alert-level", choices=[0, 1, 2], type=int, default=None,
                   help="safe latency fixture: dispatch this level for every fresh radar frame")
    p.add_argument("--latency-only", action="store_true",
                   help="measure each new radar frame immediately; skip camera and vision")
    p.add_argument("--overwrite-latency-log", action="store_true",
                   help="truncate the latency log instead of appending old samples")
    p.add_argument("--latency-log", default="logs/dk2500_e2e_latency.jsonl")
    p.add_argument("--loops", type=int, default=0, help="0 means run until Ctrl+C")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.motor == "real" and not args.confirm_motor_real:
        raise SystemExit("Real vibration requires --confirm-motor-real")

    ports = yaml.safe_load((ROOT / "configs/sensor_ports.yaml").read_text(encoding="utf-8"))
    profile = ports[args.profile]
    motor_addr = profile["motor"].get("driver_address", "0x5A")
    motor_addr = int(motor_addr, 0) if isinstance(motor_addr, str) else int(motor_addr)

    camera = None if args.latency_only else CameraFrameProducer(camera_id=args.camera_id)
    radar = RadarReader(mode="real", config=profile["radar"])
    vision = (None if args.latency_only else
              VisionAdapter(pipeline_config_path=args.vision_config,
                            vision_enabled=True, use_camera=False))
    motor = TimedMotorController(mode=args.motor,
                                 i2c_bus=int(profile["motor"]["i2c_bus"]),
                                 i2c_addr=motor_addr)
    rule = PhysicalRiskRule(
        body_width_m=args.body_width_m,
        point_gate_lateral_margin_m=args.point_gate_lateral_margin_m,
        mounting_offset_m=args.mounting_offset_m,
        mounting_uncertainty_m=args.mounting_uncertainty_m,
        configured_warning_range_m=args.configured_warning_range_m,
        radar_to_motor_p95_s=0.0,
    )
    tracker = LatencyTracker(ROOT / args.latency_log,
                             overwrite=args.overwrite_latency_log)
    fuser = None if args.latency_only else VisionRadarFusion()

    radar.start()
    if vision is not None:
        vision.start()
    motor.start()
    if args.motor == "real" and motor.is_mock:
        if vision is not None:
            vision.stop()
        radar.stop()
        if camera is not None:
            camera.release()
        tracker.close()
        raise SystemExit(
            "Real motor was requested but the controller fell back to mock. "
            "Fix smbus2/I2C permissions before collecting P95."
        )
    print(json.dumps({
        "event": "started", "platform": args.profile,
        "motor_requested": args.motor,
        "motor_actual": "mock" if motor.is_mock else "real",
        "point_gate_half_width_m": rule.point_gate_half_width_m,
        "latency_definition": "LD2451_host_receive_timestamp_to_DRV2605_GO_write",
        "latency_log": str(ROOT / args.latency_log),
    }, ensure_ascii=False))

    count = 0
    last_processed_radar_sample_ns = 0
    try:
        while not args.loops or count < args.loops:
            radar_data = radar.read_once()
            radar_sample_ns = int(getattr(radar, "last_sample_monotonic_ns", 0) or 0)
            is_new_radar_sample = (
                radar_sample_ns > 0
                and radar_sample_ns > last_processed_radar_sample_ns
            )
            if args.latency_only and not is_new_radar_sample:
                time.sleep(0.002)
                continue
            if is_new_radar_sample:
                last_processed_radar_sample_ns = radar_sample_ns

            radar_now_ns = time.monotonic_ns()
            radar_age_ms = ((radar_now_ns - radar_sample_ns) / 1_000_000.0
                            if radar_sample_ns else None)
            stale_limit = (args.target_stale_ms if radar_data.targets
                           else args.radar_communication_watchdog_ms)
            radar_fresh = (radar_age_ms is not None and
                           0.0 <= radar_age_ms <= stale_limit)

            # Safety fast path: decide and dispatch before the 135 ms vision
            # pipeline. Vision is auxiliary and must never delay radar alerts.
            decision = rule.decide(radar_data, radar_fresh=radar_fresh)
            dispatched_level = decision.level
            if (args.force_alert_level is not None and is_new_radar_sample
                    and radar_fresh and radar_data.valid):
                dispatched_level = args.force_alert_level
                if dispatched_level == 0:
                    motor.alert_low()
                elif dispatched_level == 1:
                    motor.alert_medium()
                else:
                    motor.alert_high()
            elif decision.status in {"unknown", "degraded"}:
                pass
            elif decision.level == 0:
                motor.alert_low()
            elif decision.level == 1:
                motor.alert_medium()
            else:
                motor.alert_high()

            latency_ms = None
            frame_id = count
            if motor.last_command_was_dispatched:
                latency_ms = tracker.add(
                    sample_start_ns=radar_sample_ns,
                    command_dispatch_ns=motor.last_dispatch_monotonic_ns,
                    risk_level=dispatched_level,
                    frame_id=frame_id,
                )

            vision_data = None
            fused = None
            if not args.latency_only:
                frame, capture_ns, frame_id = camera.get_bgr_frame_with_timestamp()
                if frame is not None and capture_ns > 0:
                    # Auxiliary stage runs only after the radar-to-motor path.
                    vision_data = vision.process(frame)
                    raw_vision = vision.get_latest_vision_result()
                    fused = fuser.fuse_vision_result(raw_vision, radar_data) if raw_vision else None
            if is_new_radar_sample:
                count += 1
            print(json.dumps({
                "sample": count,
                "risk_level": decision.level,
                "dispatched_level": dispatched_level,
                "forced_alert": args.force_alert_level is not None,
                "risk_label": decision.label,
                "risk_status": decision.status,
                "reason": decision.reason,
                "min_path_ttc_s": decision.min_path_ttc_s,
                "urgent_ttc_s": round(decision.urgent_ttc_s, 3),
                "path_targets": decision.path_target_count,
                "radar_valid": bool(radar_data.valid),
                "radar_fresh": radar_fresh,
                "radar_age_ms": radar_age_ms,
                "vision_valid": bool(getattr(vision_data, "valid", False)),
                "vision_radar_matches": int(fused.n_vision_radar) if fused else 0,
                "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
                "latency_summary": tracker.summary(),
            }, ensure_ascii=False))
    except KeyboardInterrupt:
        pass
    finally:
        print(json.dumps({"event": "finished", "latency_summary": tracker.summary()},
                         ensure_ascii=False))
        tracker.close()
        motor.stop()
        if vision is not None:
            vision.stop()
        radar.stop()
        if camera is not None:
            camera.release()


if __name__ == "__main__":
    main()
