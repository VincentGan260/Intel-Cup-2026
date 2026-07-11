"""DK-2500 live chain: camera/radar -> vision -> physical rule -> DRV2605.

The latency metric starts at the timestamp of the camera frame consumed by
inference and ends immediately after the DRV2605 GO register is written.  It
therefore measures software sensor-to-actuator-command latency, not the motor's
mechanical rise time.
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
    p.add_argument("--radar-lateral-error-m", type=float, required=True,
                   help="datasheet or measured lateral error")
    p.add_argument("--mounting-offset-m", type=float, default=0.0,
                   help="signed radar offset from bicycle axis; right is positive")
    p.add_argument("--mounting-uncertainty-m", type=float, default=0.06,
                   help="uncertainty when the observed offset is about 5-6 cm")
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

    camera = CameraFrameProducer(camera_id=args.camera_id)
    radar = RadarReader(mode="real", config=profile["radar"])
    vision = VisionAdapter(pipeline_config_path=args.vision_config,
                           vision_enabled=True, use_camera=False)
    motor = TimedMotorController(mode=args.motor,
                                 i2c_bus=int(profile["motor"]["i2c_bus"]),
                                 i2c_addr=motor_addr)
    rule = PhysicalRiskRule(
        body_width_m=args.body_width_m,
        radar_lateral_error_m=args.radar_lateral_error_m,
        mounting_offset_m=args.mounting_offset_m,
        mounting_uncertainty_m=args.mounting_uncertainty_m,
    )
    tracker = LatencyTracker(ROOT / args.latency_log)
    fuser = VisionRadarFusion()

    radar.start()
    vision.start()
    motor.start()
    print(json.dumps({
        "event": "started", "platform": args.profile, "motor": args.motor,
        "corridor_half_width_m": rule.corridor_half_width_m,
        "latency_definition": "camera_frame_timestamp_to_DRV2605_GO_write",
        "latency_log": str(ROOT / args.latency_log),
    }, ensure_ascii=False))

    count = 0
    try:
        while not args.loops or count < args.loops:
            frame, capture_ns, frame_id = camera.get_bgr_frame_with_timestamp()
            if frame is None or capture_ns <= 0:
                time.sleep(0.05)
                continue
            radar_data = radar.read_once()

            # Safety fast path: decide and dispatch before the 135 ms vision
            # pipeline. Vision is auxiliary and must never delay radar alerts.
            radar_decision_start_ns = time.monotonic_ns()
            elapsed_s = (radar_decision_start_ns - capture_ns) / 1_000_000_000.0
            decision = rule.decide(radar_data, elapsed_s)
            if decision.level == 0:
                motor.alert_low()
            elif decision.level == 1:
                motor.alert_medium()
            else:
                motor.alert_high()

            latency_ms = None
            if motor.last_command_was_dispatched:
                latency_ms = tracker.add(
                    sample_start_ns=capture_ns,
                    command_dispatch_ns=motor.last_dispatch_monotonic_ns,
                    risk_level=decision.level,
                    frame_id=frame_id,
                )

            # Auxiliary asynchronous-equivalent stage. This remains in the
            # demo loop for display, but it runs only after motor dispatch.
            vision_data = vision.process(frame)
            raw_vision = vision.get_latest_vision_result()
            fused = fuser.fuse_vision_result(raw_vision, radar_data) if raw_vision else None
            count += 1
            print(json.dumps({
                "frame": frame_id,
                "risk_level": decision.level,
                "risk_label": decision.label,
                "reason": decision.reason,
                "min_path_ttc_s": decision.min_path_ttc_s,
                "high_boundary_s": round(decision.high_boundary_s, 3),
                "path_targets": decision.path_target_count,
                "radar_valid": bool(radar_data.valid),
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
        vision.stop()
        radar.stop()
        camera.release()


if __name__ == "__main__":
    main()
