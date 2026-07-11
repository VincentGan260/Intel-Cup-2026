"""One-command DK-2500 sensor-to-vibration P95 measurement.

Fixed measured geometry:
  body width: 0.66 m
  radar lateral error: 0.025 m
  radar mounting offset: 0.055 m left (right-positive convention => -0.055)
  mounting measurement uncertainty: 0.005 m
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure DK-2500 camera-to-DRV2605 P95")
    parser.add_argument("--confirm-motor-real", action="store_true",
                        help="required before the real vibration motor is driven")
    parser.add_argument("--mock", action="store_true",
                        help="verify the pipeline without driving the motor")
    parser.add_argument("--loops", type=int, default=0,
                        help="0 runs until Ctrl+C")
    parser.add_argument("--log", default="",
                        help="JSONL output; default creates a new timestamped file")
    args = parser.parse_args()

    motor_mode = "mock" if args.mock else "real"
    if motor_mode == "real" and not args.confirm_motor_real:
        raise SystemExit(
            "Real motor test requires --confirm-motor-real. "
            "Use --mock first to verify NPU/GPU/radar startup."
        )

    log_path = args.log or (
        "logs/dk2500_e2e_latency_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl"
    )
    forwarded = [
        "run_risk_motor_demo.py",
        "--profile", "dk2500",
        "--body-width-m", "0.66",
        "--radar-lateral-error-m", "0.025",
        "--mounting-offset-m", "-0.055",
        "--mounting-uncertainty-m", "0.005",
        "--motor", motor_mode,
        "--latency-log", log_path,
    ]
    if args.loops:
        forwarded.extend(["--loops", str(args.loops)])
    if motor_mode == "real":
        forwarded.append("--confirm-motor-real")

    print(f"Latency log: {log_path}")
    sys.argv = forwarded
    from run_risk_motor_demo import main as run_demo
    run_demo()


if __name__ == "__main__":
    main()
