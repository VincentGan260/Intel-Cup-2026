"""Start the real Dashboard in final-demo mode.

This mode runs the same camera + vision + radar + frontend pipeline as
collection mode, but it does not write local training recordings.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scripts.dashboard_launch_utils import run_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runtime_python() -> Path | None:
    """Locate the project runtime without requiring the shell to activate it."""
    explicit = os.environ.get("RIDERGUARDIAN_PYTHON", "").strip()
    candidates = [Path(explicit)] if explicit else []
    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    if conda_exe:
        candidates.append(Path(conda_exe).resolve().parent.parent / "envs/intel/bin/python")
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        candidates.append(parent / "envs/intel/bin/python")
    candidates.append(Path.home() / "miniconda3/envs/intel/bin/python")
    candidates.extend([
        PROJECT_ROOT / ".venv/Scripts/python.exe",
        PROJECT_ROOT / ".venv/bin/python",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _ensure_runtime_environment() -> None:
    runtime = _runtime_python()
    current = Path(sys.executable).resolve()
    if runtime is None or runtime == current:
        return
    print(f"[launcher] switching runtime: {current} -> {runtime}", flush=True)
    os.execv(
        str(runtime),
        [str(runtime), str(PROJECT_ROOT / "start_demo.py"), *sys.argv[1:]],
    )


def main() -> int:
    _ensure_runtime_environment()
    parser = argparse.ArgumentParser(description="Start RiderGuardian demo mode")
    parser.add_argument("--profile", default="dk2500", choices=["windows", "dk2500"])
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-hz", type=int, default=20)
    parser.add_argument("--warning-config", default="configs/warning_rules.yaml")
    parser.add_argument("--configured-warning-range-m", type=float, default=None,
                        help="optional temporary override; normally set this in warning_rules.yaml")
    parser.add_argument("--disable-imu", action="store_true",
                        help="diagnostic fallback only; final demo enables IMU by default")
    parser.add_argument("--disable-risk-rule", action="store_true",
                        help="diagnostic fallback only; final demo enables risk rules by default")
    parser.add_argument("--disable-motor", action="store_true",
                        help="diagnostic fallback only; final demo enables the real motor by default")
    parser.add_argument("--cloud-url", "--cloud-endpoint", dest="cloud_url",
                        default="http://124.70.108.34",
                        help="cloud API base URL")
    parser.add_argument("--device-id", default="bike-001")
    parser.add_argument("--disable-cloud", action="store_true",
                        help="offline diagnostic mode; cloud upload is enabled by default")
    parser.add_argument("--database-url", default="", help="reserved for database integration")
    parser.add_argument("--demo-id", default="demo")
    args = parser.parse_args()

    env_updates = {
        "RIDERGUARDIAN_CLOUD_ENDPOINT": args.cloud_url,
        "RIDERGUARDIAN_DATABASE_URL": args.database_url,
        "RIDERGUARDIAN_DEMO_ID": args.demo_id,
    }
    if args.database_url:
        print("[demo] database variable exported for later integration")

    dashboard_args = [
        "--dashboard-mode", "real",
        "--profile", args.profile,
        "--camera-id", str(args.camera_id),
        "--host", args.host,
        "--port", str(args.port),
        "--state-hz", str(args.state_hz),
        "--enable-vision",
        "--warning-config", args.warning_config,
        "--motor-mode", "off" if args.disable_motor else "real",
    ]
    if not args.disable_imu:
        dashboard_args.append("--enable-imu")
    if not args.disable_risk_rule:
        dashboard_args.append("--enable-risk-rule")
    if not args.disable_motor:
        dashboard_args.append("--confirm-motor-real")
    if not args.disable_cloud:
        dashboard_args.extend([
            "--cloud-enable",
            "--cloud-url", args.cloud_url,
            "--device-id", args.device_id,
        ])
    if args.configured_warning_range_m is not None:
        dashboard_args.extend([
            "--configured-warning-range-m", str(args.configured_warning_range_m),
        ])
    return run_dashboard(dashboard_args, env_updates=env_updates)


if __name__ == "__main__":
    raise SystemExit(main())
