"""Start the real Dashboard in data-collection mode.

This mode runs the full camera + vision + radar frontend and writes aligned
training samples under data/recordings. GPS is optional and will be recorded as
invalid when it has no fix.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dashboard_launch_utils import PROJECT_ROOT, copy_session, newest_session, run_dashboard, zip_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Start RiderGuardian collection mode")
    parser.add_argument("--scene", default="outdoor_radar_vision_collection")
    parser.add_argument("--profile", default="dk2500", choices=["windows", "dk2500"])
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-hz", type=int, default=5)
    parser.add_argument("--record-output", default="data/recordings")
    parser.add_argument("--operator", default="team")
    parser.add_argument("--route", default="unknown")
    parser.add_argument("--weather", default="unknown")
    parser.add_argument("--road-condition", default="unknown")
    parser.add_argument("--group-id", default="")
    parser.add_argument("--risk-label", choices=["low", "mid", "high"], default=None)
    parser.add_argument("--wait-gps", action="store_true", help="only use after GPS is repaired")
    parser.add_argument("--export-dir", default="", help="copy the finished session here after Ctrl+C")
    parser.add_argument("--zip", action="store_true", help="also create a zip under data/exports")
    args = parser.parse_args()

    dashboard_args = [
        "--dashboard-mode", "real",
        "--profile", args.profile,
        "--camera-id", str(args.camera_id),
        "--host", args.host,
        "--port", str(args.port),
        "--state-hz", str(args.state_hz),
        "--enable-vision",
        "--record",
        "--scene", args.scene,
        "--record-output", args.record_output,
        "--operator", args.operator,
        "--route", args.route,
        "--weather", args.weather,
        "--road-condition", args.road_condition,
        "--group-id", args.group_id or args.scene,
    ]
    if args.risk_label:
        dashboard_args.extend(["--risk-label", args.risk_label])
    if args.wait_gps:
        dashboard_args.append("--wait-gps")

    started_at = time.time()
    code = run_dashboard(dashboard_args)

    record_root = (PROJECT_ROOT / args.record_output).resolve()
    session = newest_session(record_root, started_at)
    if session is None:
        print("[collection] no new recording session found")
        return code

    print(f"[collection] session: {session}")
    if args.export_dir:
        copied = copy_session(session, Path(args.export_dir).expanduser().resolve())
        print(f"[collection] copied to: {copied}")
    if args.zip:
        archive = zip_session(session, PROJECT_ROOT / "data" / "exports")
        print(f"[collection] zip: {archive}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
