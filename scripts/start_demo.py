"""Start the real Dashboard in final-demo mode.

This mode runs the same camera + vision + radar + frontend pipeline as
collection mode, but it does not write local training recordings.
"""

from __future__ import annotations

import argparse

from dashboard_launch_utils import run_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Start RiderGuardian demo mode")
    parser.add_argument("--profile", default="dk2500", choices=["windows", "dk2500"])
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--state-hz", type=int, default=5)
    parser.add_argument("--cloud-endpoint", default="", help="reserved for the cloud uploader")
    parser.add_argument("--database-url", default="", help="reserved for database integration")
    parser.add_argument("--demo-id", default="demo")
    args = parser.parse_args()

    env_updates = {
        "RIDERGUARDIAN_CLOUD_ENDPOINT": args.cloud_endpoint,
        "RIDERGUARDIAN_DATABASE_URL": args.database_url,
        "RIDERGUARDIAN_DEMO_ID": args.demo_id,
    }
    if args.cloud_endpoint or args.database_url:
        print("[demo] cloud/database variables exported for later integration")

    dashboard_args = [
        "--dashboard-mode", "real",
        "--profile", args.profile,
        "--camera-id", str(args.camera_id),
        "--host", args.host,
        "--port", str(args.port),
        "--state-hz", str(args.state_hz),
        "--enable-vision",
    ]
    return run_dashboard(dashboard_args, env_updates=env_updates)


if __name__ == "__main__":
    raise SystemExit(main())
