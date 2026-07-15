"""End-to-end mock test for Dashboard recorder, quality check and window builder."""

from __future__ import annotations

import shutil
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.dashboard_recorder import DashboardRecorder
from src.fusion.data_types import VisionData, VisionObject
from src.sensors.gps_reader import GPSReader
from src.sensors.radar_reader import RadarReader


def main() -> int:
    output = Path("/tmp/intelcup_dashboard_recording_mock")
    if output.exists():
        shutil.rmtree(output)
    cfg = yaml.safe_load((ROOT / "configs" / "dashboard_recording.yaml").read_text(encoding="utf-8"))
    recorder = DashboardRecorder(output, "mock_scene", "windows", recording_config=cfg,
                                 session_fields={"group_id": "mock-group"}, risk_label="low")
    radar_reader, gps_reader = RadarReader("mock"), GPSReader("mock")
    radar_reader.start(); gps_reader.start()
    for index in range(7):
        capture = time.monotonic_ns()
        radar_start = time.monotonic_ns(); radar = radar_reader.read_once(); radar_end = time.monotonic_ns()
        gps_start = time.monotonic_ns(); gps = gps_reader.read_once(); gps_end = time.monotonic_ns()
        vision_start = time.monotonic_ns()
        vision = VisionData(valid=True, objects=[VisionObject(class_name="obstacle", confidence=.8)],
                            drivable_area_ratio=.4, max_confidence=.8)
        vision_finish = time.monotonic_ns()
        stamps = {"frame_capture_monotonic_ns": capture,
                  "radar_read_start_monotonic_ns": radar_start,
                  "radar_read_end_monotonic_ns": radar_end,
                  "radar_sample_monotonic_ns": radar_end,
                  "gps_read_start_monotonic_ns": gps_start,
                  "gps_read_end_monotonic_ns": gps_end,
                  "gps_sample_monotonic_ns": gps_end,
                  "vision_start_monotonic_ns": vision_start,
                  "vision_finish_monotonic_ns": vision_finish,
                  "radar_delta_ms": (radar_end-capture)/1e6,
                  "gps_delta_ms": (gps_end-capture)/1e6,
                  "vision_latency_ms": (vision_finish-vision_start)/1e6}
        recorder.write(np.zeros((48, 64, 3), np.uint8), radar, gps, vision, None,
                       stamps["vision_latency_ms"], stamps, camera_frame_id=index,
                       radar_valid=True, gps_valid=True)
        time.sleep(.01)
    checkpoint = json.loads((recorder.session_dir / "session.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "recording" and checkpoint["sample_count"] == 7
    recorder.close()
    recorder.close()
    completed = json.loads((recorder.session_dir / "session.json").read_text(encoding="utf-8"))
    assert completed["status"] == "complete" and completed["sample_count"] == 7
    check = subprocess.run([sys.executable, str(ROOT / "scripts/check_dashboard_recording.py"),
                            str(recorder.session_dir)], check=False)
    dataset = output / "windows.npz"
    build = subprocess.run([sys.executable, str(ROOT / "scripts/build_gt_mrfn_dataset.py"),
                            str(recorder.session_dir), "--output", str(dataset)], check=False)
    assert check.returncode == 0 and build.returncode == 0 and dataset.is_file()
    with np.load(dataset) as data:
        assert data["X"].shape[0] == 3 and data["X"].shape[1] == 5
    print("Dashboard recording mock E2E: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
