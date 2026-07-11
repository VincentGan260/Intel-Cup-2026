"""Record camera, radar, IMU and GPS into one timestamped session.

Each stream keeps its own acquisition timestamp.  Use monotonic_ns for later
alignment; wall_time_ns is retained only for human-readable/event time.

Examples:
  python scripts/record_multimodal.py --mode mock --duration 10 --scene desk_test
  python scripts/record_multimodal.py --mode real --profile windows --scene low_clear
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sensors.gps_reader import GPSReader
from src.sensors.imu_reader import IMUReader
from src.sensors.radar_reader import RadarReader


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._file = path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _stamp(session_mono_ns: int, mono_ns: int | None = None) -> dict[str, int | float]:
    """Create a host-clock stamp. Alignment always uses monotonic_ns."""
    mono_ns = mono_ns if mono_ns is not None else time.monotonic_ns()
    return {
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": mono_ns,
        "relative_ms": (mono_ns - session_mono_ns) / 1_000_000.0,
    }


def _sensor_worker(
    name: str,
    reader: Any,
    writer: JsonlWriter,
    session_mono_ns: int,
    stop: threading.Event,
    hz: float,
    counts: dict[str, int],
) -> None:
    sequence = 0
    period = 1.0 / max(hz, 0.1)
    try:
        reader.start()
        while not stop.is_set():
            loop_start = time.monotonic()
            read_start_ns = time.monotonic_ns()
            error = None
            try:
                sample = reader.read_once()
                payload = _jsonable(sample)
            except Exception as exc:  # keep other streams recording
                payload = {"valid": False, "source": name}
                error = f"{type(exc).__name__}: {exc}"
            read_end_ns = time.monotonic_ns()
            # 串口报文缺少硬件时间戳时，以read完成时刻作为保守的到达时间。
            row = {
                "sample_id": sequence,
                **_stamp(session_mono_ns, read_end_ns),
                "read_start_monotonic_ns": read_start_ns,
                "read_end_monotonic_ns": read_end_ns,
                "read_duration_ms": (read_end_ns - read_start_ns) / 1_000_000.0,
                "data": payload,
            }
            if error:
                row["error"] = error
            writer.write(row)
            sequence += 1
            counts[name] = sequence
            stop.wait(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        try:
            reader.stop()
        except Exception:
            pass


def _camera_worker(
    mode: str,
    camera_id: int,
    width: int,
    height: int,
    fps: float,
    frames_dir: Path,
    writer: JsonlWriter,
    session_mono_ns: int,
    stop: threading.Event,
    counts: dict[str, int],
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        writer.write({"frame_id": 0, **_stamp(session_mono_ns), "valid": False,
                      "error": f"camera dependency missing: {exc}"})
        stop.set()
        return

    cap = None
    if mode == "real":
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            writer.write({"frame_id": 0, **_stamp(session_mono_ns), "valid": False,
                          "error": f"camera {camera_id} open failed"})
            stop.set()
            return

    frame_id = 0
    period = 1.0 / max(fps, 0.1)
    try:
        while not stop.is_set():
            loop_start = time.monotonic()
            read_start_ns = time.monotonic_ns()
            if cap is not None:
                ok, frame = cap.read()
            else:
                ok = True
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, f"MOCK frame {frame_id}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)

            read_end_ns = time.monotonic_ns()
            stamp = _stamp(session_mono_ns, read_end_ns)

            row: dict[str, Any] = {
                "frame_id": frame_id,
                **stamp,
                "read_start_monotonic_ns": read_start_ns,
                "read_end_monotonic_ns": read_end_ns,
                "read_duration_ms": (read_end_ns - read_start_ns) / 1_000_000.0,
                "valid": bool(ok),
            }
            if ok and frame is not None:
                filename = f"frame_{frame_id:06d}_{stamp['wall_time_ns']}.jpg"
                path = frames_dir / filename
                saved = bool(cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]))
                row.update({"valid": saved, "path": f"frames/{filename}",
                            "width": int(frame.shape[1]), "height": int(frame.shape[0])})
                if not saved:
                    row["error"] = "cv2.imwrite failed"
            else:
                row["error"] = "camera read failed"
            writer.write(row)
            frame_id += 1
            counts["camera"] = frame_id
            stop.wait(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        if cap is not None:
            cap.release()


def _load_ports(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = PROJECT_ROOT / "configs" / "sensor_ports.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get(profile, {}), cfg.get("camera", {})


def _validate_real_ports(ports: dict[str, Any]) -> None:
    """Fail before recording if two serial sensors are configured to one port."""
    owners: dict[str, str] = {}
    for name in ("gps", "imu", "radar"):
        port = str(ports.get(name, {}).get("port", "")).strip()
        if not port:
            raise ValueError(f"{name} serial port is empty")
        key = port.lower()
        if key in owners:
            raise ValueError(f"duplicate serial port {port}: {owners[key]} and {name}")
        owners[key] = name


def main() -> int:
    ap = argparse.ArgumentParser(description="RiderGuardian multi-modal recorder")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--profile", choices=["windows", "dk2500"], default="windows")
    ap.add_argument("--scene", required=True, help="e.g. low_clear/high_crossing")
    ap.add_argument("--session-name", default="")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means Ctrl+C")
    ap.add_argument("--output", default="data/recordings")
    ap.add_argument("--camera-id", type=int, default=None)
    ap.add_argument("--camera-fps", type=float, default=None)
    ap.add_argument("--sensor-hz", type=float, default=20.0)
    args = ap.parse_args()

    ports, camera_cfg = _load_ports(args.profile)
    if args.mode == "real":
        try:
            _validate_real_ports(ports)
        except ValueError as exc:
            print(f"[PRE-FLIGHT FAIL] {exc}", file=sys.stderr)
            print("Fix configs/sensor_ports.yaml before collecting data.", file=sys.stderr)
            return 2
    started_wall_ns = time.time_ns()
    started_mono_ns = time.monotonic_ns()
    label = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.session_name}" if args.session_name else ""
    session_dir = (PROJECT_ROOT / args.output / f"{label}_{args.scene}{suffix}").resolve()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "schema_version": 2,
        "scene": args.scene,
        "session_name": args.session_name,
        "mode": args.mode,
        "profile": args.profile,
        "started_wall_time_ns": started_wall_ns,
        "started_monotonic_ns": started_mono_ns,
        "host": platform.node(),
        "camera": camera_cfg,
        "ports": ports,
        "alignment_clock": "monotonic_ns",
        "timestamp_semantics": {
            "monotonic_ns": "host monotonic clock at read completion; alignment key",
            "wall_time_ns": "host realtime clock paired at read completion; human/event time",
            "read_start_monotonic_ns": "host time immediately before blocking read",
            "read_end_monotonic_ns": "host time immediately after blocking read",
        },
    }
    (session_dir / "session.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    writers = {name: JsonlWriter(session_dir / f"{name}.jsonl")
               for name in ("frames", "radar", "imu", "gps")}
    stop = threading.Event()
    counts = {"camera": 0, "radar": 0, "imu": 0, "gps": 0}
    readers = {
        "radar": RadarReader(args.mode, ports.get("radar", {})),
        "imu": IMUReader(args.mode, ports.get("imu", {})),
        "gps": GPSReader(args.mode, ports.get("gps", {})),
    }
    threads = [
        threading.Thread(target=_sensor_worker,
                         args=(name, reader, writers[name], started_mono_ns, stop,
                               args.sensor_hz, counts), daemon=True, name=f"rec-{name}")
        for name, reader in readers.items()
    ]
    threads.append(threading.Thread(
        target=_camera_worker,
        args=(args.mode, args.camera_id if args.camera_id is not None else int(camera_cfg.get("device_id", 0)),
              int(camera_cfg.get("width", 640)), int(camera_cfg.get("height", 480)),
              args.camera_fps or float(camera_cfg.get("fps", 30)), frames_dir, writers["frames"],
              started_mono_ns, stop, counts), daemon=True, name="rec-camera"))

    def request_stop(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    print(f"Recording to: {session_dir}")
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    while not stop.wait(0.2):
        if deadline is not None and time.monotonic() >= deadline:
            stop.set()
    for thread in threads:
        thread.join(timeout=3.0)
    for writer in writers.values():
        writer.close()

    metadata["ended_wall_time_ns"] = time.time_ns()
    metadata["counts"] = counts
    (session_dir / "session.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done: {counts}")
    return 0 if counts["camera"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
