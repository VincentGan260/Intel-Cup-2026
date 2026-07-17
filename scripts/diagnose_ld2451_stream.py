"""Read-only LD2451 serial stream diagnostic.

The tool never sends commands to the radar. It reports whether bytes, V1.03
headers, tails and complete frames are present so a baud/protocol mismatch can
be distinguished from a parser defect.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sensors.radar_reader import DATA_END, DATA_HEADER, RadarReader


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only LD2451 stream diagnostic")
    parser.add_argument("--port", default="/dev/ttyRadarLD2451")
    parser.add_argument("--baudrate", type=int, default=256000)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    import serial

    raw = bytearray()
    reader = RadarReader(mode="real", config={
        "port": args.port,
        "baudrate": args.baudrate,
    })
    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.1) as stream:
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                chunk = stream.read(max(1, stream.in_waiting))
                if chunk:
                    raw.extend(chunk)
                    reader._buffer.extend(chunk)
                    while reader._extract_frame() is not None:
                        reader.valid_frame_count += 1
    except Exception as exc:
        print(json.dumps({
            "port": args.port,
            "baudrate": args.baudrate,
            "status": "port_error",
            "error": str(exc),
        }, ensure_ascii=False, indent=2))
        return 2

    diagnostics = reader.get_diagnostics()
    header_count = raw.count(DATA_HEADER)
    tail_count = raw.count(DATA_END)
    if not raw:
        status = "no_bytes"
    elif reader.valid_frame_count:
        status = "v103_frames_decoded"
    elif not header_count:
        status = "bytes_without_v103_header"
    else:
        status = "v103_header_but_no_complete_frame"
    print(json.dumps({
        "port": args.port,
        "baudrate": args.baudrate,
        "duration_s": args.seconds,
        "status": status,
        "bytes_received": len(raw),
        "v103_header_count": header_count,
        "v103_tail_count": tail_count,
        "valid_frame_count": reader.valid_frame_count,
        "invalid_frame_count": diagnostics["invalid_frame_count"],
        "length_error_count": diagnostics["length_error_count"],
        "tail_error_count": diagnostics["tail_error_count"],
        "first_64_bytes_hex": bytes(raw[:64]).hex(" "),
        "last_64_bytes_hex": bytes(raw[-64:]).hex(" "),
    }, ensure_ascii=False, indent=2))
    return 0 if reader.valid_frame_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
