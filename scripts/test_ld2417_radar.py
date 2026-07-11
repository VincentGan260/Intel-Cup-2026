"""HLK-LD2417 read-only serial connection and target-frame tester.

Protocol (V1.0.0): TTL UART, 115200 baud, 8N1, little-endian.
Report frame: AA AA | target_count (1B) | target_count * 10B | 55 55.

The V1.0.0 table is ambiguous about target-state width. Hardware captures
show it is uint32 (4 bytes), making each target record 10 bytes.

This tool deliberately sends no configuration commands.  It is safe for an
initial wiring/protocol test and can optionally save every parsed frame as
JSONL for later inspection.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEADER = b"\xAA\xAA"
TAIL = b"\x55\x55"
TARGET_BYTES = 10
MAX_TARGETS = 64


def parse_target(raw: bytes) -> dict[str, Any]:
    """Parse one 10-byte LD2417 target record."""
    if len(raw) != TARGET_BYTES:
        raise ValueError(f"target must be {TARGET_BYTES} bytes")
    target_id = raw[0]
    direction_code = raw[1]
    distance_raw, speed_raw, state_raw = struct.unpack_from("<HHI", raw, 2)
    direction = {1: "left", 2: "right"}.get(direction_code, "unknown")
    return {
        "target_id": target_id,
        "direction_code": direction_code,
        "direction": direction,
        "y_distance_m": distance_raw / 100.0,
        "y_speed_kmh": speed_raw / 100.0,
        "y_speed_mps": speed_raw / 360.0,
        "is_high_speed": bool(state_raw & 0x0001),
        "associated": bool(state_raw & 0x0002),
        "track_duration_frames": (state_raw >> 8) & 0xFF,
        "state_raw": state_raw,
    }


def parse_report_frame(frame: bytes) -> dict[str, Any]:
    if len(frame) < 5 or not frame.startswith(HEADER) or not frame.endswith(TAIL):
        raise ValueError("invalid LD2417 report frame boundary")
    count = frame[2]
    expected = 2 + 1 + count * TARGET_BYTES + 2
    if len(frame) != expected:
        raise ValueError(f"invalid frame size: got {len(frame)}, expected {expected}")
    targets = [
        parse_target(frame[3 + i * TARGET_BYTES: 3 + (i + 1) * TARGET_BYTES])
        for i in range(count)
    ]
    return {"target_count": count, "targets": targets}


class FrameDecoder:
    """Incremental decoder resilient to partial reads and leading garbage."""
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.bad_frames = 0

    def feed(self, chunk: bytes) -> list[tuple[bytes, dict[str, Any]]]:
        self.buffer.extend(chunk)
        decoded: list[tuple[bytes, dict[str, Any]]] = []
        while True:
            pos = self.buffer.find(HEADER)
            if pos < 0:
                keep = 1 if self.buffer.endswith(b"\xAA") else 0
                self.discarded_bytes += max(0, len(self.buffer) - keep)
                if keep:
                    self.buffer[:] = self.buffer[-1:]
                else:
                    self.buffer.clear()
                break
            if pos:
                self.discarded_bytes += pos
                del self.buffer[:pos]
            if len(self.buffer) < 3:
                break
            count = self.buffer[2]
            if count > MAX_TARGETS:
                self.bad_frames += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            size = 2 + 1 + count * TARGET_BYTES + 2
            if len(self.buffer) < size:
                break
            candidate = bytes(self.buffer[:size])
            if candidate[-2:] != TAIL:
                self.bad_frames += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            try:
                report = parse_report_frame(candidate)
            except ValueError:
                self.bad_frames += 1
                del self.buffer[0]
                continue
            del self.buffer[:size]
            decoded.append((candidate, report))
        if len(self.buffer) > 65536:
            self.discarded_bytes += len(self.buffer)
            self.buffer.clear()
        return decoded


def list_ports() -> int:
    try:
        from serial.tools import list_ports as serial_list_ports
    except ImportError:
        print("pyserial is not installed. Run: python -m pip install pyserial", file=sys.stderr)
        return 2
    ports = list(serial_list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return 1
    for p in ports:
        print(f"{p.device:12} {p.description}  hwid={p.hwid}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only HLK-LD2417 UART tester")
    ap.add_argument("--port", help="Windows COM port or Linux /dev/ttyUSBx")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=float, default=30.0, help="0 means Ctrl+C")
    ap.add_argument("--timeout", type=float, default=0.1)
    ap.add_argument("--out", type=Path, help="optional parsed-frame JSONL output")
    ap.add_argument("--raw-bin", type=Path, help="save every received UART byte for protocol diagnosis")
    ap.add_argument("--raw", action="store_true", help="print raw frame hex")
    ap.add_argument("--list-ports", action="store_true")
    args = ap.parse_args()
    if args.list_ports:
        return list_ports()
    if not args.port:
        ap.error("--port is required unless --list-ports is used")

    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial is not installed. Run: python -m pip install pyserial", file=sys.stderr)
        return 2
    out_file = None
    raw_file = None
    if args.out:
        out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8", buffering=1)
    if args.raw_bin:
        raw_path = args.raw_bin if args.raw_bin.is_absolute() else PROJECT_ROOT / args.raw_bin
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_file = raw_path.open("wb")

    decoder = FrameDecoder()
    frame_count = target_frames = byte_count = 0
    started = time.monotonic()
    last_data = started
    try:
        with serial.Serial(args.port, args.baud, timeout=args.timeout,
                           bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                           stopbits=serial.STOPBITS_ONE) as ser:
            ser.reset_input_buffer()
            print(f"LD2417 read-only test: {args.port} @ {args.baud} 8N1")
            print("Move a target in front of the radar. Press Ctrl+C to stop.")
            while args.duration <= 0 or time.monotonic() - started < args.duration:
                chunk = ser.read(max(1, ser.in_waiting))
                if not chunk:
                    if time.monotonic() - last_data > 3.0:
                        print("[WARN] no serial bytes for 3 s; check power, GND, TX/RX and port")
                        last_data = time.monotonic()
                    continue
                byte_count += len(chunk)
                if raw_file:
                    raw_file.write(chunk)
                    raw_file.flush()
                last_data = time.monotonic()
                for raw, report in decoder.feed(chunk):
                    frame_count += 1
                    if report["target_count"]:
                        target_frames += 1
                    now_ns = time.time_ns()
                    row = {"frame_id": frame_count - 1, "wall_time_ns": now_ns,
                           "relative_ms": round((time.monotonic() - started) * 1000, 3),
                           **report}
                    if args.raw:
                        row["raw_hex"] = raw.hex(" ")
                    print(f"#{frame_count:05d} targets={report['target_count']}", end="")
                    for t in report["targets"]:
                        print(f" | id={t['target_id']} {t['direction']} "
                              f"d={t['y_distance_m']:.2f}m v={t['y_speed_kmh']:.2f}km/h "
                              f"high={int(t['is_high_speed'])} assoc={int(t['associated'])} "
                              f"age={t['track_duration_frames']}", end="")
                    print()
                    if args.raw:
                        print("  RAW:", row["raw_hex"])
                    if out_file:
                        out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"[ERROR] serial connection failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if out_file:
            out_file.close()
        if raw_file:
            raw_file.close()

    elapsed = max(time.monotonic() - started, 1e-9)
    print("\nSummary")
    print(f"  bytes={byte_count}, valid_frames={frame_count}, frames_with_target={target_frames}")
    print(f"  frame_rate={frame_count / elapsed:.2f} Hz, discarded_bytes={decoder.discarded_bytes}, bad_frames={decoder.bad_frames}")
    if byte_count == 0:
        print("  FAIL: no UART data received")
        return 3
    if frame_count == 0:
        print("  FAIL: bytes received but no valid AA AA ... 55 55 report frame decoded")
        return 4
    print("  PASS: LD2417 report frames decoded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
