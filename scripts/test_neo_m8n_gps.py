"""Read-only NEO-M8N UART tester and raw NMEA recorder.

It sends no UBX/NMEA configuration commands. Every received line is saved
with host wall/monotonic timestamps, checksum status and parsed GGA/RMC data.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def nmea_checksum(line: str) -> tuple[bool, int | None, int | None]:
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return False, None, None
    body, supplied = line[1:].rsplit("*", 1)
    value = 0
    for ch in body:
        value ^= ord(ch)
    try:
        expected = int(supplied[:2], 16)
    except ValueError:
        return False, value, None
    return value == expected, value, expected


def dm_to_degrees(raw: str, hemisphere: str, degree_digits: int) -> float | None:
    if not raw:
        return None
    try:
        degrees = float(raw[:degree_digits])
        minutes = float(raw[degree_digits:])
    except ValueError:
        return None
    result = degrees + minutes / 60.0
    if hemisphere in ("S", "W"):
        result = -result
    return result


def parse_nmea(line: str) -> dict[str, Any]:
    body = line[1:line.rfind("*")] if line.startswith("$") and "*" in line else line.lstrip("$")
    parts = body.split(",")
    message = parts[0] if parts else ""
    kind = message[-3:] if len(message) >= 3 else message
    talker = message[:-3]
    parsed: dict[str, Any] = {"talker": talker, "type": kind}
    try:
        if kind == "GGA" and len(parts) >= 10:
            parsed.update({
                "utc_time": parts[1],
                "latitude": dm_to_degrees(parts[2], parts[3], 2),
                "longitude": dm_to_degrees(parts[4], parts[5], 3),
                "fix_quality": int(parts[6] or 0),
                "satellites": int(parts[7] or 0),
                "hdop": float(parts[8]) if parts[8] else None,
                "altitude_m": float(parts[9]) if parts[9] else None,
            })
        elif kind == "RMC" and len(parts) >= 10:
            speed_knots = float(parts[7]) if parts[7] else 0.0
            parsed.update({
                "utc_time": parts[1], "status": parts[2],
                "latitude": dm_to_degrees(parts[3], parts[4], 2),
                "longitude": dm_to_degrees(parts[5], parts[6], 3),
                "speed_knots": speed_knots,
                "speed_kmh": speed_knots * 1.852,
                "course_deg": float(parts[8]) if parts[8] else None,
                "utc_date": parts[9],
            })
        elif kind == "VTG" and len(parts) >= 8:
            parsed.update({
                "course_true_deg": float(parts[1]) if parts[1] else None,
                "speed_knots": float(parts[5]) if parts[5] else None,
                "speed_kmh": float(parts[7]) if parts[7] else None,
            })
    except (ValueError, IndexError) as exc:
        parsed["parse_error"] = f"{type(exc).__name__}: {exc}"
    return parsed


def scan_bauds(port: str) -> int:
    import serial
    candidates = (4800, 9600, 19200, 38400, 57600, 115200, 230400)
    best = None
    for baud in candidates:
        try:
            with serial.Serial(port, baud, timeout=0.2) as ser:
                ser.reset_input_buffer()
                deadline = time.monotonic() + 2.2
                raw = bytearray()
                while time.monotonic() < deadline:
                    raw.extend(ser.read(max(1, ser.in_waiting)))
            text = raw.decode("ascii", errors="ignore")
            lines = [x.strip() for x in text.splitlines() if x.strip().startswith("$")]
            valid = sum(nmea_checksum(x)[0] for x in lines)
            print(f"{baud:6d}: bytes={len(raw):5d}, NMEA lines={len(lines):3d}, valid checksum={valid:3d}")
            score = (valid, len(lines), len(raw))
            if best is None or score > best[0]:
                best = (score, baud)
        except serial.SerialException as exc:
            print(f"{baud:6d}: {exc}")
            return 2
    if best and best[0][0] > 0:
        print(f"Detected baud: {best[1]}")
        return 0
    print("No checksum-valid NMEA found; check GPS TX->USB RX, GND, power and antenna.")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only NEO-M8N NMEA tester/recorder")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("logs/neo_m8n_test.jsonl"))
    ap.add_argument("--scan-baud", action="store_true")
    ap.add_argument("--print-all", action="store_true", help="also print non-GGA/RMC sentences")
    args = ap.parse_args()
    if args.scan_baud:
        return scan_bauds(args.port)

    import serial
    output = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    started_mono_ns = time.monotonic_ns()
    counts: dict[str, int] = {}
    checksum_ok = checksum_bad = valid_fix = 0
    arrival_ms: list[float] = []
    previous_ns = None
    raw_bytes = 0
    try:
        with serial.Serial(args.port, args.baud, timeout=0.5) as ser, output.open("w", encoding="utf-8", buffering=1) as f:
            ser.reset_input_buffer()
            print(f"NEO-M8N read-only test: {args.port} @ {args.baud} 8N1 -> {output}")
            print("Place antenna outdoors with a clear sky view. Ctrl+C to stop.")
            deadline = time.monotonic() + args.duration if args.duration > 0 else None
            while deadline is None or time.monotonic() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                received_mono_ns = time.monotonic_ns()
                received_wall_ns = time.time_ns()
                raw_bytes += len(raw)
                line = raw.decode("ascii", errors="replace").strip()
                ok, calculated, supplied = nmea_checksum(line)
                parsed = parse_nmea(line) if line.startswith("$") else {"type": "NON_NMEA"}
                kind = str(parsed.get("type", "UNKNOWN"))
                counts[kind] = counts.get(kind, 0) + 1
                checksum_ok += int(ok)
                checksum_bad += int(not ok)
                if previous_ns is not None:
                    arrival_ms.append((received_mono_ns - previous_ns) / 1e6)
                previous_ns = received_mono_ns
                if (kind == "GGA" and int(parsed.get("fix_quality", 0)) > 0) or (
                    kind == "RMC" and parsed.get("status") == "A"
                ):
                    valid_fix += 1
                row = {
                    "wall_time_ns": received_wall_ns,
                    "monotonic_ns": received_mono_ns,
                    "relative_ms": (received_mono_ns - started_mono_ns) / 1e6,
                    "checksum_ok": ok,
                    "checksum_calculated": calculated,
                    "checksum_supplied": supplied,
                    "raw": line,
                    "parsed": parsed,
                }
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if args.print_all or kind in ("GGA", "RMC"):
                    if kind == "GGA":
                        print(f"GGA ok={ok} fix={parsed.get('fix_quality')} sats={parsed.get('satellites')} "
                              f"lat={parsed.get('latitude')} lon={parsed.get('longitude')}")
                    elif kind == "RMC":
                        print(f"RMC ok={ok} status={parsed.get('status')} speed={parsed.get('speed_kmh', 0):.2f}km/h")
                    else:
                        print(f"{kind} ok={ok}: {line}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print("\nSummary")
    print(f"  raw_bytes={raw_bytes}, sentences={sum(counts.values())}, types={counts}")
    print(f"  checksum_ok={checksum_ok}, checksum_bad={checksum_bad}, valid_fix_sentences={valid_fix}")
    if arrival_ms:
        print(f"  arrival_interval_ms median={statistics.median(arrival_ms):.1f}, max={max(arrival_ms):.1f}")
    if checksum_ok == 0:
        print("  FAIL: no checksum-valid NMEA sentence")
        return 3
    if valid_fix == 0:
        print("  UART PASS, POSITION NOT FIXED: move antenna outdoors and wait for satellites")
        return 4
    print("  PASS: UART, NMEA checksum and GNSS fix are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
