#!/usr/bin/env python3
"""Read and decode WHEELTEC G60 NMEA data on Linux.

This implementation deliberately uses only Python's standard library, so it
works on a fresh Ubuntu installation without pyserial.  The G60 USB bridge is
normally exposed by Linux as /dev/ttyACM* (USB ID 1a86:55d4).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import select
import sys
import termios
import time


G60_USB_IDS = {("1a86", "55d4")}
BAUD_RATES = {
    4800: termios.B4800,
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
}


def _usb_ids(tty: str) -> tuple[str, str] | None:
    """Find USB VID/PID by walking from a tty's sysfs node to its parents."""
    try:
        node = Path("/sys/class/tty", Path(tty).name).resolve()
        for parent in (node, *node.parents):
            vendor = parent / "idVendor"
            product = parent / "idProduct"
            if vendor.exists() and product.exists():
                return vendor.read_text().strip().lower(), product.read_text().strip().lower()
    except OSError:
        pass
    return None


def find_device() -> str:
    candidates = sorted(
        glob.glob("/dev/ttyACM*")
        + glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyCH343USB*")
    )
    matches = [path for path in candidates if _usb_ids(path) in G60_USB_IDS]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "检测到多个 G60 串口：" + ", ".join(matches) + "；请用 --device 指定"
        )
    if len(candidates) == 1:
        return candidates[0]
    detail = ", ".join(candidates) if candidates else "无"
    raise RuntimeError(
        f"未自动找到 USB ID 1a86:55d4 的 G60（现有串口：{detail}）。"
        "请检查数据线，或用 --device /dev/ttyXXX 指定。"
    )


def open_serial(device: str, baud: int) -> int:
    fd = os.open(device, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    attrs[0] = termios.IGNBRK
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
    attrs[3] = 0
    attrs[4] = BAUD_RATES[baud]
    attrs[5] = BAUD_RATES[baud]
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIFLUSH)
    return fd


def valid_nmea(line: str) -> bool:
    if not line.startswith("$") or "*" not in line:
        return False
    body, supplied = line[1:].rsplit("*", 1)
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    try:
        return checksum == int(supplied[:2], 16)
    except ValueError:
        return False


def coordinate(value: str, hemisphere: str) -> float | None:
    if not value or hemisphere not in {"N", "S", "E", "W"}:
        return None
    try:
        raw = float(value)
    except ValueError:
        return None
    degrees = int(raw // 100)
    result = degrees + (raw - degrees * 100) / 60
    return -result if hemisphere in {"S", "W"} else result


def parse_sentence(line: str, state: dict[str, object]) -> bool:
    """Update state; return True when a GGA sentence was processed."""
    fields = line[1 : line.index("*")].split(",")
    kind = fields[0][-3:]
    if kind == "GGA" and len(fields) >= 10:
        try:
            quality = int(fields[6] or 0)
            satellites = int(fields[7] or 0)
        except ValueError:
            quality, satellites = 0, 0
        state.update(
            utc=fields[1] or None,
            latitude=coordinate(fields[2], fields[3]),
            longitude=coordinate(fields[4], fields[5]),
            fix_quality=quality,
            satellites=satellites,
            hdop=float(fields[8]) if fields[8] else None,
            altitude_m=float(fields[9]) if fields[9] else None,
        )
        return True
    if kind == "RMC" and len(fields) >= 9:
        state["rmc_valid"] = fields[2] == "A"
        state["speed_knots"] = float(fields[7]) if fields[7] else None
        state["course_deg"] = float(fields[8]) if fields[8] else None
    elif kind == "TXT" and len(fields) >= 5:
        state["receiver_message"] = fields[4]
    return False


def print_state(state: dict[str, object], as_json: bool) -> None:
    fixed = bool(state.get("fix_quality")) and state.get("latitude") is not None
    output = {"fixed": fixed, **state}
    if as_json:
        print(json.dumps(output, ensure_ascii=False), flush=True)
        return
    if not fixed:
        antenna = state.get("receiver_message", "等待接收机状态")
        print(
            f"等待定位 | 卫星 {state.get('satellites', 0)} | {antenna}",
            flush=True,
        )
        return
    print(
        "定位成功 | "
        f"纬度 {state['latitude']:.8f} | 经度 {state['longitude']:.8f} | "
        f"海拔 {state.get('altitude_m')} m | 卫星 {state.get('satellites')} | "
        f"HDOP {state.get('hdop')}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="读取并解析 WHEELTEC G60 GPS 数据")
    parser.add_argument("--device", help="串口设备；默认按 USB ID 自动识别")
    parser.add_argument("--baud", type=int, choices=BAUD_RATES, default=9600)
    parser.add_argument("--raw", action="store_true", help="同时打印原始 NMEA 帧")
    parser.add_argument("--json", action="store_true", help="以 JSON Lines 输出状态")
    parser.add_argument("--once", action="store_true", help="首次成功定位后退出")
    parser.add_argument("--timeout", type=float, default=0, help="超时秒数，0 表示不超时")
    args = parser.parse_args()

    try:
        device = args.device or find_device()
        fd = open_serial(device, args.baud)
    except (OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(f"G60 串口已打开：{device}，{args.baud} bps", file=sys.stderr, flush=True)
    state: dict[str, object] = {}
    buffer = b""
    started = time.monotonic()
    try:
        while not args.timeout or time.monotonic() - started < args.timeout:
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                continue
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                line = raw_line.strip().decode("ascii", errors="replace")
                if not valid_nmea(line):
                    continue
                if args.raw:
                    print(line, flush=True)
                if parse_sentence(line, state):
                    print_state(state, args.json)
                    if args.once and state.get("fix_quality"):
                        return 0
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        print(f"串口读取失败：{exc}", file=sys.stderr)
        return 2
    finally:
        os.close(fd)

    print(f"在 {args.timeout:g} 秒内未获得有效定位", file=sys.stderr)
    return 1 if args.once else 0


if __name__ == "__main__":
    raise SystemExit(main())
