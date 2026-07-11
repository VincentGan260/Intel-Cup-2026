"""Inspect LD2417 raw UART capture without assuming a target record size."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    args = ap.parse_args()
    data = args.raw.read_bytes()
    starts = []
    pos = 0
    while True:
        pos = data.find(b"\xaa\xaa", pos)
        if pos < 0:
            break
        starts.append(pos)
        pos += 2

    spans = [b - a for a, b in zip(starts, starts[1:])]
    print(f"bytes={len(data)}, AA-AA headers={len(starts)}")
    print("header-to-header span counts:", Counter(spans).most_common(20))
    for i, start in enumerate(starts[:30]):
        end = starts[i + 1] if i + 1 < len(starts) else min(len(data), start + 80)
        packet = data[start:end]
        print(f"#{i:03d} offset={start:06d} span={len(packet):3d} count_byte={packet[2] if len(packet)>2 else None}: {packet[:80].hex(' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
