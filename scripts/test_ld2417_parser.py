"""Dependency-free parser smoke tests for test_ld2417_radar.py."""
from __future__ import annotations

import struct

from test_ld2417_radar import FrameDecoder, parse_report_frame


def make_target(target_id: int, direction: int, distance_cm: int,
                speed_001_kmh: int, state: int) -> bytes:
    return bytes((target_id, direction)) + struct.pack("<HHI", distance_cm, speed_001_kmh, state)


def main() -> None:
    t1 = make_target(3, 1, 1234, 567, 0x2A03)
    frame = b"\xAA\xAA\x01" + t1 + b"\x55\x55"
    result = parse_report_frame(frame)
    assert result["target_count"] == 1
    target = result["targets"][0]
    assert target["target_id"] == 3
    assert target["direction"] == "left"
    assert target["y_distance_m"] == 12.34
    assert target["y_speed_kmh"] == 5.67
    assert target["is_high_speed"] is True
    assert target["associated"] is True
    assert target["track_duration_frames"] == 42

    decoder = FrameDecoder()
    assert decoder.feed(b"garbage" + frame[:5]) == []
    decoded = decoder.feed(frame[5:] + b"\xAA\xAA\x00\x55\x55")
    assert len(decoded) == 2
    assert decoded[0][1]["target_count"] == 1
    assert decoded[1][1]["target_count"] == 0
    print("LD2417 parser smoke tests: PASS")


if __name__ == "__main__":
    main()
