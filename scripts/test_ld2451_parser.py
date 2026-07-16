"""LD2451 V1.03 parser regression tests, including the manual example."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sensors.radar_reader import DATA_HEADER, RadarReader, parse_radar_frame


def main() -> None:
    # V1.03 section 1.3 example. The printed PDF contains "IE" for target 2
    # distance; it is evidently 1E (30 m), so the binary regression uses 0x1E.
    frame = bytes.fromhex(
        "F4 F3 F2 F1 11 00 "
        "03 01 "
        "8A 28 00 3C 15 "
        "8A 1E 01 3C 0F "
        "76 5F 00 3C 0F "
        "F8 F7 F6 F5"
    )
    result = parse_radar_frame(frame, approaching_direction_code=0, angle_sign=1)
    assert result is not None
    assert len(result["targets"]) == 3
    first, second, third = result["targets"]
    assert (first["angle_deg"], first["distance_m"]) == (10.0, 40.0)
    assert first["relative_speed_mps"] < 0 and first["is_approaching"]
    assert second["distance_m"] == 30.0 and second["relative_speed_mps"] > 0
    assert (third["angle_deg"], third["distance_m"]) == (-10.0, 95.0)

    empty = bytes.fromhex("F4 F3 F2 F1 00 00 F8 F7 F6 F5")
    empty_result = parse_radar_frame(empty)
    assert empty_result == {"targets": []}

    # Direction mapping must be reversible after the hardware calibration.
    reversed_result = parse_radar_frame(frame, approaching_direction_code=1, angle_sign=1)
    assert reversed_result is not None
    assert reversed_result["targets"][0]["relative_speed_mps"] > 0
    assert reversed_result["targets"][1]["relative_speed_mps"] < 0
    mounted_result = parse_radar_frame(frame, approaching_direction_code=1, angle_sign=-1)
    assert mounted_result is not None
    assert mounted_result["targets"][0]["angle_deg"] == -10.0

    # A corrupt length/tail must not hide the following valid frame.
    reader = RadarReader(mode="real")
    reader._buffer.extend(
        b"noise" + DATA_HEADER + b"\xff\xffbroken" + frame
    )
    recovered = reader._extract_latest_frame()
    assert recovered is not None and len(recovered["targets"]) == 3
    diagnostics = reader.get_diagnostics()
    assert diagnostics["length_error_count"] >= 1
    assert diagnostics["discarded_byte_count"] >= len(b"noise")
    print("LD2451 V1.03 parser regression: PASS")


if __name__ == "__main__":
    main()
