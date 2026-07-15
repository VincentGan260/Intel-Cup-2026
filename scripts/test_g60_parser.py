"""WHEELTEC G60 NMEA parser regression tests."""
from __future__ import annotations

from test_g60_gps import nmea_checksum, parse_nmea


def main() -> None:
    gga = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    assert nmea_checksum(gga)[0]
    p = parse_nmea(gga)
    assert p["type"] == "GGA" and p["fix_quality"] == 1 and p["satellites"] == 8
    assert abs(p["latitude"] - 48.1173) < 1e-6
    assert abs(p["longitude"] - 11.5166667) < 1e-6

    rmc = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
    assert nmea_checksum(rmc)[0]
    p = parse_nmea(rmc)
    assert p["type"] == "RMC" and p["status"] == "A"
    assert abs(p["speed_kmh"] - 41.4848) < 1e-4
    assert not nmea_checksum(gga[:-2] + "00")[0]
    gn_gga = "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*59"
    # G60默认使用GN talker；校验talker无关的GGA解析。
    body = gn_gga[1:gn_gga.index("*")]
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    gn_gga = f"${body}*{checksum:02X}"
    assert nmea_checksum(gn_gga)[0]
    assert parse_nmea(gn_gga)["type"] == "GGA"
    print("WHEELTEC G60 NMEA parser regression: PASS")


if __name__ == "__main__":
    main()
