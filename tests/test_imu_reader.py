from __future__ import annotations

import time
import unittest

from src.sensors.imu_reader import IMUReader


def _frame(packet_type: int, values: tuple[int, int, int, int]) -> bytes:
    frame = bytearray((0x55, packet_type))
    for value in values:
        value &= 0xFFFF
        frame.extend((value & 0xFF, value >> 8))
    frame.append(sum(frame) & 0xFF)
    return bytes(frame)


class _FakeSerial:
    is_open = True

    def __init__(self) -> None:
        self.data = b""

    @property
    def in_waiting(self) -> int:
        return len(self.data)

    def read(self, size: int) -> bytes:
        result, self.data = self.data[:size], self.data[size:]
        return result


class IMUReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = IMUReader(mode="real", config={"max_data_age_sec": 0.05})
        self.serial = _FakeSerial()
        self.reader._serial = self.serial
        self.acc = _frame(0x51, (0, 0, 2048, 0))
        self.gyro = _frame(0x52, (0, 16384, 0, 0))
        self.angle = _frame(0x53, (16384, -8192, 8192, 0))

    def test_real_mode_without_serial_is_invalid(self) -> None:
        reader = IMUReader(mode="real")
        data = reader.read_once()
        self.assertFalse(data.valid)
        self.assertEqual(data.acc_z, 0.0)

    def test_components_are_combined_across_reads(self) -> None:
        for packet in (self.acc, self.gyro):
            self.serial.data = packet
            self.assertFalse(self.reader.read_once().valid)

        self.serial.data = self.angle
        data = self.reader.read_once()
        self.assertTrue(data.valid)
        self.assertAlmostEqual(data.acc_z, 9.8, places=2)
        self.assertAlmostEqual(data.gyro_y, 1000.0, places=2)
        self.assertAlmostEqual(data.roll, 90.0, places=2)
        self.assertAlmostEqual(data.pitch, -45.0, places=2)

    def test_installation_offsets_drive_body_angles_and_tilt_score(self) -> None:
        reader = IMUReader(mode="real", config={
            "max_data_age_sec": 0.05,
            "roll_offset_deg": 90.0,
            "pitch_offset_deg": -45.0,
        })
        reader._serial = self.serial
        self.serial.data = self.acc + self.gyro + self.angle
        data = reader.read_once()
        self.assertTrue(data.valid)
        self.assertAlmostEqual(data.body_roll, 0.0, places=2)
        self.assertAlmostEqual(data.body_pitch, 0.0, places=2)
        self.assertEqual(data.tilt_score, 0.0)

    def test_recent_complete_sample_remains_valid_without_new_bytes(self) -> None:
        self.serial.data = self.acc + self.gyro + self.angle
        first = self.reader.read_once()
        second = self.reader.read_once()
        self.assertTrue(first.valid)
        self.assertTrue(second.valid)
        self.assertEqual(first.bump_score, second.bump_score)

    def test_sample_becomes_invalid_after_timeout(self) -> None:
        self.serial.data = self.acc + self.gyro + self.angle
        self.assertTrue(self.reader.read_once().valid)
        time.sleep(0.06)
        self.assertFalse(self.reader.read_once().valid)

    def test_bad_checksum_is_rejected_and_following_frame_parses(self) -> None:
        bad = bytearray(self.acc)
        bad[-1] ^= 0xFF
        self.reader._buffer.extend(bad + self.angle)
        parsed, _ = self.reader._parse_next_packet()
        self.assertEqual(parsed["type"], "angle")
        self.assertAlmostEqual(parsed["roll"], 90.0)
        diagnostics = self.reader.get_diagnostics()
        self.assertEqual(diagnostics["bad_checksum_count"], 1)
        self.assertEqual(diagnostics["packet_counts"]["angle"], 1)

    def test_bad_checksum_is_not_counted_again_on_next_read(self) -> None:
        bad = bytearray(self.acc)
        bad[-1] ^= 0xFF
        self.reader._buffer.extend(bad)
        self.assertEqual(self.reader._parse_next_packet(), (None, 0))
        self.assertEqual(self.reader._parse_next_packet(), (None, 0))
        self.assertEqual(self.reader.get_diagnostics()["bad_checksum_count"], 1)

    def test_diagnostics_report_component_age_and_skew(self) -> None:
        self.serial.data = self.acc + self.gyro + self.angle
        self.assertTrue(self.reader.read_once().valid)
        diagnostics = self.reader.get_diagnostics()
        self.assertTrue(diagnostics["connected"])
        self.assertEqual(
            diagnostics["packet_counts"],
            {"acc": 1, "gyro": 1, "angle": 1},
        )
        self.assertEqual(
            set(diagnostics["component_arrival_monotonic_ns"]),
            {"acc", "gyro", "angle"},
        )
        self.assertIsNotNone(diagnostics["component_skew_ms"])
        self.assertLess(diagnostics["component_skew_ms"], 200.0)


if __name__ == "__main__":
    unittest.main()
