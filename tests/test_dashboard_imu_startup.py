from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import run_dashboard
import yaml
from src.sensors.imu_reader import IMUReader


class DashboardIMUStartupTests(unittest.TestCase):
    def test_systemd_service_enables_imu(self) -> None:
        service = (
            Path(__file__).resolve().parents[1]
            / "deploy" / "edge" / "rider-dashboard.service"
        ).read_text(encoding="utf-8")
        exec_start = next(
            line for line in service.splitlines() if line.startswith("ExecStart=")
        )
        self.assertIn("--enable-imu", exec_start.split())

    def test_dk2500_serial_sensors_use_distinct_verified_usb_paths(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "sensor_ports.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        radar_port = config["dk2500"]["radar"]["port"]
        imu_port = config["dk2500"]["imu"]["port"]
        self.assertEqual(
            imu_port,
            "/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.4:1.0-port0",
        )
        self.assertEqual(
            radar_port,
            "/dev/serial/by-path/pci-0000:00:14.0-usb-0:6.3:1.0-port0",
        )
        self.assertNotEqual(imu_port, radar_port)

    @patch("src.sensors.imu_reader.IMUReader")
    def test_start_keeps_reader_when_serial_is_late(self, reader_cls) -> None:
        reader = MagicMock()
        reader._serial = None
        reader_cls.return_value = reader

        result = run_dashboard._start_imu_reader(
            "dk2500",
            {"imu": {"port": "/dev/serial/by-path/late", "baudrate": 115200}},
            {"roll_offset_deg": 1.0, "pitch_offset_deg": 2.0},
        )

        self.assertIs(result, reader)
        reader.start.assert_called_once_with()

    def test_first_delayed_connection_gets_startup_calibration(self) -> None:
        reader = IMUReader(mode="real")
        fake_serial = MagicMock()
        fake_serial.is_open = True
        with patch.object(
            reader, "_connect_serial", side_effect=lambda: setattr(
                reader, "_serial", fake_serial
            ) or True
        ), patch.object(reader, "_run_calibration") as calibrate, patch.object(
            reader, "_read_real", return_value=MagicMock()
        ):
            reader.read_once()
            reader._serial = None
            reader.read_once()

        calibrate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
