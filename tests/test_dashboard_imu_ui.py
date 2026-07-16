from __future__ import annotations

import unittest
from pathlib import Path

from src.dashboard.mock_state import build_mock_state


class DashboardIMUUITests(unittest.TestCase):
    def test_mock_state_exposes_complete_imu_payload(self) -> None:
        imu = build_mock_state(camera_available=False)["imu_data"]
        expected = {
            "valid", "roll", "pitch", "yaw",
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "brake_score", "bump_score", "tilt_score",
        }
        self.assertTrue(expected.issubset(imu))

    def test_right_panel_contains_imu_card_and_render_fields(self) -> None:
        html = (Path(__file__).resolve().parents[1]
                / "src" / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="imu-data-card"', html)
        self.assertIn('id="imu-data-container"', html)
        for field in ("risk", "attitude", "acceleration", "angular_velocity", "event_scores"):
            self.assertIn(f'"{field}"', html)


if __name__ == "__main__":
    unittest.main()
