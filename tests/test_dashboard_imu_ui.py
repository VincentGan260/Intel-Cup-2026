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

    def test_imu_summary_is_merged_into_realtime_sensor_card(self) -> None:
        html = (Path(__file__).resolve().parents[1]
                / "src" / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('id="imu-data-card"', html)
        self.assertNotIn('id="imu-data-container"', html)
        self.assertIn('id="modality-risk-card"', html)
        for source in ("radar", "vision", "imu"):
            self.assertIn(f'id="{source}-risk-fill"', html)
            self.assertIn(f'id="{source}-risk-value"', html)
        self.assertIn('"imu_lateral"', html)

    def test_video_mode_compares_annotated_and_clean_raw_streams(self) -> None:
        html = (Path(__file__).resolve().parents[1]
                / "src" / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('btn.textContent="原始画面"', html)
        self.assertNotIn('原始画面 + 前端叠加', html)
        raw_branch = html.split('btn.textContent="原始画面"', 1)[1].split("} else {", 1)[0]
        self.assertIn("overlayEnabled = false", raw_branch)
        self.assertIn('setVideoSource("/video_feed")', raw_branch)


if __name__ == "__main__":
    unittest.main()
