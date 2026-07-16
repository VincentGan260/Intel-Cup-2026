from __future__ import annotations

import unittest

from src.dashboard.risk_score_variation import (
    RiskScoreVariation,
    RiskScoreVariationConfig,
)
from src.dashboard.state_store import DashboardStateStore


def warning_state() -> dict:
    return {
        "timestamp": 1.0,
        "risk_score": 0.61,
        "risk_level": 1,
        "risk_score_state": "current",
        "radar_score": 0.52,
        "radar_level": 1,
        "vision_score": 0.61,
        "vision_level": 1,
        "imu_score": 0.12,
        "imu_level": 0,
        "imu_data": {"risk_score": 0.12},
        "risk_rule": {"imu_score": 0.12},
        "risk_items": {"dist": 0.52, "obs": 0.61, "pose": 0.12, "speed": 0.0},
    }


class RiskScoreVariationTests(unittest.TestCase):
    def make_variation(self) -> RiskScoreVariation:
        return RiskScoreVariation(RiskScoreVariationConfig(
            max_amplitude=0.012, time_constant_s=1.0, seed=7))

    def test_constant_input_varies_smoothly_and_preserves_level_bands(self) -> None:
        variation = self.make_variation()
        outputs = [variation.apply(warning_state(), now_monotonic=index * 0.05)
                   for index in range(200)]

        self.assertGreater(len({round(row["vision_score"], 2) for row in outputs}), 1)
        for row in outputs:
            self.assertGreaterEqual(row["radar_score"], 0.35)
            self.assertLess(row["radar_score"], 0.70)
            self.assertGreaterEqual(row["vision_score"], 0.35)
            self.assertLess(row["vision_score"], 0.70)
            self.assertGreaterEqual(row["imu_score"], 0.0)
            self.assertLess(row["imu_score"], 0.35)
            self.assertEqual(row["risk_score"], max(
                row["radar_score"], row["vision_score"], row["imu_score"]))
            self.assertEqual(row["risk_level"], 1)
            self.assertEqual(row["imu_data"]["risk_score"], row["imu_score"])
            self.assertEqual(row["risk_rule"]["imu_score"], row["imu_score"])
            self.assertEqual(row["risk_items"]["dist"], row["radar_score"])
            self.assertEqual(row["risk_items"]["obs"], row["vision_score"])
            self.assertEqual(row["risk_items"]["pose"], row["imu_score"])

    def test_variation_is_bounded_and_does_not_mutate_decision_state(self) -> None:
        state = warning_state()
        original = warning_state()
        published = self.make_variation().apply(state, now_monotonic=1.0)

        self.assertEqual(state, original)
        for key in ("risk_score", "radar_score", "vision_score", "imu_score"):
            self.assertLessEqual(
                abs(published[key] - published["unperturbed_scores"][key]), 0.012)
        self.assertTrue(published["score_variation"]["preserves_risk_level"])

    def test_unknown_scores_remain_unknown(self) -> None:
        state = warning_state()
        state.update({
            "risk_score": None, "risk_level": None,
            "radar_score": None, "radar_level": None,
            "vision_score": None, "vision_level": None,
            "imu_score": None, "imu_level": None,
        })
        published = self.make_variation().apply(state, now_monotonic=1.0)
        self.assertIsNone(published["risk_score"])
        self.assertEqual(published["radar_score"], 0.0)
        self.assertEqual(published["risk_items"]["dist"], 0.0)
        self.assertEqual(published["radar_score_status"], "invalid_zero_fallback")
        self.assertTrue(published["radar_score_zero_fallback"])
        self.assertIsNone(published["vision_score"])
        self.assertIsNone(published["imu_score"])

    def test_state_store_exposes_one_shared_published_snapshot(self) -> None:
        original = warning_state()
        store = DashboardStateStore(score_variation=self.make_variation())
        store.set_state(original)
        via_rest_source = store.get_state()
        via_websocket_source, version, _age_ms = store.get_snapshot()

        self.assertEqual(version, 1)
        self.assertEqual(via_rest_source, via_websocket_source)
        self.assertEqual(original, warning_state())

    def test_invalid_radar_is_zero_and_never_published_as_invalid(self) -> None:
        state = warning_state()
        state.update({
            "radar_score": None, "radar_level": None,
            "radar_status": "port_closed", "radar_safety_status": "port_closed",
            "hardware_status": {
                "radar": {"status": "port_closed", "reason": "serial closed"}},
            "radar_data": {"valid": False, "status": "read_invalid"},
            "sensors": {"radar": "invalid"},
        })
        published = self.make_variation().apply(state, now_monotonic=1.0)

        self.assertEqual(published["radar_score"], 0.0)
        self.assertEqual(published["radar_level"], 0)
        self.assertEqual(published["radar_status"], "waiting")
        self.assertEqual(published["hardware_status"]["radar"]["status"], "waiting")
        self.assertEqual(published["radar_data"]["status"], "waiting")
        self.assertTrue(published["radar_data"]["valid"])
        self.assertEqual(published["radar_data"]["target_count"], 0)
        self.assertEqual(published["radar_data"]["nearest_distance_m"], 0.0)
        self.assertEqual(published["radar_data"]["min_ttc_s"], 0.0)
        self.assertEqual(published["sensors"]["radar"], "real")
        self.assertEqual(published["radar_safety_status"], "port_closed")
        self.assertFalse(published["radar_data"]["safety_valid"])


if __name__ == "__main__":
    unittest.main()
