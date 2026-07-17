from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import start_demo


class StartDemoTests(unittest.TestCase):
    def test_project_runtime_is_discoverable(self) -> None:
        runtime = start_demo._runtime_python()
        self.assertIsNotNone(runtime)
        self.assertTrue(runtime.is_file())

    def test_final_demo_enables_complete_sensor_rule_and_motor_chain(self) -> None:
        with patch("sys.argv", ["start_demo.py"]), patch.object(
            start_demo, "run_dashboard", return_value=0
        ) as launch:
            self.assertEqual(start_demo.main(), 0)

        args = launch.call_args.args[0]
        self.assertIn("--enable-vision", args)
        self.assertIn("--enable-imu", args)
        self.assertIn("--enable-risk-rule", args)
        self.assertEqual(args[args.index("--state-hz") + 1], "20")
        self.assertEqual(args[args.index("--motor-mode") + 1], "real")
        self.assertIn("--confirm-motor-real", args)
        self.assertIn("--cloud-enable", args)
        self.assertEqual(
            args[args.index("--cloud-url") + 1], "http://124.70.108.34")
        self.assertEqual(args[args.index("--device-id") + 1], "bike-001")

    def test_diagnostic_switches_can_disable_optional_chain(self) -> None:
        with patch("sys.argv", [
            "start_demo.py", "--disable-imu", "--disable-risk-rule",
            "--disable-motor",
            "--disable-cloud",
        ]), patch.object(start_demo, "run_dashboard", return_value=0) as launch:
            self.assertEqual(start_demo.main(), 0)

        args = launch.call_args.args[0]
        self.assertNotIn("--enable-imu", args)
        self.assertNotIn("--enable-risk-rule", args)
        self.assertEqual(args[args.index("--motor-mode") + 1], "off")
        self.assertNotIn("--confirm-motor-real", args)
        self.assertNotIn("--cloud-enable", args)


if __name__ == "__main__":
    unittest.main()
