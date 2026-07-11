"""Motor controller with a timestamp at the DRV2605 GO write."""
from __future__ import annotations

import time

from src.actuator.motor_controller import MotorController


class TimedMotorController(MotorController):
    """Adds software dispatch observability without changing vibration patterns."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_dispatch_monotonic_ns = 0
        self.last_command_was_dispatched = False

    def execute(self, command) -> None:
        previous_level = self._last_level
        previous_time = self._last_level_time
        self.last_command_was_dispatched = False
        if self.is_mock:
            # In mock mode this is the command-request timestamp.
            self.last_dispatch_monotonic_ns = time.monotonic_ns()
        super().execute(command)
        changed = self._last_level != previous_level or self._last_level_time != previous_time
        if self.is_mock:
            self.last_command_was_dispatched = changed

    def _play_effect(self, effect_id: int, duration: float) -> None:
        if self._bus is None:
            return
        self._bus.write_byte_data(self.i2c_addr, 0x04, effect_id)
        self._bus.write_byte_data(self.i2c_addr, 0x0C, 0x01)
        if not self.last_command_was_dispatched:
            self.last_dispatch_monotonic_ns = time.monotonic_ns()
            self.last_command_was_dispatched = True
        time.sleep(duration)
        self._bus.write_byte_data(self.i2c_addr, 0x0C, 0x00)
