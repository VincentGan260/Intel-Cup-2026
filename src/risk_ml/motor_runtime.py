"""Fail-closed real-motor adapter for the standalone XGBoost runtime."""

from __future__ import annotations

import threading
import time


class XGBoostMotorRuntime:
    """Own one MotorController and expose a small XGBoost-only contract."""

    def __init__(self, *, mode: str, i2c_bus: int, i2c_addr: int) -> None:
        if mode not in {"mock", "real"}:
            raise ValueError(f"unsupported motor mode: {mode}")
        self.mode = mode
        self.i2c_bus = int(i2c_bus)
        self.i2c_addr = int(i2c_addr)
        self._controller = None
        self._lock = threading.Lock()
        self._connected = False
        self._faulted = False
        self._last_error = ""
        self._gate_open = False
        self._gate_reason = "starting"
        self._commanded_level = 0
        self._dispatch_count = 0
        self._last_dispatch_wall = 0.0

    def start(self) -> None:
        from src.actuator.motor_controller import MotorController

        controller = MotorController(
            mode=self.mode,
            i2c_bus=self.i2c_bus,
            i2c_addr=self.i2c_addr,
        )
        controller.start()
        connected = (
            controller.is_real and getattr(controller, "_bus", None) is not None
        )
        if self.mode == "real" and not connected:
            error = controller.last_error or "real motor did not open its I2C bus"
            controller.stop()
            raise RuntimeError(error)
        with self._lock:
            self._controller = controller
            self._connected = connected if self.mode == "real" else True
            self._faulted = False
            self._last_error = ""
            self._gate_reason = "waiting_for_prediction"

    def apply_prediction(
        self,
        *,
        level: int,
        risk_score: float,
        gate_open: bool,
        gate_reason: str,
    ) -> None:
        with self._lock:
            self._gate_open = bool(gate_open)
            self._gate_reason = str(gate_reason)
            controller = self._controller
            if controller is None or self._faulted:
                return

            commanded_level = int(level) if gate_open else 0
            commanded_level = max(0, min(2, commanded_level))
            now = time.time()
            if (
                commanded_level == 1
                and self._commanded_level == 1
                and self._last_dispatch_wall
                and now - self._last_dispatch_wall
                < float(getattr(controller, "cooldown_sec", 2.0))
            ):
                return
            try:
                if commanded_level == 2:
                    controller.alert_high(risk_score)
                elif commanded_level == 1:
                    controller.alert_medium(risk_score)
                else:
                    controller.alert_low()
                self._commanded_level = commanded_level
                self._dispatch_count += int(commanded_level > 0)
                self._last_dispatch_wall = (
                    now if commanded_level > 0
                    else self._last_dispatch_wall
                )
            except Exception as exc:
                self._faulted = True
                self._connected = False
                self._last_error = str(exc)
                self._gate_open = False
                self._gate_reason = "motor_fault"
                try:
                    controller.stop()
                finally:
                    self._controller = None

    def fail_closed(self, reason: str) -> None:
        self.apply_prediction(
            level=0,
            risk_score=0.0,
            gate_open=False,
            gate_reason=reason,
        )

    def snapshot(self) -> dict:
        with self._lock:
            controller = self._controller
            latest = controller.get_latest() if controller is not None else None
            return {
                "enabled": True,
                "mode": self.mode,
                "connected": self._connected,
                "faulted": self._faulted,
                "gate_open": self._gate_open,
                "gate_reason": self._gate_reason,
                "commanded_level": self._commanded_level,
                "dispatch_count": self._dispatch_count,
                "last_dispatch_wall": (
                    round(self._last_dispatch_wall, 6)
                    if self._last_dispatch_wall else None
                ),
                "last_error": self._last_error,
                "i2c_bus": self.i2c_bus,
                "i2c_address": f"0x{self.i2c_addr:02X}",
                "medium_repeat_interval_sec": (
                    float(getattr(controller, "cooldown_sec", 2.0))
                    if controller is not None else 2.0
                ),
                "last_command": (
                    {
                        "risk_level": int(latest.risk_level),
                        "risk_score": round(float(latest.risk_score), 6),
                        "pattern": str(latest.pattern),
                        "duration_ms": int(latest.duration_ms),
                    }
                    if latest is not None else None
                ),
            }

    def shutdown(self) -> None:
        with self._lock:
            controller = self._controller
            self._gate_open = False
            self._gate_reason = "shutdown"
            self._commanded_level = 0
            if controller is None:
                self._connected = False
                return
            try:
                controller.alert_low()
                controller.stop()
            finally:
                self._controller = None
                self._connected = False
