"""Provisional lateral-instability warning rule for the WT61C IMU."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.fusion.warning_events import ModalityEvent


@dataclass(frozen=True)
class ImuWarningRuleConfig:
    calibration_status: str = "stationary_measured_pending_vehicle_validation"
    roll_offset_deg: float = -0.231277
    pitch_offset_deg: float = -0.518034
    turn_sign: float = -1.0
    gravity_mps2: float = 9.80665
    min_turn_compensation_speed_kmh: float = 3.0
    attention_error_deg: float = 10.0
    critical_error_deg: float = 25.0
    attention_outward_rate_deg_s: float = 5.0
    urgent_outward_rate_deg_s: float = 10.0
    attention_persistence_ms: float = 150.0
    prediction_horizon_s: float = 0.8
    urgent_min_error_deg: float = 8.0
    urgent_consistent_samples: int = 3
    max_sample_gap_ms: float = 250.0


class ImuWarningRule:
    """Convert roll dynamics into a continuous, independently usable risk event.

    Normal cornering is compensated with phi_eq = atan(v * yaw_rate / g).
    The thresholds are theory-derived engineering starting points and remain
    explicitly provisional until outdoor labelled runs are available.
    """

    def __init__(self, config: ImuWarningRuleConfig) -> None:
        self.config = config
        self._previous_capture_ns = 0
        self._previous_equilibrium_roll_deg = 0.0
        self._previous_turn_compensation_valid = False
        self._attention_started_ns = 0
        self._urgent_consistent_count = 0
        self._previous_error_sign = 0

    def reset(self) -> None:
        self._previous_capture_ns = 0
        self._previous_equilibrium_roll_deg = 0.0
        self._previous_turn_compensation_valid = False
        self._attention_started_ns = 0
        self._urgent_consistent_count = 0
        self._previous_error_sign = 0

    def evaluate_event(
        self,
        imu,
        *,
        capture_monotonic_ns: int,
        completed_monotonic_ns: int,
        sequence: int,
        gps_speed_kmh: Optional[float] = None,
        gps_usable: bool = False,
    ) -> ModalityEvent:
        if not bool(getattr(imu, "valid", False)):
            self.reset()
            return self._unavailable(
                capture_monotonic_ns, completed_monotonic_ns, sequence,
                "imu_sample_invalid", "invalid")
        if capture_monotonic_ns <= 0 or completed_monotonic_ns < capture_monotonic_ns:
            self.reset()
            return self._unavailable(
                capture_monotonic_ns, completed_monotonic_ns, sequence,
                "imu_timestamp_invalid", "invalid")

        values = (
            getattr(imu, "roll", None), getattr(imu, "gyro_x", None),
            getattr(imu, "gyro_z", None),
        )
        try:
            roll_deg, gyro_x_deg_s, gyro_z_deg_s = (float(value) for value in values)
        except (TypeError, ValueError):
            self.reset()
            return self._unavailable(
                capture_monotonic_ns, completed_monotonic_ns, sequence,
                "imu_numeric_value_invalid", "invalid")
        if not all(math.isfinite(value) for value in (
                roll_deg, gyro_x_deg_s, gyro_z_deg_s)):
            self.reset()
            return self._unavailable(
                capture_monotonic_ns, completed_monotonic_ns, sequence,
                "imu_numeric_value_invalid", "invalid")

        dt_s = None
        if self._previous_capture_ns:
            dt_s = (capture_monotonic_ns - self._previous_capture_ns) / 1_000_000_000.0
            if dt_s <= 0.0:
                return self._unavailable(
                    capture_monotonic_ns, completed_monotonic_ns, sequence,
                    "imu_sample_out_of_order", "out_of_order")
            if dt_s * 1000.0 > self.config.max_sample_gap_ms:
                self._attention_started_ns = 0
                self._urgent_consistent_count = 0

        speed_kmh = self._finite_float(gps_speed_kmh)
        turn_compensation_valid = bool(
            gps_usable and speed_kmh is not None
            and speed_kmh >= self.config.min_turn_compensation_speed_kmh)
        equilibrium_roll_deg = 0.0
        if turn_compensation_valid:
            speed_mps = speed_kmh / 3.6
            yaw_rate_rad_s = math.radians(gyro_z_deg_s)
            equilibrium_roll_deg = self.config.turn_sign * math.degrees(math.atan(
                speed_mps * yaw_rate_rad_s / self.config.gravity_mps2))

        equilibrium_rate_deg_s = 0.0
        if (turn_compensation_valid and self._previous_turn_compensation_valid
                and dt_s is not None
                and dt_s * 1000.0 <= self.config.max_sample_gap_ms):
            equilibrium_rate_deg_s = (
                equilibrium_roll_deg - self._previous_equilibrium_roll_deg) / dt_s

        calibrated_roll = self._finite_float(getattr(imu, "body_roll", None))
        body_roll_deg = (calibrated_roll if calibrated_roll is not None
                         else roll_deg - self.config.roll_offset_deg)
        roll_error_deg = body_roll_deg - equilibrium_roll_deg
        abs_error_deg = abs(roll_error_deg)
        error_sign = 1 if roll_error_deg > 0.0 else -1 if roll_error_deg < 0.0 else 0
        error_rate_deg_s = gyro_x_deg_s - equilibrium_rate_deg_s
        outward_rate_deg_s = max(0.0, error_sign * error_rate_deg_s)

        if self._previous_error_sign and error_sign != self._previous_error_sign:
            self._attention_started_ns = 0
            self._urgent_consistent_count = 0

        predicted_error_deg = (
            abs_error_deg + outward_rate_deg_s * self.config.prediction_horizon_s)
        time_to_critical_s = None
        if outward_rate_deg_s > 0.0:
            time_to_critical_s = max(
                0.0, (self.config.critical_error_deg - abs_error_deg)
                / outward_rate_deg_s)

        attention_condition = bool(
            abs_error_deg >= self.config.attention_error_deg
            and outward_rate_deg_s >= self.config.attention_outward_rate_deg_s)
        if attention_condition:
            if not self._attention_started_ns:
                self._attention_started_ns = capture_monotonic_ns
        else:
            self._attention_started_ns = 0
        attention_duration_ms = (
            (capture_monotonic_ns - self._attention_started_ns) / 1_000_000.0
            if self._attention_started_ns else 0.0)

        urgent_condition = bool(
            turn_compensation_valid
            and abs_error_deg >= self.config.urgent_min_error_deg
            and outward_rate_deg_s >= self.config.urgent_outward_rate_deg_s
            and time_to_critical_s is not None
            and time_to_critical_s <= self.config.prediction_horizon_s)
        self._urgent_consistent_count = (
            self._urgent_consistent_count + 1 if urgent_condition else 0)

        if self._urgent_consistent_count >= self.config.urgent_consistent_samples:
            level = 2
            urgency = 1.0 - min(
                1.0, float(time_to_critical_s) / self.config.prediction_horizon_s)
            risk_score = 0.70 + 0.30 * urgency
            reason = "imu_predicted_lateral_instability"
        elif (attention_condition
              and attention_duration_ms >= self.config.attention_persistence_ms):
            level = 1
            progress = self._progress(
                predicted_error_deg, self.config.attention_error_deg,
                self.config.critical_error_deg)
            risk_score = 0.35 + 0.35 * progress
            reason = "imu_sustained_outward_lean"
        else:
            level = 0
            risk_score = min(
                math.nextafter(0.35, 0.0),
                0.35 * predicted_error_deg / self.config.attention_error_deg)
            reason = "imu_lateral_state_nominal"

        self._previous_capture_ns = capture_monotonic_ns
        self._previous_equilibrium_roll_deg = equilibrium_roll_deg
        self._previous_turn_compensation_valid = turn_compensation_valid
        self._previous_error_sign = error_sign
        return ModalityEvent(
            source="imu", source_id=str(capture_monotonic_ns), sequence=sequence,
            capture_monotonic_ns=capture_monotonic_ns,
            completed_monotonic_ns=completed_monotonic_ns,
            usable=True, level=level, reason=reason,
            risk_score=max(0.0, min(1.0, risk_score)), status="usable",
            details={
                "calibration_status": self.config.calibration_status,
                "roll_raw_deg": roll_deg,
                "roll_offset_deg": self.config.roll_offset_deg,
                "pitch_offset_deg": self.config.pitch_offset_deg,
                "body_roll_deg": body_roll_deg,
                "gyro_x_deg_s": gyro_x_deg_s,
                "gyro_z_deg_s": gyro_z_deg_s,
                "gps_speed_kmh": speed_kmh,
                "turn_compensation_valid": turn_compensation_valid,
                "turn_compensation_status": (
                    "usable" if turn_compensation_valid else "degraded_speed_unavailable_or_low"),
                "equilibrium_roll_deg": equilibrium_roll_deg,
                "equilibrium_roll_rate_deg_s": equilibrium_rate_deg_s,
                "roll_error_deg": roll_error_deg,
                "error_rate_deg_s": error_rate_deg_s,
                "outward_rate_deg_s": outward_rate_deg_s,
                "predicted_error_deg": predicted_error_deg,
                "time_to_critical_s": time_to_critical_s,
                "attention_duration_ms": attention_duration_ms,
                "urgent_consistent_count": self._urgent_consistent_count,
                "high_risk_cap": None if turn_compensation_valid else 1,
            },
        )

    @staticmethod
    def _finite_float(value) -> Optional[float]:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    @staticmethod
    def _progress(value: float, start: float, end: float) -> float:
        if end <= start:
            return 1.0
        return max(0.0, min(1.0, (value - start) / (end - start)))

    @staticmethod
    def _unavailable(capture_ns: int, completed_ns: int, sequence: int,
                     reason: str, status: str) -> ModalityEvent:
        return ModalityEvent(
            source="imu", source_id=str(capture_ns), sequence=sequence,
            capture_monotonic_ns=capture_ns,
            completed_monotonic_ns=completed_ns,
            usable=False, level=None, reason=reason, risk_score=None,
            status=status,
        )
