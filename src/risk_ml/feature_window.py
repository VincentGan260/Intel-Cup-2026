"""Independent one-second feature extraction for the XGBoost runtime.

No deterministic warning-rule class is imported here.  The extractor only
turns raw sensor/vision values into the 31 fields used during training.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

from src.fusion.data_types import GPSData, IMUData, RadarData, RadarTarget, VisionData, VisionObject


FEATURE_NAMES = (
    "gps_valid",
    "gps_speed_kmh",
    "imu_valid",
    "pitch_abs_deg",
    "roll_abs_deg",
    "roll_error_deg",
    "outward_rate_deg_s",
    "imu_attention_duration_ms",
    "imu_urgent_consistent_samples",
    "acc_norm_mean_mps2",
    "acc_delta_signed_mps2",
    "acc_change_abs_mps2",
    "jerk_abs_mps3",
    "radar_valid",
    "radar_target_count",
    "radar_path_target_count",
    "radar_min_distance_m",
    "radar_relative_speed_mps",
    "radar_closing_speed_mps",
    "radar_ttc_s",
    "vision_valid",
    "object_count",
    "path_object_count",
    "max_path_bottom_ratio",
    "box_growth_rate_per_s",
    "growth_duration_s",
    "visual_tau_s",
    "vision_confidence",
)

@dataclass(frozen=True)
class FeatureWindowConfig:
    window_s: float = 1.0
    warmup_s: float = 0.5
    path_gate_half_width_m: float = 0.355
    radar_max_abs_angle_deg: float = 15.0
    vision_center_x_min_ratio: float = 0.25
    vision_center_x_max_ratio: float = 0.75
    vision_path_bottom_min_ratio: float = 0.40
    imu_turn_sign: float = -1.0
    gravity_mps2: float = 9.80665
    min_turn_compensation_speed_kmh: float = 3.0
    imu_attention_error_deg: float = 10.0
    imu_critical_error_deg: float = 25.0
    imu_attention_outward_rate_deg_s: float = 5.0
    imu_urgent_min_error_deg: float = 8.0
    imu_urgent_outward_rate_deg_s: float = 10.0
    imu_prediction_horizon_s: float = 0.8
    imu_max_sample_gap_ms: float = 250.0
    vision_growth_epsilon_per_s: float = 0.02

    @classmethod
    def from_mapping(cls, values: dict) -> "FeatureWindowConfig":
        return cls(**{
            field: values[field]
            for field in cls.__dataclass_fields__
            if field in values
        })


@dataclass(frozen=True)
class FeatureFrame:
    timestamp_monotonic: float
    values: dict[str, float | int | None]
    warm: bool
    window_age_s: float
    diagnostics: dict


def _finite(value: object, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _acc_norm(imu: IMUData) -> float:
    return math.sqrt(
        _finite(imu.acc_x) ** 2 + _finite(imu.acc_y) ** 2 + _finite(imu.acc_z) ** 2
    )


def _object_scale(obj: VisionObject, frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = (_finite(value) for value in obj.bbox)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = max(1.0, float(frame_width * frame_height))
    return math.sqrt(area / frame_area)


class XGBoostFeatureWindow:
    """Stateful feature extractor for raw live sensor snapshots."""

    def __init__(self, config: FeatureWindowConfig | None = None) -> None:
        self.config = config or FeatureWindowConfig()
        self._started_at: Optional[float] = None
        self._imu_samples: deque[tuple[float, float, float]] = deque()
        self._last_imu_source_ts: Optional[float] = None
        self._last_imu_update_time: Optional[float] = None
        self._previous_equilibrium_roll_deg = 0.0
        self._previous_turn_compensation_valid = False
        self._previous_error_sign = 0
        self._attention_started_at: Optional[float] = None
        self._urgent_consistent_samples = 0
        self._last_vision_source_ts: Optional[float] = None
        self._last_target_scale: Optional[float] = None
        self._last_target_time: Optional[float] = None
        self._growth_started_at: Optional[float] = None
        self._box_growth_rate = 0.0
        self._growth_duration_s = 0.0
        self._visual_tau_s: Optional[float] = None

    def update(
        self,
        *,
        now_monotonic: float,
        gps: GPSData,
        imu: IMUData,
        radar: RadarData,
        vision: VisionData,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> FeatureFrame:
        if self._started_at is None:
            self._started_at = now_monotonic

        roll = _finite(imu.body_roll if imu.body_roll is not None else imu.roll)
        pitch = _finite(imu.body_pitch if imu.body_pitch is not None else imu.pitch)
        roll_abs = abs(roll)
        pitch_abs = abs(pitch)
        if imu.valid:
            roll_error, outward_rate = self._update_imu(
                now_monotonic, imu, roll,
                gps_speed_kmh=_finite(gps.speed_kmh),
                gps_usable=bool(gps.valid),
            )
        else:
            self._reset_imu_state()
            roll_error, outward_rate = roll_abs, 0.0

        radar_features = self._radar_features(radar)
        vision_features = self._vision_features(
            now_monotonic, vision, frame_width, frame_height
        )
        acceleration = self._acceleration_features()
        attention_duration_ms = (
            max(0.0, (now_monotonic - self._attention_started_at) * 1000.0)
            if self._attention_started_at is not None else 0.0
        )
        values: dict[str, float | int | None] = {
            "gps_valid": int(bool(gps.valid)),
            "gps_speed_kmh": _finite(gps.speed_kmh) if gps.valid else None,
            "imu_valid": int(bool(imu.valid)),
            "pitch_abs_deg": pitch_abs,
            "roll_abs_deg": roll_abs,
            # Model-local feature engineering; no old warning-rule object is
            # called or imported.
            "roll_error_deg": roll_error,
            "outward_rate_deg_s": outward_rate,
            "imu_attention_duration_ms": attention_duration_ms,
            "imu_urgent_consistent_samples": self._urgent_consistent_samples,
            **acceleration,
            **radar_features,
            **vision_features,
        }
        if tuple(values) != FEATURE_NAMES:
            raise RuntimeError("feature extractor order drifted from FEATURE_NAMES")

        age = max(0.0, now_monotonic - self._started_at)
        return FeatureFrame(
            timestamp_monotonic=now_monotonic,
            values=values,
            warm=age >= self.config.warmup_s,
            window_age_s=min(age, self.config.window_s),
            diagnostics={
                "imu_samples": len(self._imu_samples),
                "vision_track_active": self._last_target_scale is not None,
                "missing_values": [
                    name for name, value in values.items()
                    if value is None or (
                        isinstance(value, float) and not math.isfinite(value)
                    )
                ],
            },
        )

    def _update_imu(
        self,
        now: float,
        imu: IMUData,
        body_roll_deg: float,
        *,
        gps_speed_kmh: float,
        gps_usable: bool,
    ) -> tuple[float, float]:
        source_ts = _finite(imu.timestamp, now)
        dt = (
            max(1e-3, now - self._last_imu_update_time)
            if self._last_imu_update_time is not None else None
        )
        if dt is not None and dt * 1000.0 > self.config.imu_max_sample_gap_ms:
            self._attention_started_at = None
            self._urgent_consistent_samples = 0

        compensation_valid = (
            gps_usable
            and gps_speed_kmh >= self.config.min_turn_compensation_speed_kmh
        )
        equilibrium_roll = 0.0
        if compensation_valid:
            speed_mps = gps_speed_kmh / 3.6
            yaw_rate_rad_s = math.radians(_finite(imu.gyro_z))
            equilibrium_roll = self.config.imu_turn_sign * math.degrees(math.atan(
                speed_mps * yaw_rate_rad_s / self.config.gravity_mps2
            ))
        equilibrium_rate = 0.0
        if (
            compensation_valid
            and self._previous_turn_compensation_valid
            and dt is not None
            and dt * 1000.0 <= self.config.imu_max_sample_gap_ms
        ):
            equilibrium_rate = (
                equilibrium_roll - self._previous_equilibrium_roll_deg
            ) / dt
        signed_error = body_roll_deg - equilibrium_roll
        roll_error = abs(signed_error)
        error_sign = 1 if signed_error > 0.0 else -1 if signed_error < 0.0 else 0
        outward_rate = max(
            0.0, error_sign * (_finite(imu.gyro_x) - equilibrium_rate)
        )
        if self._previous_error_sign and error_sign != self._previous_error_sign:
            self._attention_started_at = None
            self._urgent_consistent_samples = 0

        if self._last_imu_source_ts != source_ts:
            self._last_imu_source_ts = source_ts
            self._imu_samples.append((now, _acc_norm(imu), abs(body_roll_deg)))
        cutoff = now - self.config.window_s
        while self._imu_samples and self._imu_samples[0][0] < cutoff:
            self._imu_samples.popleft()

        attention = (
            roll_error >= self.config.imu_attention_error_deg
            and outward_rate >= self.config.imu_attention_outward_rate_deg_s
        )
        if attention:
            if self._attention_started_at is None:
                self._attention_started_at = now
        else:
            self._attention_started_at = None

        time_to_critical = (
            max(0.0, (self.config.imu_critical_error_deg - roll_error) / outward_rate)
            if outward_rate > 0.0 else None
        )
        urgent = (
            compensation_valid
            and roll_error >= self.config.imu_urgent_min_error_deg
            and outward_rate >= self.config.imu_urgent_outward_rate_deg_s
            and time_to_critical is not None
            and time_to_critical <= self.config.imu_prediction_horizon_s
        )
        self._urgent_consistent_samples = (
            self._urgent_consistent_samples + 1 if urgent else 0
        )
        self._last_imu_update_time = now
        self._previous_equilibrium_roll_deg = equilibrium_roll
        self._previous_turn_compensation_valid = compensation_valid
        self._previous_error_sign = error_sign
        return roll_error, outward_rate

    def _reset_imu_state(self) -> None:
        self._imu_samples.clear()
        self._last_imu_source_ts = None
        self._last_imu_update_time = None
        self._previous_equilibrium_roll_deg = 0.0
        self._previous_turn_compensation_valid = False
        self._previous_error_sign = 0
        self._attention_started_at = None
        self._urgent_consistent_samples = 0

    def _acceleration_features(self) -> dict[str, float]:
        if not self._imu_samples:
            return {
                "acc_norm_mean_mps2": 0.0,
                "acc_delta_signed_mps2": 0.0,
                "acc_change_abs_mps2": 0.0,
                "jerk_abs_mps3": 0.0,
            }
        norms = [sample[1] for sample in self._imu_samples]
        delta = norms[-1] - norms[0]
        max_jerk = 0.0
        for previous, current in zip(self._imu_samples, list(self._imu_samples)[1:]):
            dt = max(1e-3, current[0] - previous[0])
            max_jerk = max(max_jerk, abs(current[1] - previous[1]) / dt)
        return {
            "acc_norm_mean_mps2": sum(norms) / len(norms),
            "acc_delta_signed_mps2": delta,
            "acc_change_abs_mps2": abs(delta),
            "jerk_abs_mps3": max_jerk,
        }

    def _radar_features(self, radar: RadarData) -> dict[str, float | int | None]:
        targets = list(radar.targets) if radar.valid else []
        usable = [
            target for target in targets
            if 0.0 < _finite(target.distance_m)
            and abs(_finite(target.angle_deg)) <= self.config.radar_max_abs_angle_deg
        ]
        path_targets = [
            target for target in usable
            if abs(_finite(target.distance_m) * math.sin(
                math.radians(_finite(target.angle_deg))
            )) <= self.config.path_gate_half_width_m
        ]
        approaching_path = [
            target for target in path_targets if _finite(target.relative_speed_mps) < -0.01
        ]
        approaching = [
            target for target in usable if _finite(target.relative_speed_mps) < -0.01
        ]

        selected: Optional[RadarTarget] = None
        if approaching_path:
            selected = min(
                approaching_path,
                key=lambda target: _finite(target.distance_m)
                / max(0.01, -_finite(target.relative_speed_mps)),
            )
        elif approaching:
            selected = min(
                approaching,
                key=lambda target: _finite(target.distance_m)
                / max(0.01, -_finite(target.relative_speed_mps)),
            )
        elif usable:
            selected = min(usable, key=lambda target: _finite(target.distance_m))

        distance: Optional[float] = None
        relative_speed = 0.0
        closing_speed = 0.0
        ttc: Optional[float] = None
        if selected is not None:
            distance = _finite(selected.distance_m)
            relative_speed = _finite(selected.relative_speed_mps)
            closing_speed = max(0.0, -relative_speed)
            if closing_speed > 0.01:
                ttc = distance / closing_speed
        return {
            "radar_valid": int(bool(radar.valid)),
            "radar_target_count": len(targets),
            "radar_path_target_count": len(path_targets),
            "radar_min_distance_m": distance,
            "radar_relative_speed_mps": relative_speed,
            "radar_closing_speed_mps": closing_speed,
            "radar_ttc_s": ttc,
        }

    def _vision_features(
        self,
        now: float,
        vision: VisionData,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, float | int | None]:
        if not vision.valid:
            self._reset_visual_track()
        objects = list(vision.objects) if vision.valid else []
        path_objects = [
            obj for obj in objects
            if self._is_path_object(obj, frame_width, frame_height)
        ]
        selected = max(
            path_objects,
            key=lambda obj: _finite(obj.bbox[3]) / max(1.0, frame_height),
            default=None,
        )

        bottom_ratio = 0.0
        confidence = 0.0
        if selected is not None:
            bottom_ratio = max(0.0, min(
                1.0, _finite(selected.bbox[3]) / max(1.0, frame_height)
            ))
            confidence = max(0.0, min(1.0, _finite(selected.confidence)))
            self._update_visual_track(
                now, vision.timestamp, selected, frame_width, frame_height
            )
        elif self._last_vision_source_ts != _finite(vision.timestamp):
            self._reset_visual_track()
            self._last_vision_source_ts = _finite(vision.timestamp)

        return {
            "vision_valid": int(bool(vision.valid)),
            "object_count": len(objects),
            "path_object_count": len(path_objects),
            "max_path_bottom_ratio": bottom_ratio,
            "box_growth_rate_per_s": self._box_growth_rate if selected else 0.0,
            "growth_duration_s": self._growth_duration_s if selected else 0.0,
            "visual_tau_s": self._visual_tau_s if selected else None,
            "vision_confidence": confidence,
        }

    def _is_path_object(
        self, obj: VisionObject, frame_width: int, frame_height: int
    ) -> bool:
        if obj.in_drivable_area is not None:
            return bool(obj.in_drivable_area)
        x1, _, x2, y2 = (_finite(value) for value in obj.bbox)
        center_x_ratio = ((x1 + x2) * 0.5) / max(1.0, frame_width)
        bottom_ratio = y2 / max(1.0, frame_height)
        return (
            self.config.vision_center_x_min_ratio <= center_x_ratio
            <= self.config.vision_center_x_max_ratio
            and bottom_ratio >= self.config.vision_path_bottom_min_ratio
        )

    def _update_visual_track(
        self,
        now: float,
        source_timestamp: float,
        selected: VisionObject,
        frame_width: int,
        frame_height: int,
    ) -> None:
        source_ts = _finite(source_timestamp, now)
        if self._last_vision_source_ts == source_ts:
            return
        self._last_vision_source_ts = source_ts
        scale = _object_scale(selected, frame_width, frame_height)
        growth = 0.0
        if (
            self._last_target_scale is not None
            and self._last_target_time is not None
            and self._last_target_scale > 1e-6
        ):
            dt = max(1e-3, now - self._last_target_time)
            growth = max(
                0.0, (scale - self._last_target_scale) / self._last_target_scale / dt
            )

        if growth >= self.config.vision_growth_epsilon_per_s:
            if self._growth_started_at is None:
                self._growth_started_at = self._last_target_time or now
            self._growth_duration_s = max(0.0, now - self._growth_started_at)
            self._visual_tau_s = 1.0 / growth
        else:
            self._growth_started_at = None
            self._growth_duration_s = 0.0
            self._visual_tau_s = None
        self._box_growth_rate = growth
        self._last_target_scale = scale
        self._last_target_time = now

    def _reset_visual_track(self) -> None:
        self._last_target_scale = None
        self._last_target_time = None
        self._growth_started_at = None
        self._box_growth_rate = 0.0
        self._growth_duration_s = 0.0
        self._visual_tau_s = None
