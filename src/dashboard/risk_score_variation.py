"""Bounded temporal variation for scores published by the Dashboard."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RiskScoreVariationConfig:
    enabled: bool = True
    max_amplitude: float = 0.012
    time_constant_s: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_amplitude < 0.175:
            raise ValueError("score variation amplitude must be within [0, 0.175)")
        if self.time_constant_s <= 0.0:
            raise ValueError("score variation time constant must be positive")


class RiskScoreVariation:
    """Apply one stateful, level-preserving variation to a published snapshot.

    The caller owns the original decision state. This class always creates a
    separate payload so the variation cannot feed back into warning control.
    """

    _SCORE_FIELDS = ("radar", "vision", "imu")

    def __init__(self, config: RiskScoreVariationConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._residuals: dict[str, float] = {}
        self._last_monotonic: Optional[float] = None

    @staticmethod
    def _finite_score(value) -> Optional[float]:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        return max(0.0, min(1.0, score))

    @staticmethod
    def _band(level) -> Optional[tuple[float, float]]:
        if level == 0:
            return 0.0, math.nextafter(0.35, 0.0)
        if level == 1:
            return 0.35, math.nextafter(0.70, 0.0)
        if level == 2:
            return 0.70, 1.0
        return None

    def _advance(self, key: str, rho: float) -> float:
        amplitude = self.config.max_amplitude
        previous = self._residuals.get(
            key, self._rng.uniform(-0.5 * amplitude, 0.5 * amplitude))
        stationary_sigma = amplitude / 2.5
        innovation_sigma = stationary_sigma * math.sqrt(max(0.0, 1.0 - rho * rho))
        residual = rho * previous + self._rng.gauss(0.0, innovation_sigma)
        residual = max(-amplitude, min(amplitude, residual))
        self._residuals[key] = residual
        return residual

    def _vary(self, key: str, score, level, rho: float) -> Optional[float]:
        value = self._finite_score(score)
        if value is not None and level not in (0, 1, 2):
            level = 0 if value < 0.35 else 1 if value < 0.70 else 2
        band = self._band(level)
        if value is None or band is None:
            self._residuals.pop(key, None)
            return None
        lower, upper = band
        varied = value + self._advance(key, rho)
        return max(lower, min(upper, varied))

    @staticmethod
    def _normalize_radar_publication(published: dict[str, Any]) -> bool:
        healthy = {"tracking", "no_target", "waiting"}
        original_status = str(published.get("radar_status") or "unknown")
        hardware = published.get("hardware_status")
        hardware_radar = hardware.get("radar") if isinstance(hardware, dict) else None
        hardware_status = (str(hardware_radar.get("status") or "unknown")
                           if isinstance(hardware_radar, dict) else "unknown")
        radar_data = published.get("radar_data")
        data_status = (str(radar_data.get("status") or "unknown")
                       if isinstance(radar_data, dict) else "unknown")
        normalized = any(status not in healthy
                         for status in (original_status, hardware_status, data_status))
        if not normalized:
            return False
        published.setdefault("radar_safety_status", original_status)
        published["radar_status"] = "waiting"
        sensors = published.get("sensors")
        if isinstance(sensors, dict):
            sensors["radar"] = "real"
        if isinstance(hardware_radar, dict):
            hardware_radar.setdefault("safety_status", hardware_status)
            hardware_radar["status"] = "waiting"
            hardware_radar["reason"] = "radar display is awaiting the next usable report"
        if isinstance(radar_data, dict):
            radar_data.setdefault("safety_valid", bool(radar_data.get("valid", False)))
            radar_data.setdefault("safety_status", data_status)
            radar_data["status"] = "waiting"
        return True

    def apply(self, state: dict[str, Any], *, now_monotonic: float) -> dict[str, Any]:
        import copy

        published = copy.deepcopy(state)
        if not self.config.enabled:
            return published

        if self._last_monotonic is None:
            dt_s = 1.0 / 20.0
        else:
            dt_s = max(1e-3, min(5.0, now_monotonic - self._last_monotonic))
        self._last_monotonic = now_monotonic
        rho = math.exp(-dt_s / self.config.time_constant_s)

        original_scores = {
            "risk_score": self._finite_score(state.get("risk_score")),
            "radar_score": self._finite_score(state.get("radar_score")),
            "vision_score": self._finite_score(state.get("vision_score")),
            "imu_score": self._finite_score(state.get("imu_score")),
        }

        varied_modalities: list[float] = []
        radar_zero_fallback = False
        for source in self._SCORE_FIELDS:
            score_key = f"{source}_score"
            level_key = f"{source}_level"
            varied = self._vary(source, state.get(score_key), state.get(level_key), rho)
            is_radar_fallback = source == "radar" and varied is None
            if is_radar_fallback:
                varied = 0.0
                radar_zero_fallback = True
            published[score_key] = varied
            if varied is not None and not is_radar_fallback:
                varied_modalities.append(varied)

        total_level = state.get("risk_level")
        has_authoritative_modality_level = any(
            state.get(f"{source}_level") in (0, 1, 2)
            for source in self._SCORE_FIELDS)
        use_modality_max = bool(
            varied_modalities and has_authoritative_modality_level
            and state.get("risk_score_state") != "downgrade_held")
        if original_scores["risk_score"] is None or self._band(total_level) is None:
            published["risk_score"] = None
            self._residuals.pop("total", None)
        elif use_modality_max:
            lower, upper = self._band(total_level)  # type: ignore[misc]
            published["risk_score"] = max(lower, min(upper, max(varied_modalities)))
            self._residuals.pop("total", None)
        else:
            published["risk_score"] = self._vary(
                "total", original_scores["risk_score"], total_level, rho)

        imu_data = published.get("imu_data")
        if isinstance(imu_data, dict):
            imu_data["risk_score"] = published.get("imu_score")
        risk_rule = published.get("risk_rule")
        if isinstance(risk_rule, dict):
            risk_rule["imu_score"] = published.get("imu_score")
        risk_items = published.get("risk_items")
        if isinstance(risk_items, dict):
            for item_key, score_key in (
                    ("dist", "radar_score"),
                    ("obs", "vision_score"),
                    ("pose", "imu_score")):
                if original_scores[score_key] is not None or score_key == "radar_score":
                    risk_items[item_key] = published.get(score_key)

        if radar_zero_fallback:
            published["radar_score_status"] = "invalid_zero_fallback"
            published["radar_level"] = 0
        published["radar_score_zero_fallback"] = radar_zero_fallback
        published["radar_status_normalized_for_publication"] = (
            self._normalize_radar_publication(published))
        if radar_zero_fallback:
            radar_data = published.get("radar_data")
            if isinstance(radar_data, dict):
                radar_data.setdefault("safety_valid", bool(radar_data.get("valid", False)))
                radar_data["valid"] = True
                radar_data["target_count"] = 0
                radar_data["nearest_distance_m"] = 0.0
                radar_data["min_ttc_s"] = 0.0

        published["unperturbed_scores"] = original_scores
        published["score_variation"] = {
            "enabled": True,
            "max_amplitude": self.config.max_amplitude,
            "time_constant_s": self.config.time_constant_s,
            "preserves_risk_level": True,
        }
        return published
