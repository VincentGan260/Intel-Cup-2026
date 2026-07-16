"""Bounded GPS context for uncalibrated single-frame visual proximity."""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.fusion.risk_score_contract import HIGH_SCORE


@dataclass(frozen=True)
class GpsSpeedModifierConfig:
    neutral_below_kmh: float = 5.0
    full_effect_kmh: float = 30.0
    max_factor: float = 1.25
    high_score_boundary: float = HIGH_SCORE

    def __post_init__(self) -> None:
        if self.neutral_below_kmh < 0.0:
            raise ValueError("neutral speed must be non-negative")
        if self.full_effect_kmh <= self.neutral_below_kmh:
            raise ValueError("full-effect speed must exceed neutral speed")
        if self.max_factor < 1.0:
            raise ValueError("maximum speed factor must be at least one")
        if not 0.0 < self.high_score_boundary <= 1.0:
            raise ValueError("high score boundary must be within (0, 1]")


DEFAULT_GPS_SPEED_MODIFIER = GpsSpeedModifierConfig()


def gps_speed_factor(
    speed_kmh: float,
    *,
    usable: bool,
    config: GpsSpeedModifierConfig = DEFAULT_GPS_SPEED_MODIFIER,
) -> float:
    """Return a bounded context factor; invalid GPS is always neutral."""
    try:
        speed = float(speed_kmh)
    except (TypeError, ValueError):
        return 1.0
    if not usable or not math.isfinite(speed) or speed < 0.0:
        return 1.0
    progress = ((speed - config.neutral_below_kmh)
                / (config.full_effect_kmh - config.neutral_below_kmh))
    progress = max(0.0, min(1.0, progress))
    return 1.0 + (config.max_factor - 1.0) * progress


def adjust_visual_proximity_score(
    proximity_score: float,
    speed_kmh: float,
    *,
    gps_usable: bool,
    config: GpsSpeedModifierConfig = DEFAULT_GPS_SPEED_MODIFIER,
) -> float:
    """Adjust only visual proximity and never cross the high-risk boundary."""
    try:
        score = float(proximity_score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    score = max(0.0, min(1.0, score))
    factor = gps_speed_factor(speed_kmh, usable=gps_usable, config=config)
    adjusted = 1.0 - math.pow(1.0 - score, factor)
    high_cap = math.nextafter(config.high_score_boundary, 0.0)
    return max(0.0, min(high_cap, adjusted))
