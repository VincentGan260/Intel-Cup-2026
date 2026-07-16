"""Dependency-light checks for the optional GPS visual-risk context."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.gps_risk_context import (
    GpsSpeedModifierConfig,
    adjust_visual_proximity_score,
    gps_speed_factor,
)


def main() -> None:
    assert gps_speed_factor(0.0, usable=True) == 1.0
    assert gps_speed_factor(5.0, usable=True) == 1.0
    assert abs(gps_speed_factor(17.5, usable=True) - 1.125) < 1e-12
    assert gps_speed_factor(30.0, usable=True) == 1.25
    assert gps_speed_factor(60.0, usable=True) == 1.25

    for invalid_speed in (-1.0, math.nan, math.inf, None, "invalid"):
        assert gps_speed_factor(invalid_speed, usable=True) == 1.0
    assert gps_speed_factor(30.0, usable=False) == 1.0

    base = 0.50
    assert adjust_visual_proximity_score(base, 30.0, gps_usable=False) == base
    adjusted = adjust_visual_proximity_score(base, 30.0, gps_usable=True)
    assert base < adjusted < 0.70
    assert adjust_visual_proximity_score(0.0, 30.0, gps_usable=True) == 0.0
    assert adjust_visual_proximity_score(0.699, 100.0, gps_usable=True) < 0.70
    assert adjust_visual_proximity_score(1.0, 100.0, gps_usable=True) < 0.70

    try:
        GpsSpeedModifierConfig(full_effect_kmh=5.0)
        raise AssertionError("invalid GPS speed range should fail")
    except ValueError:
        pass

    print("GPS risk context: all tests passed")


if __name__ == "__main__":
    main()
