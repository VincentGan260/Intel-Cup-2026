"""Deterministic tests for the competition TTC urgency rule."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.physical_risk_rule import PhysicalRiskRule
from src.fusion.data_types import RadarData
from src.sensors.radar_reader import RadarReader


def radar(valid=True, targets=None):
    return NS(valid=valid, targets=[] if targets is None else targets)


def target(distance=5.0, speed=-1.0, angle=0.0, confidence=1.0):
    return NS(distance_m=distance, relative_speed_mps=speed,
              angle_deg=angle, confidence=confidence)


def main() -> None:
    rule = PhysicalRiskRule(
        body_width_m=0.66,
        point_gate_lateral_margin_m=0.025,
        mounting_offset_m=-0.055,
        mounting_uncertainty_m=0.005,
        configured_warning_range_m=20.0,
        radar_to_motor_p95_s=0.20,
    )

    assert rule.decide(radar(False), radar_fresh=True).status == "unknown"
    assert rule.decide(radar(), radar_fresh=False).reason == "radar_frame_stale"
    assert rule.decide(radar(), radar_fresh=True).level == 0

    one_bad_one_good = radar(targets=[target(distance=math.nan), target(distance=5, speed=-1)])
    d = rule.decide(one_bad_one_good, radar_fresh=True)
    assert d.level == 1 and d.invalid_target_count == 1 and d.valid_target_count == 1

    all_bad = radar(targets=[target(distance=math.nan), target(angle=20)])
    d = rule.decide(all_bad, radar_fresh=True)
    assert d.status == "degraded" and d.reason == "all_reported_targets_invalid"

    outside_configured_range = radar(targets=[target(distance=21, speed=-20)])
    d = rule.decide(outside_configured_range, radar_fresh=True)
    assert d.status == "normal" and d.level == 0 and d.valid_target_count == 1

    center_angle = math.degrees(math.asin(0.055 / 5.0))
    center_target = radar(targets=[target(distance=5, speed=-1, angle=center_angle)])
    d = rule.decide(center_target, radar_fresh=True)
    assert d.level == 1 and abs(d.critical_lateral_m or 0.0) < 1e-9

    urgent = radar(targets=[target(distance=2.5, speed=-1, angle=center_angle)])
    d = rule.decide(urgent, radar_fresh=True)
    assert d.level == 2 and abs(d.urgent_ttc_s - 2.7) < 1e-9

    receding = radar(targets=[target(distance=2, speed=1)])
    assert rule.decide(receding, radar_fresh=True).level == 0

    # Real reader must retain an explicit no-target frame between the module's
    # approximately 1 s no-target reports; age-based logic owns staleness.
    reader = RadarReader(mode="real", config={})
    reader._serial = NS(is_open=True)
    sequence = iter([RadarData(valid=True, targets=[]), RadarData(valid=False)])
    reader._read_real = lambda _ts: next(sequence)
    first = reader.read_once()
    second = reader.read_once()
    assert first.valid and second is first

    print("competition risk rule: all tests passed")


if __name__ == "__main__":
    main()
