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


def build_rule(*, margin=0.025, attention_reference_s=4.0,
               urgent_reference_s=2.5):
    return PhysicalRiskRule(
        body_width_m=0.66,
        point_gate_lateral_margin_m=margin,
        mounting_offset_m=-0.055,
        mounting_uncertainty_m=0.005,
        configured_warning_range_m=20.0,
        radar_to_motor_p95_s=0.20,
        attention_reference_s=attention_reference_s,
        urgent_reference_s=urgent_reference_s,
    )


def main() -> None:
    rule = build_rule()

    invalid = rule.decide(radar(False), radar_fresh=True)
    assert invalid.status == "unknown" and invalid.risk_score is None
    stale = rule.decide(radar(), radar_fresh=False)
    assert stale.reason == "radar_frame_stale" and stale.risk_score is None
    clear = rule.decide(radar(), radar_fresh=True)
    assert clear.level == 0 and clear.risk_score == 0.0

    one_bad_one_good = radar(targets=[target(distance=math.nan), target(distance=4, speed=-1)])
    d = rule.decide(one_bad_one_good, radar_fresh=True)
    assert d.level == 1 and d.invalid_target_count == 1 and d.valid_target_count == 1

    all_bad = radar(targets=[target(distance=math.nan), target(angle=20)])
    d = rule.decide(all_bad, radar_fresh=True)
    assert d.status == "degraded" and d.reason == "all_reported_targets_invalid"

    outside_configured_range = radar(targets=[target(distance=21, speed=-20)])
    d = rule.decide(outside_configured_range, radar_fresh=True)
    assert d.status == "normal" and d.level == 0 and d.valid_target_count == 1

    center_angle = math.degrees(math.asin(0.055 / 5.0))
    center_target = radar(targets=[target(distance=5, speed=-1.5, angle=center_angle)])
    d = rule.decide(center_target, radar_fresh=True)
    assert d.level == 1 and 0.35 <= (d.risk_score or 0.0) < 0.70
    assert abs(d.critical_lateral_m or 0.0) < 1e-9

    urgent = radar(targets=[target(distance=2.5, speed=-1, angle=center_angle)])
    d = rule.decide(urgent, radar_fresh=True)
    assert d.level == 2 and d.risk_score is not None and d.risk_score > 0.70
    assert abs(d.urgent_ttc_s - 2.7) < 1e-9
    assert abs(d.attention_ttc_s - 4.2) < 1e-9

    at_attention = radar(targets=[target(distance=4.2, speed=-1, angle=center_angle)])
    after_attention = radar(targets=[target(distance=4.201, speed=-1, angle=center_angle)])
    attention_decision = rule.decide(at_attention, radar_fresh=True)
    after_attention_decision = rule.decide(after_attention, radar_fresh=True)
    assert attention_decision.level == 1
    assert abs((attention_decision.risk_score or 0.0) - 0.35) < 1e-9
    assert after_attention_decision.level == 0
    assert 0.0 < (after_attention_decision.risk_score or 0.0) < 0.35
    assert after_attention_decision.reason == "approaching_target_beyond_attention_reference"

    receding = radar(targets=[target(distance=2, speed=1)])
    assert rule.decide(receding, radar_fresh=True).level == 0

    expected_half_widths = {
        0.015: 0.35,
        0.025: 0.36,
        0.035: 0.37,
    }
    for margin, half_width in expected_half_widths.items():
        candidate_rule = build_rule(margin=margin)
        assert abs(candidate_rule.point_gate_half_width_m - half_width) < 1e-9
        boundary_lateral = half_width + 0.001
        boundary_angle = math.degrees(math.asin((boundary_lateral + 0.055) / 5.0))
        d = candidate_rule.decide(
            radar(targets=[target(distance=5, speed=-1, angle=boundary_angle)]),
            radar_fresh=True,
        )
        assert d.level == 0 and d.risk_score == 0.0
        assert d.reason == "no_approaching_target_in_point_gate"

    for urgent_reference_s, expected in ((2.0, 2.20), (2.5, 2.70), (3.0, 3.20)):
        candidate_rule = build_rule(urgent_reference_s=urgent_reference_s)
        assert abs(candidate_rule.urgent_ttc_s - expected) < 1e-9
        at_boundary = radar(targets=[target(distance=expected, speed=-1, angle=center_angle)])
        just_after = radar(targets=[target(distance=expected + 0.001, speed=-1,
                                           angle=center_angle)])
        boundary_decision = candidate_rule.decide(at_boundary, radar_fresh=True)
        after_decision = candidate_rule.decide(just_after, radar_fresh=True)
        assert boundary_decision.level == 2
        assert abs((boundary_decision.risk_score or 0.0) - 0.70) < 1e-9
        assert after_decision.level == 1
        assert 0.35 <= (after_decision.risk_score or 0.0) < 0.70

        ttc_90 = expected / 3.0
        score_90 = candidate_rule.decide(
            radar(targets=[target(distance=ttc_90, speed=-1, angle=center_angle)]),
            radar_fresh=True,
        )
        assert score_90.level == 2
        assert abs((score_90.risk_score or 0.0) - 0.90) < 1e-9

    event = rule.evaluate_event(
        center_target, radar_fresh=True, sequence=1,
        packet_monotonic_ns=1, completed_monotonic_ns=2,
    )
    assert event.risk_score == rule.decide(center_target, radar_fresh=True).risk_score
    assert event.details["risk_score_semantics"] == "intervention_urgency_not_probability"

    approach_scores = [
        rule.decide(radar(targets=[target(distance=distance, speed=-1)]),
                    radar_fresh=True).risk_score
        for distance in (15.0, 10.0, 5.0, 2.7, 1.0)
    ]
    assert all(score is not None for score in approach_scores)
    assert all(left < right for left, right in zip(approach_scores, approach_scores[1:]))

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
