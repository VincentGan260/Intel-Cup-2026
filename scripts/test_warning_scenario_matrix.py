"""Offline acceptance matrix for warning levels, fusion, and state transitions."""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.physical_risk_rule import PhysicalRiskRule
from src.fusion.warning_arbiter import arbitrate_warning_events
from src.fusion.warning_events import ArbitrationResult, ModalityEvent
from src.fusion.warning_state import WarningState


@dataclass(frozen=True)
class Scenario:
    name: str
    radar: ModalityEvent
    vision: ModalityEvent
    expected_level: Optional[int]
    expected_status: str
    expected_reason: str
    gps: Optional[ModalityEvent] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    now_offset_ms: float = 0.0


def build_radar_rule() -> PhysicalRiskRule:
    return PhysicalRiskRule(
        body_width_m=0.66,
        point_gate_lateral_margin_m=0.025,
        mounting_offset_m=-0.055,
        mounting_uncertainty_m=0.005,
        configured_warning_range_m=20.0,
        radar_parsed_to_motor_go_p95_s=0.000469,
        attention_reference_s=4.0,
        urgent_reference_s=2.5,
        max_abs_angle_deg=15.0,
    )


def radar_event(rule: PhysicalRiskRule, now_ns: int, sequence: int, *,
                ttc_s: Optional[float] = None, lateral_m: float = 0.0,
                closing: bool = True, relative_speed_mps: Optional[float] = None,
                valid: bool = True) -> ModalityEvent:
    targets = []
    if ttc_s is not None:
        distance_m = max(0.1, float(ttc_s))
        speed_mps = (float(relative_speed_mps) if relative_speed_mps is not None
                     else -distance_m / float(ttc_s) if closing else 1.0)
        angle_deg = math.degrees(math.asin(
            (lateral_m - rule.mounting_offset_m) / distance_m))
        targets = [NS(distance_m=distance_m, relative_speed_mps=speed_mps,
                      angle_deg=angle_deg, confidence=1.0)]
    radar = NS(valid=valid, targets=targets)
    return rule.evaluate_event(
        radar, radar_fresh=True, sequence=sequence,
        packet_monotonic_ns=now_ns, completed_monotonic_ns=now_ns,
    )


def event(source: str, level: Optional[int], score: Optional[float], now_ns: int,
          sequence: int, *, reason: str, usable: bool = True,
          details: Optional[dict] = None, status: str = "usable") -> ModalityEvent:
    return ModalityEvent(
        source=source, source_id=str(sequence), sequence=sequence,
        capture_monotonic_ns=now_ns, completed_monotonic_ns=now_ns,
        usable=usable, level=level if usable else None,
        reason=reason, risk_score=score if usable else None,
        status=status if not usable else "usable", details=details or {},
    )


def vision_event(now_ns: int, sequence: int, *, level: int, proximity: float,
                 tau: float, reason: str) -> ModalityEvent:
    return event(
        "vision", level, max(proximity, tau), now_ns, sequence, reason=reason,
        details={"proximity_risk_score": proximity, "tau_risk_score": tau},
    )


def unavailable(source: str, now_ns: int, sequence: int) -> ModalityEvent:
    return event(
        source, None, None, now_ns, sequence, usable=False,
        reason=f"{source}_invalid", status="invalid",
    )


def check_scenario(scenario: Scenario, now_ns: int) -> ArbitrationResult:
    result = arbitrate_warning_events(
        scenario.radar, scenario.vision, scenario.gps,
        now_ns=now_ns + int(scenario.now_offset_ms * 1_000_000),
        target_stale_ms=500.0, vision_stale_ms=500.0,
        gps_stale_ms=1000.0, radar_communication_watchdog_ms=2000.0,
    )
    assert result.final_level == scenario.expected_level, scenario.name
    assert result.system_status == scenario.expected_status, scenario.name
    assert result.warning_reason == scenario.expected_reason, scenario.name
    if scenario.score_min is None:
        assert result.risk_score is None, scenario.name
    else:
        assert result.risk_score is not None, scenario.name
        assert scenario.score_min <= result.risk_score, scenario.name
        if scenario.score_max is not None:
            assert result.risk_score < scenario.score_max, scenario.name
    return result


def check_state_transitions(now_ns: int) -> list[tuple[str, Optional[int], str]]:
    state = WarningState(release_hold_ms=500.0)
    state.request(0, reason="clear", source="test", sequence=1, now_ns=now_ns)
    state.request(2, reason="urgent", source="test", sequence=2, now_ns=now_ns + 1)
    assert state.snapshot().current_level == 2

    state.request(0, reason="clear", source="test", sequence=3,
                  now_ns=now_ns + 2)
    assert state.snapshot().current_level == 2
    state.request(0, reason="clear", source="test", sequence=4,
                  now_ns=now_ns + 499_000_002)
    assert state.snapshot().current_level == 2
    state.request(0, reason="clear", source="test", sequence=5,
                  now_ns=now_ns + 500_000_002)
    assert state.snapshot().current_level == 0

    state.request(None, reason="all_sources_lost", source="test", sequence=6,
                  now_ns=now_ns + 500_000_003)
    snapshot = state.snapshot()
    assert snapshot.current_level is None
    assert snapshot.last_known_level == 0
    return [
        ("risk upgrade", 2, "immediate"),
        ("risk downgrade before 500 ms", 2, "held"),
        ("risk downgrade at 500 ms", 0, "released"),
        ("all sources unavailable", None, "unknown"),
    ]


def main() -> None:
    now_ns = time.monotonic_ns()
    rule = build_radar_rule()
    radar_clear = radar_event(rule, now_ns, 1)
    radar_attention = radar_event(rule, now_ns, 2, ttc_s=rule.attention_ttc_s)
    radar_before_attention = radar_event(
        rule, now_ns, 3, ttc_s=rule.attention_ttc_s + 0.001)
    radar_urgent = radar_event(rule, now_ns, 4, ttc_s=rule.urgent_ttc_s)
    radar_side_approaching = radar_event(
        rule, now_ns, 5, ttc_s=2.0, lateral_m=0.37)
    radar_static = radar_event(
        rule, now_ns, 6, ttc_s=2.0, lateral_m=0.0, relative_speed_mps=0.0)

    vision_clear = vision_event(
        now_ns, 1, level=0, proximity=0.0, tau=0.0,
        reason="no_visual_path_obstacle")
    vision_medium = vision_event(
        now_ns, 2, level=1, proximity=0.50, tau=0.0,
        reason="visual_path_obstacle_attention")
    vision_high = vision_event(
        now_ns, 3, level=2, proximity=0.60, tau=0.82,
        reason="visual_tau_entered_urgency_reference")
    gps_fast = event(
        "gps", 0, None, now_ns, 1, reason="gps_speed_context",
        details={"speed_kmh": 30.0})
    gps_invalid = unavailable("gps", now_ns, 2)

    scenarios = [
        Scenario("clear road", radar_clear, vision_clear, 0, "normal",
                 "no_warning_event", score_min=0.0, score_max=0.001),
        Scenario("radar just beyond attention", radar_before_attention, vision_clear,
                 0, "normal", "no_warning_event", score_min=0.0, score_max=0.35),
        Scenario("radar attention boundary", radar_attention, vision_clear,
                 1, "normal", "ttc_entered_attention_reference",
                 score_min=0.35, score_max=0.70),
        Scenario("radar urgency boundary", radar_urgent, vision_clear,
                 2, "normal", "ttc_entered_reaction_time_urgency_reference",
                 score_min=0.70, score_max=0.71),
        Scenario("approaching target outside point gate", radar_side_approaching,
                 vision_clear, 0, "normal", "no_warning_event",
                 score_min=0.0, score_max=0.001),
        Scenario("static target inside point gate", radar_static, vision_clear,
                 0, "normal", "no_warning_event", score_min=0.0, score_max=0.001),
        Scenario("vision medium with GPS context", radar_clear, vision_medium,
                 1, "normal", "visual_path_obstacle_attention", gps=gps_fast,
                 score_min=0.50, score_max=0.70),
        Scenario("vision medium with invalid GPS", radar_clear, vision_medium,
                 1, "normal", "visual_path_obstacle_attention", gps=gps_invalid,
                 score_min=0.50, score_max=0.501),
        Scenario("vision-only high while radar fails", unavailable("radar", now_ns, 8),
                 vision_high, 2, "degraded",
                 "visual_tau_entered_urgency_reference",
                 score_min=0.82, score_max=0.83),
        Scenario("radar high cannot be vetoed", radar_urgent, vision_clear,
                 2, "normal", "ttc_entered_reaction_time_urgency_reference",
                 score_min=0.70, score_max=0.71),
        Scenario("vision high wins modality conflict", radar_attention, vision_high,
                 2, "normal", "visual_tau_entered_urgency_reference",
                 score_min=0.82, score_max=0.83),
        Scenario("both warning modalities invalid", unavailable("radar", now_ns, 10),
                 unavailable("vision", now_ns, 10), None, "unknown",
                 "no_usable_modality"),
        Scenario("both warning modalities stale", radar_attention, vision_medium,
                 None, "unknown", "no_usable_modality", now_offset_ms=501.0),
    ]

    rows = []
    for scenario in scenarios:
        result = check_scenario(scenario, now_ns)
        score = "null" if result.risk_score is None else f"{result.risk_score:.4f}"
        level = "unknown" if result.final_level is None else str(result.final_level)
        rows.append((scenario.name, level, score, result.system_status,
                     result.warning_reason))

    transitions = check_state_transitions(now_ns)

    print("| scenario | level | risk score | status | reason |")
    print("|---|---:|---:|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print("\nState transitions:")
    for name, level, behavior in transitions:
        shown_level = "unknown" if level is None else str(level)
        print(f"- {name}: level={shown_level}, behavior={behavior}")
    print("\noffline warning scenario matrix: all tests passed")


if __name__ == "__main__":
    main()
