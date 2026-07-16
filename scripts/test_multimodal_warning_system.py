"""Dependency-light checks for the competition multimodal warning core."""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.vision_warning_rule import VisionWarningRule
from src.fusion.warning_arbiter import arbitrate_warning_events
from src.fusion.warning_events import ModalityEvent
from src.fusion.warning_system import MultimodalWarningSystem


class Motor:
    def __init__(self): self.calls = []
    def alert_low(self): self.calls.append(0)
    def alert_medium(self): self.calls.append(1)
    def alert_high(self): self.calls.append(2)


def event(source, level, sequence, now_ns, *, usable=True, reason="test",
          risk_score=None, details=None):
    return ModalityEvent(
        source=source, source_id=str(sequence), sequence=sequence,
        capture_monotonic_ns=now_ns, completed_monotonic_ns=now_ns,
        usable=usable, level=level if usable else None, reason=reason,
        risk_score=risk_score if usable else None,
        status="usable" if usable else "invalid",
        details=details or {},
    )


def main() -> None:
    motor = Motor()
    system = MultimodalWarningSystem(
        motor=motor, target_stale_ms=500, vision_stale_ms=500,
        release_hold_ms=500, radar_communication_watchdog_ms=2000)
    t0 = time.monotonic_ns()

    scored_result = arbitrate_warning_events(
        event("radar", 1, 100, t0, risk_score=0.52),
        event("vision", 1, 100, t0, risk_score=0.61),
        now_ns=t0,
    )
    assert scored_result.final_level == 1
    assert scored_result.risk_score == 0.61
    assert scored_result.radar_score == 0.52
    assert scored_result.vision_score == 0.61
    assert scored_result.radar_score_status == "valid"
    assert scored_result.vision_score_status == "valid"
    assert scored_result.decision_monotonic_ns == t0

    future_ns = t0 + 1_000_000_000
    future_high = event("radar", 2, 99, future_ns, risk_score=0.90)
    future_result = arbitrate_warning_events(
        future_high, event("vision", 0, 99, t0, risk_score=0.0),
        now_ns=t0,
    )
    assert future_result.final_level == 0
    assert future_result.radar_status == "future_timestamp"

    reversed_time = ModalityEvent(
        source="vision", source_id="bad-time", sequence=99,
        capture_monotonic_ns=t0, completed_monotonic_ns=t0 - 1,
        usable=True, level=2, reason="vision_high", risk_score=0.90,
    )
    reversed_result = arbitrate_warning_events(
        event("radar", 0, 98, t0, risk_score=0.0), reversed_time,
        now_ns=t0,
    )
    assert reversed_result.final_level == 0
    assert reversed_result.vision_status == "invalid_timestamp_order"

    future_motor = Motor()
    future_system = MultimodalWarningSystem(motor=future_motor)
    future_system.publish_radar(future_high, fast=True)
    assert future_system.snapshot()["warning_level"] is None
    assert future_motor.calls == []

    malformed_high = arbitrate_warning_events(
        event("radar", 2, 100, t0, risk_score=0.40),
        event("vision", 0, 100, t0, risk_score=0.0),
        now_ns=t0,
    )
    assert malformed_high.final_level == 2
    assert malformed_high.radar_score == 0.70
    assert malformed_high.radar_score_status == "clamped_to_level"

    malformed_low = arbitrate_warning_events(
        event("radar", 0, 100, t0, risk_score=0.90),
        event("vision", 0, 100, t0, risk_score=0.0),
        now_ns=t0,
    )
    assert malformed_low.final_level == 0
    assert malformed_low.risk_score is not None and malformed_low.risk_score < 0.35

    missing_high = arbitrate_warning_events(
        event("radar", 2, 100, t0),
        event("vision", 0, 100, t0, risk_score=0.0),
        now_ns=t0,
    )
    assert missing_high.final_level == 2
    assert missing_high.risk_score == 0.70
    assert missing_high.radar_score_status == "missing_fallback"

    vision_components = {"proximity_risk_score": 0.50, "tau_risk_score": 0.0}
    gps_details = {"speed_kmh": 30.0}
    gps_adjusted = arbitrate_warning_events(
        event("radar", 0, 100, t0, risk_score=0.20),
        event("vision", 1, 100, t0, risk_score=0.50,
              details=vision_components),
        event("gps", 0, 100, t0, details=gps_details),
        now_ns=t0,
    )
    assert gps_adjusted.final_level == 1
    assert 0.50 < gps_adjusted.vision_score < 0.70
    assert gps_adjusted.risk_score == gps_adjusted.vision_score
    assert gps_adjusted.gps_speed_factor == 1.25

    gps_neutral = arbitrate_warning_events(
        event("radar", 1, 100, t0, risk_score=0.60),
        event("vision", 1, 100, t0, risk_score=0.50,
              details=vision_components),
        event("gps", 0, 100, t0, usable=False, details=gps_details),
        now_ns=t0,
    )
    assert gps_neutral.risk_score == 0.60
    assert gps_neutral.vision_score == 0.50
    assert gps_neutral.gps_speed_factor == 1.0
    assert gps_neutral.system_status == "normal"

    tau_dominant = arbitrate_warning_events(
        event("radar", 0, 100, t0, risk_score=0.10),
        event("vision", 1, 100, t0, risk_score=0.65,
              details={"proximity_risk_score": 0.40,
                       "tau_risk_score": 0.65}),
        event("gps", 0, 100, t0, details=gps_details),
        now_ns=t0,
    )
    assert tau_dominant.vision_score == 0.65

    visual_high = arbitrate_warning_events(
        event("radar", 0, 100, t0, risk_score=0.10),
        event("vision", 2, 100, t0, risk_score=0.80,
              details={"proximity_risk_score": 0.69,
                       "tau_risk_score": 0.80}),
        event("gps", 0, 100, t0, details=gps_details),
        now_ns=t0,
    )
    assert visual_high.final_level == 2
    assert visual_high.vision_score == 0.80

    stale_gps = arbitrate_warning_events(
        event("radar", 0, 100, t0, risk_score=0.20),
        event("vision", 1, 100, t0, risk_score=0.50,
              details=vision_components),
        event("gps", 0, 100, t0, details=gps_details),
        now_ns=t0 + 1_001_000_000,
        vision_stale_ms=2000,
    )
    assert stale_gps.vision_score == 0.50
    assert stale_gps.gps_status == "stale"

    gps_system = MultimodalWarningSystem(motor=Motor())
    gps_system.publish_vision(event(
        "vision", 1, 1, t0, risk_score=0.50, details=vision_components))
    gps_system.publish_gps(event("gps", 0, 1, t0, details=gps_details))
    gps_snapshot = gps_system.snapshot(t0)
    assert gps_snapshot["warning_level"] == 1
    assert 0.50 < gps_snapshot["risk_score"] < 0.70
    assert gps_snapshot["gps_event"].source == "gps"

    # Fusion/failure matrix: one usable high-risk source cannot be vetoed.
    matrix = (
        (event("radar", 2, 101, t0, reason="radar_high"),
         event("vision", 0, 101, t0, reason="vision_low"),
         2, "normal", "radar_high"),
        (event("radar", 0, 102, t0, reason="radar_low"),
         event("vision", 2, 102, t0, reason="vision_high"),
         2, "normal", "vision_high"),
        (event("radar", 0, 103, t0, usable=False, reason="radar_invalid"),
         event("vision", 2, 103, t0, reason="vision_high"),
         2, "degraded", "vision_high"),
        (event("radar", 2, 104, t0, reason="radar_high"),
         event("vision", 0, 104, t0, usable=False, reason="vision_invalid"),
         2, "degraded", "radar_high"),
        (event("radar", 1, 105, t0, reason="radar_mid"),
         event("vision", 2, 105, t0, reason="vision_high"),
         2, "normal", "vision_high"),
        (event("radar", 0, 106, t0, reason="radar_low"),
         event("vision", 0, 106, t0, reason="vision_low"),
         0, "normal", "no_warning_event"),
        (event("radar", 0, 107, t0, usable=False, reason="radar_invalid"),
         event("vision", 0, 107, t0, usable=False, reason="vision_invalid"),
         None, "unknown", "no_usable_modality"),
    )
    for radar_event, vision_event, expected_level, expected_status, expected_reason in matrix:
        result = arbitrate_warning_events(
            radar_event, vision_event, now_ns=t0,
            target_stale_ms=500, vision_stale_ms=500,
            radar_communication_watchdog_ms=2000,
        )
        assert result.final_level == expected_level
        assert result.system_status == expected_status
        assert result.warning_reason == expected_reason

    stale_result = arbitrate_warning_events(
        event("radar", 2, 108, t0), event("vision", 2, 108, t0),
        now_ns=t0 + 501_000_000,
        target_stale_ms=500, vision_stale_ms=500,
        radar_communication_watchdog_ms=2000,
    )
    assert stale_result.final_level is None
    assert stale_result.risk_score is None
    assert stale_result.system_status == "unknown"

    # A recently completed radar evaluation must not revive an old capture.
    delayed_radar = event("radar", 2, 99, t0 - 600_000_000)
    delayed_radar = ModalityEvent(
        **{**delayed_radar.__dict__, "completed_monotonic_ns": t0})
    delayed_result = arbitrate_warning_events(
        delayed_radar, None, now_ns=t0,
        target_stale_ms=500, vision_stale_ms=500,
        radar_communication_watchdog_ms=2000,
    )
    assert delayed_result.final_level is None
    assert delayed_result.radar_status == "stale"

    system.publish_radar(event("radar", 0, 1, t0))
    system.publish_vision(event("vision", 1, 1, t0))
    assert system.snapshot(t0)["warning_level"] == 1
    assert motor.calls == [1]

    # Same level and a source-set change do not retrigger medium.
    system.publish_radar(event("radar", 1, 2, t0 + 1))
    assert motor.calls == [1]

    # Radar urgent immediately upgrades once; arbitration is idempotent.
    system.publish_radar(event("radar", 2, 3, t0 + 2), fast=True)
    assert system.snapshot(t0 + 2)["warning_level"] == 2
    assert motor.calls == [1, 2]

    # A single clear request starts, but does not finish, downgrade.
    system.publish_radar(event("radar", 0, 4, t0 + 3))
    system.publish_vision(event("vision", 0, 2, t0 + 3))
    held_snapshot = system.snapshot(t0 + 400_000_000)
    assert held_snapshot["warning_level"] == 2
    assert held_snapshot["final_level"] == 2
    assert held_snapshot["risk_score"] is not None
    assert held_snapshot["risk_score"] >= 0.70
    assert held_snapshot["raw_final_level"] == 0
    assert held_snapshot["raw_risk_score"] == 0.0
    assert held_snapshot["risk_score_state"] == "downgrade_held"
    assert held_snapshot["warning_reason"] != "no_warning_event"
    assert held_snapshot["raw_warning_reason"] == "no_warning_event"
    assert held_snapshot["risk_decision_monotonic_ns"] == t0 + 400_000_000
    assert held_snapshot["risk_timestamp_alignment"] == "downgrade_held"
    assert held_snapshot["risk_effective_updated_monotonic_ns"] < (
        held_snapshot["risk_decision_monotonic_ns"])
    for timing in held_snapshot["risk_source_timing"].values():
        assert timing["capture_monotonic_ns"] <= held_snapshot["risk_decision_monotonic_ns"]
        assert timing["completed_monotonic_ns"] <= held_snapshot["risk_decision_monotonic_ns"]
        assert timing["capture_age_ms"] >= 0.0
        assert timing["completion_age_ms"] >= 0.0
    system.publish_radar(event("radar", 0, 5, t0 + 550_000_000),
                         now_ns=t0 + 550_000_000)
    released_snapshot = system.snapshot(t0 + 600_000_000)
    assert released_snapshot["warning_level"] == 0
    assert released_snapshot["risk_score"] == 0.0
    assert released_snapshot["risk_score_state"] == "current"
    assert released_snapshot["risk_timestamp_alignment"] == "as_of_latest_fresh"
    assert released_snapshot["risk_effective_updated_monotonic_ns"] == (
        released_snapshot["risk_decision_monotonic_ns"])
    assert motor.calls[-1] == 0

    # Both stale becomes unknown; recovery to medium alerts immediately.
    unknown_snapshot = system.snapshot(t0 + 3_000_000_000)
    assert unknown_snapshot["warning_level"] is None
    assert unknown_snapshot["risk_score"] is None
    assert unknown_snapshot["risk_score_state"] == "unknown"
    assert unknown_snapshot["risk_source_timing"] == {}
    assert unknown_snapshot["risk_timestamp_alignment"] == "no_usable_warning_modality"
    system.publish_vision(event("vision", 1, 3, t0 + 3_000_000_001),
                          now_ns=t0 + 3_000_000_001)
    assert system.snapshot(t0 + 3_000_000_001)["warning_level"] == 1
    assert motor.calls[-1] == 1

    same_level_system = MultimodalWarningSystem(motor=Motor())
    same_level_system.publish_radar(
        event("radar", 1, 1, t0, risk_score=0.40, reason="radar_mid"))
    assert same_level_system.snapshot(t0)["risk_score"] == 0.40
    same_level_system.publish_radar(
        event("radar", 1, 2, t0 + 1, risk_score=0.60, reason="radar_mid"))
    assert same_level_system.snapshot(t0 + 1)["risk_score"] == 0.60

    order_system = MultimodalWarningSystem(motor=Motor())
    order_now = time.monotonic_ns()
    order_system.publish_radar(event(
        "radar", 1, 2, order_now, risk_score=0.50, reason="radar_mid"))
    order_system.publish_radar(event(
        "radar", 2, 1, order_now, risk_score=0.90, reason="radar_high"))
    assert order_system.snapshot(order_now)["warning_level"] == 1

    vision_order_system = MultimodalWarningSystem(motor=Motor())
    vision_now = time.monotonic_ns()
    vision_order_system.publish_vision(event(
        "vision", 0, 2, vision_now, risk_score=0.0, reason="vision_low"))
    vision_order_system.publish_vision(event(
        "vision", 2, 1, vision_now - 1, risk_score=0.90, reason="vision_high"))
    assert vision_order_system.snapshot(vision_now)["warning_level"] == 0

    recovery_motor = Motor()
    recovery_system = MultimodalWarningSystem(motor=recovery_motor)
    invalid_now = time.monotonic_ns()
    recovery_system.publish_radar(event(
        "radar", 0, 1, invalid_now, usable=False, reason="radar_invalid"))
    recovered_now = time.monotonic_ns()
    recovery_system.publish_radar(event(
        "radar", 2, 2, recovered_now, risk_score=0.80, reason="radar_high"),
        fast=True)
    assert recovery_system.snapshot(recovered_now)["warning_level"] == 2
    assert recovery_motor.calls == [2]

    # A near path target immediately emits medium after corridor filtering.
    class Det:
        class_name = "obstacle"
        bbox = (2, 2, 6, 6.2)
    class Result:
        detections = [Det()]
        drivable_mask = np.zeros((10, 10), dtype=np.uint8)
    Result.drivable_mask[7, 4] = 1
    ve = VisionWarningRule().evaluate(
        Result(), source_frame_id=7, capture_monotonic_ns=t0,
        completed_monotonic_ns=t0 + 1, sequence=8)
    assert ve.level == 1 and ve.usable
    assert ve.risk_score is not None and 0.35 <= ve.risk_score < 0.70

    center_only = np.zeros((10, 10), dtype=np.uint8)
    center_only[7, 4] = 1
    edge_only = np.zeros((10, 10), dtype=np.uint8)
    edge_only[7, 2] = 1
    two_points = np.zeros((10, 10), dtype=np.uint8)
    two_points[7, 2] = 1
    two_points[7, 6] = 1

    assert VisionWarningRule(path_policy="any").path_related(Det.bbox, edge_only)
    assert not VisionWarningRule(path_policy="center").path_related(Det.bbox, edge_only)
    assert VisionWarningRule(path_policy="center").path_related(Det.bbox, center_only)
    assert not VisionWarningRule(path_policy="two_of_three").path_related(Det.bbox, edge_only)
    assert VisionWarningRule(path_policy="two_of_three").path_related(Det.bbox, two_points)

    class SideDet: bbox = (0, 2, 2, 6)
    side_mask = np.zeros((10, 10), dtype=np.uint8)
    side_mask[7, 1] = 1
    assert not VisionWarningRule().path_related(SideDet.bbox, side_mask)

    class FarDet: bbox = (4, 0, 6, 2)
    far_mask = np.ones((10, 10), dtype=np.uint8)
    assert not VisionWarningRule().path_related(FarDet.bbox, far_mask)

    try:
        VisionWarningRule(path_policy="invalid")
        raise AssertionError("invalid vision path policy should fail")
    except ValueError:
        pass

    def visual_event(rule, size, now_ns, sequence, *, bottom_ratio=0.55):
        class SequenceDet:
            class_name = "obstacle"
            bbox = (50.0 - size / 2.0, bottom_ratio * 100.0 - size,
                    50.0 + size / 2.0, bottom_ratio * 100.0)
        class SequenceResult:
            detections = [SequenceDet()]
            drivable_mask = np.ones((100, 100), dtype=np.uint8)
        return rule.evaluate(
            SequenceResult(), source_frame_id=sequence,
            capture_monotonic_ns=now_ns,
            completed_monotonic_ns=now_ns + 1, sequence=sequence,
        )

    medium_rule = VisionWarningRule()
    observing_event = visual_event(medium_rule, 10.0, t0, 20)
    assert observing_event.level == 0
    assert observing_event.risk_score is not None
    assert 0.0 < observing_event.risk_score < 0.35
    assert visual_event(medium_rule, 10.5, t0 + 250_000_000, 21).level == 0
    medium_event = visual_event(medium_rule, 11.3, t0 + 500_000_000, 22)
    assert medium_event.level == 1
    assert 2.5 < medium_event.details["visual_tau_s"] <= 4.0
    assert medium_event.risk_score is not None
    assert 0.35 <= medium_event.risk_score < 0.70

    urgent_rule = VisionWarningRule()
    assert visual_event(urgent_rule, 10.0, t0, 30).level == 0
    assert visual_event(urgent_rule, 11.0, t0 + 250_000_000, 31).level == 0
    urgent_event = visual_event(urgent_rule, 12.5, t0 + 500_000_000, 32)
    assert urgent_event.level == 2
    assert urgent_event.details["visual_tau_s"] <= 2.5
    assert urgent_event.risk_score is not None and urgent_event.risk_score >= 0.70

    vision_high_system = MultimodalWarningSystem(motor=Motor())
    result = vision_high_system.publish_vision(
        urgent_event, now_ns=urgent_event.completed_monotonic_ns)
    assert result.final_level == 2
    assert result.warning_reason == "visual_tau_entered_urgency_reference"
    try:
        VisionWarningRule(corridor_top_width_ratio=0.6,
                          corridor_bottom_width_ratio=0.5)
        raise AssertionError("invalid visual corridor should fail")
    except ValueError:
        pass

    print("multimodal warning system: all tests passed")


if __name__ == "__main__":
    main()
