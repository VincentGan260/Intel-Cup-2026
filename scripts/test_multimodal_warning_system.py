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
from src.fusion.warning_events import ModalityEvent
from src.fusion.warning_system import MultimodalWarningSystem


class Motor:
    def __init__(self): self.calls = []
    def alert_low(self): self.calls.append(0)
    def alert_medium(self): self.calls.append(1)
    def alert_high(self): self.calls.append(2)


def event(source, level, sequence, now_ns, *, usable=True, reason="test"):
    return ModalityEvent(
        source=source, source_id=str(sequence), sequence=sequence,
        capture_monotonic_ns=now_ns, completed_monotonic_ns=now_ns,
        usable=usable, level=level if usable else None, reason=reason,
        status="usable" if usable else "invalid",
    )


def main() -> None:
    motor = Motor()
    system = MultimodalWarningSystem(
        motor=motor, target_stale_ms=500, vision_stale_ms=500,
        release_hold_ms=500, radar_communication_watchdog_ms=2000)
    t0 = time.monotonic_ns()

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
    assert system.snapshot(t0 + 400_000_000)["warning_level"] == 2
    system.publish_radar(event("radar", 0, 5, t0 + 550_000_000))
    assert system.snapshot(t0 + 600_000_000)["warning_level"] == 0
    assert motor.calls[-1] == 0

    # Both stale becomes unknown; recovery to medium alerts immediately.
    assert system.snapshot(t0 + 3_000_000_000)["warning_level"] is None
    system.publish_vision(event("vision", 1, 3, t0 + 3_000_000_001))
    assert system.snapshot(t0 + 3_000_000_001)["warning_level"] == 1
    assert motor.calls[-1] == 1

    # Visual three-point path heuristic uses a same-frame mask without tuning.
    class Det: bbox = (2, 2, 6, 6)
    class Result:
        detections = [Det()]
        drivable_mask = np.zeros((10, 10), dtype=np.uint8)
    Result.drivable_mask[7, 4] = 1
    ve = VisionWarningRule().evaluate(
        Result(), source_frame_id=7, capture_monotonic_ns=t0,
        completed_monotonic_ns=t0 + 1, sequence=8)
    assert ve.level == 1 and ve.usable

    print("multimodal warning system: all tests passed")


if __name__ == "__main__":
    main()
