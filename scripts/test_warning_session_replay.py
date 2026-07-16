"""Dependency-light test for Dashboard warning-session replay."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_warning_session import replay_session
from src.fusion.warning_config import load_warning_rule_config


def recorded_event(level: int, score: float, now_ns: int, sequence: int) -> dict:
    return {
        "source": "radar", "source_id": str(sequence), "sequence": sequence,
        "capture_monotonic_ns": now_ns, "completed_monotonic_ns": now_ns,
        "usable": True, "level": level, "reason": f"radar_level_{level}",
        "risk_score": score, "status": "usable", "details": {},
    }


def main() -> None:
    config = load_warning_rule_config(ROOT / "configs" / "warning_rules.yaml")
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
        session = Path(temporary)
        (session / "session.json").write_text(json.dumps({
            "warning_rule_config": {"effective_parameters": {"radar": {
                **config.section("radar"), "configured_warning_range_m": 20.0}}}
        }), encoding="utf-8")
        rows = []
        for index, (level, score) in enumerate(((0, 0.0), (1, 0.5), (2, 0.8))):
            now_ns = 1_000_000_000 + index * 100_000_000
            rows.append({
                "sample_id": index, "monotonic_ns": now_ns,
                "timestamps": {"risk_decision_monotonic_ns": now_ns},
                "radar": {"valid": True, "targets": []},
                "gps": {"valid": False},
                "risk_decision": {
                    "risk_decision_monotonic_ns": now_ns,
                    "radar_event": recorded_event(level, score, now_ns, index + 1),
                },
            })
        (session / "samples.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        report = replay_session(session, config)
        assert [row["level"] for row in report["rows"]] == [0, 1, 2]
        assert report["recorded_event_row_count"] == 3

        failed = replay_session(session, config, fail_after={"radar": 1})
        assert failed["rows"][0]["level"] == 0
        assert failed["rows"][1]["level"] is None
        assert failed["rows"][2]["level"] is None

    print("warning session replay: all tests passed")


if __name__ == "__main__":
    main()

