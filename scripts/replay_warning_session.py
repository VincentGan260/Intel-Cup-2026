"""Replay Dashboard JSONL recordings through the competition warning system."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.gps_risk_context import GpsSpeedModifierConfig
from src.fusion.physical_risk_rule import PhysicalRiskRule
from src.fusion.warning_config import WarningRuleConfig, load_warning_rule_config
from src.fusion.warning_events import ModalityEvent
from src.fusion.warning_system import MultimodalWarningSystem
from src.sensors.radar_replay import dict_to_radar


EVENT_FIELDS = {item.name for item in fields(ModalityEvent)}


def _read_jsonl_rows(path: Path) -> list[dict]:
    """Read JSONL while tolerating zero-filled crash padding at EOF."""
    rows: list[dict] = []
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip("\x00 \t\r\n"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL row at {path}:{line_number}: {exc.msg}") from exc
    return rows


def _event_from_dict(data: Optional[dict], *, sequence: int) -> Optional[ModalityEvent]:
    if not isinstance(data, dict):
        return None
    values = {key: value for key, value in data.items() if key in EVENT_FIELDS}
    required = {"source", "capture_monotonic_ns", "completed_monotonic_ns",
                "usable", "level", "reason"}
    if not required.issubset(values):
        return None
    values["source_id"] = str(values.get("source_id", sequence))
    values["sequence"] = sequence
    values["details"] = dict(values.get("details") or {})
    return ModalityEvent(**values)


def _unavailable(source: str, decision_ns: int, sequence: int,
                 reason: str) -> ModalityEvent:
    return ModalityEvent(
        source=source, source_id=str(sequence), sequence=sequence,
        capture_monotonic_ns=decision_ns, completed_monotonic_ns=decision_ns,
        usable=False, level=None, reason=reason, status="invalid",
    )


def _decision_ns(row: dict, previous_ns: int) -> int:
    risk = row.get("risk_decision") or {}
    timestamps = row.get("timestamps") or {}
    value = (risk.get("risk_decision_monotonic_ns")
             or risk.get("decision_monotonic_ns")
             or timestamps.get("risk_decision_monotonic_ns")
             or timestamps.get("radar_read_end_monotonic_ns")
             or row.get("monotonic_ns"))
    decision_ns = int(value or 0)
    if decision_ns <= previous_ns:
        raise ValueError("recording risk decision timestamps must be strictly increasing")
    return decision_ns


def _build_rule(config: WarningRuleConfig, session_meta: dict,
                warning_range_m: Optional[float]) -> PhysicalRiskRule:
    radar = config.section("radar")
    recorded = (((session_meta.get("warning_rule_config") or {})
                 .get("effective_parameters") or {}).get("radar") or {})
    radar.update(recorded)
    configured_range = warning_range_m or radar.get("configured_warning_range_m")
    if configured_range is None:
        raise ValueError(
            "recording has no configured radar range; provide --warning-range-m")
    return PhysicalRiskRule(
        body_width_m=float(radar["body_width_m"]),
        point_gate_lateral_margin_m=float(radar["point_gate_lateral_margin_m"]),
        mounting_offset_m=float(radar["mounting_offset_m"]),
        mounting_uncertainty_m=float(radar["mounting_uncertainty_m"]),
        configured_warning_range_m=float(configured_range),
        radar_parsed_to_motor_go_p95_s=(
            float(radar["radar_parsed_to_motor_go_p95_ms"]) / 1000.0),
        attention_reference_s=float(radar["attention_reference_s"]),
        urgent_reference_s=float(radar["urgent_reference_s"]),
        max_abs_angle_deg=float(radar["max_abs_angle_deg"]),
    )


def _fallback_radar_event(row: dict, rule: PhysicalRiskRule, decision_ns: int,
                          sequence: int) -> ModalityEvent:
    radar_dict = row.get("radar") or {}
    radar = dict_to_radar(radar_dict)
    timestamps = row.get("timestamps") or {}
    capture_ns = int(timestamps.get("radar_sample_monotonic_ns") or decision_ns)
    fresh = bool(radar_dict.get("valid", False))
    return rule.evaluate_event(
        radar, radar_fresh=fresh, sequence=sequence,
        packet_monotonic_ns=capture_ns, completed_monotonic_ns=decision_ns,
    )


def _fallback_gps_event(row: dict, decision_ns: int, sequence: int) -> ModalityEvent:
    gps = row.get("gps") or {}
    timestamps = row.get("timestamps") or {}
    capture_ns = int(timestamps.get("gps_sample_monotonic_ns") or decision_ns)
    usable = bool(gps.get("valid", False))
    return ModalityEvent(
        source="gps", source_id=str(sequence), sequence=sequence,
        capture_monotonic_ns=capture_ns, completed_monotonic_ns=decision_ns,
        usable=usable, level=0 if usable else None,
        reason="gps_speed_context" if usable else "gps_replay_invalid",
        risk_score=None, status="usable" if usable else "invalid",
        details={"speed_kmh": float(gps.get("speed_kmh", 0.0) or 0.0)},
    )


def replay_session(session_dir: Path, config: WarningRuleConfig, *,
                   warning_range_m: Optional[float] = None,
                   fail_after: Optional[dict[str, int]] = None) -> dict:
    session_dir = session_dir.resolve()
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    rows = _read_jsonl_rows(session_dir / "samples.jsonl")
    if not rows:
        raise ValueError("recording contains no samples")

    rule = _build_rule(config, meta, warning_range_m)
    gps = config.section("gps")
    freshness = config.section("freshness")
    has_recorded_imu = any(
        bool((row.get("risk_decision") or {}).get("imu_event")) for row in rows)
    system = MultimodalWarningSystem(
        target_stale_ms=float(freshness["target_stale_ms"]),
        vision_stale_ms=float(freshness["vision_stale_ms"]),
        gps_stale_ms=float(freshness["gps_stale_ms"]),
        imu_stale_ms=float(freshness["imu_stale_ms"]),
        imu_enabled=has_recorded_imu,
        radar_communication_watchdog_ms=float(
            freshness["radar_communication_watchdog_ms"]),
        release_hold_ms=float(config.section("state")["release_hold_ms"]),
        gps_modifier_config=GpsSpeedModifierConfig(
            neutral_below_kmh=float(gps["neutral_below_kmh"]),
            full_effect_kmh=float(gps["full_effect_kmh"]),
            max_factor=float(gps["max_factor"])),
        rule_config_metadata=config.metadata,
    )
    fail_after = fail_after or {}
    output_rows = []
    previous_ns = 0
    previous_level = None
    transition_count = 0
    recorded_event_rows = 0

    for index, row in enumerate(rows):
        sequence = index + 1
        decision_ns = _decision_ns(row, previous_ns)
        previous_ns = decision_ns
        recorded = row.get("risk_decision") or {}
        radar_event = _event_from_dict(recorded.get("radar_event"), sequence=sequence)
        vision_event = _event_from_dict(recorded.get("vision_event"), sequence=sequence)
        gps_event = _event_from_dict(recorded.get("gps_event"), sequence=sequence)
        imu_event = _event_from_dict(recorded.get("imu_event"), sequence=sequence)
        if radar_event is not None or vision_event is not None or imu_event is not None:
            recorded_event_rows += 1
        radar_event = radar_event or _fallback_radar_event(
            row, rule, decision_ns, sequence)
        vision_event = vision_event or _unavailable(
            "vision", decision_ns, sequence, "vision_rule_event_not_recorded")
        gps_event = gps_event or _fallback_gps_event(row, decision_ns, sequence)
        if has_recorded_imu:
            imu_event = imu_event or _unavailable(
                "imu", decision_ns, sequence, "imu_rule_event_not_recorded")

        events = {"radar": radar_event, "vision": vision_event, "gps": gps_event,
                  "imu": imu_event}
        for source, start_index in fail_after.items():
            if index >= start_index:
                events[source] = _unavailable(
                    source, decision_ns, sequence, f"injected_{source}_failure")

        system.publish_radar(events["radar"], fast=False, now_ns=decision_ns)
        system.publish_vision(events["vision"], now_ns=decision_ns)
        system.publish_gps(events["gps"], now_ns=decision_ns)
        if has_recorded_imu and events["imu"] is not None:
            system.publish_imu(events["imu"], now_ns=decision_ns)
        snapshot = system.snapshot(decision_ns)
        level = snapshot["warning_level"]
        if index and level != previous_level:
            transition_count += 1
        previous_level = level
        output_rows.append({
            "sample_id": row.get("sample_id", index),
            "decision_monotonic_ns": decision_ns,
            "level": level,
            "risk_score": snapshot["risk_score"],
            "system_status": snapshot["system_status"],
            "warning_reason": snapshot["warning_reason"],
            "risk_score_state": snapshot["risk_score_state"],
            "radar_status": snapshot["radar_status"],
            "vision_status": snapshot["vision_status"],
            "gps_status": snapshot["gps_status"],
            "imu_status": snapshot["imu_status"],
        })

    counts = Counter("unknown" if row["level"] is None else str(row["level"])
                     for row in output_rows)
    return {
        "session": str(session_dir),
        "warning_rule_config": config.metadata,
        "sample_count": len(output_rows),
        "recorded_event_row_count": recorded_event_rows,
        "level_counts": dict(counts),
        "transition_count": transition_count,
        "fault_injection": fail_after,
        "rows": output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Dashboard warning session")
    parser.add_argument("session", type=Path)
    parser.add_argument("--warning-config", type=Path,
                        default=ROOT / "configs" / "warning_rules.yaml")
    parser.add_argument("--warning-range-m", type=float)
    parser.add_argument("--fail-radar-at", type=int)
    parser.add_argument("--fail-vision-at", type=int)
    parser.add_argument("--fail-gps-at", type=int)
    parser.add_argument("--fail-imu-at", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fail_after = {source: value for source, value in {
        "radar": args.fail_radar_at,
        "vision": args.fail_vision_at,
        "gps": args.fail_gps_at,
        "imu": args.fail_imu_at,
    }.items() if value is not None}
    report = replay_session(
        args.session, load_warning_rule_config(args.warning_config),
        warning_range_m=args.warning_range_m, fail_after=fail_after)
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written: {args.output}")


if __name__ == "__main__":
    main()
