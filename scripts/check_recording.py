"""Preflight/quality report for a RiderGuardian recording session."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    if not path.exists():
        return rows, [f"missing: {path.name}"]
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                if line.strip():
                    rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{n}: {exc}")
    return rows, errors


def stream_stats(rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    ts = [int(r["monotonic_ns"]) for r in rows if "monotonic_ns" in r]
    intervals = [(b - a) / 1e6 for a, b in zip(ts, ts[1:]) if b >= a]
    valid = sum(bool(r.get("valid", r.get("data", {}).get("valid", False))) for r in rows)
    read_ms = [float(r["read_duration_ms"]) for r in rows if r.get("read_duration_ms") is not None]
    return {
        "count": len(rows), "valid_count": valid,
        "valid_ratio": round(valid / len(rows), 4) if rows else 0.0,
        "rate_hz": round(len(rows) / duration_s, 2) if duration_s > 0 else 0.0,
        "median_interval_ms": round(statistics.median(intervals), 2) if intervals else None,
        "max_interval_ms": round(max(intervals), 2) if intervals else None,
        "monotonic": all(b >= a for a, b in zip(ts, ts[1:])),
        "read_duration_ms_median": round(statistics.median(read_ms), 2) if read_ms else None,
        "read_duration_ms_max": round(max(read_ms), 2) if read_ms else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check recording completeness and timing")
    ap.add_argument("session", type=Path)
    ap.add_argument("--require-fusion", action="store_true")
    args = ap.parse_args()
    session = args.session.resolve()
    problems: list[str] = []
    streams: dict[str, list[dict[str, Any]]] = {}
    for name in ("frames", "radar", "gps"):
        streams[name], errors = read_jsonl(session / f"{name}.jsonl")
        problems.extend(errors)
        if not streams[name]:
            problems.append(f"{name}: no records")

    all_ts = [int(r["monotonic_ns"]) for rows in streams.values() for r in rows if "monotonic_ns" in r]
    duration_s = (max(all_ts) - min(all_ts)) / 1e9 if len(all_ts) > 1 else 0.0
    report = {"session": str(session), "duration_s": round(duration_s, 3), "streams": {}}
    for name, rows in streams.items():
        report["streams"][name] = stream_stats(rows, duration_s)
        if not report["streams"][name]["monotonic"]:
            problems.append(f"{name}: timestamp moved backwards")

    missing_images = []
    for row in streams["frames"]:
        if row.get("valid") and row.get("path") and not (session / row["path"]).exists():
            missing_images.append(row["path"])
    report["missing_image_count"] = len(missing_images)
    if missing_images:
        problems.append(f"missing image files: {len(missing_images)}")

    fusion, fusion_errors = read_jsonl(session / "fusion.jsonl")
    if args.require_fusion:
        problems.extend(fusion_errors)
        if not fusion:
            problems.append("fusion.jsonl: no records")
    if fusion:
        report["fusion"] = {
            "count": len(fusion),
            "valid_ratio": {
                name: round(sum(bool(r.get(name, {}).get("valid")) for r in fusion) / len(fusion), 4)
                for name in ("radar", "gps")
            },
            "max_abs_delta_ms": {
                name: round(max(abs(float(r[name]["delta_ms"])) for r in fusion if r.get(name, {}).get("delta_ms") is not None), 3)
                for name in ("radar", "gps")
            },
        }

    report["problems"] = problems
    out = session / "quality_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
