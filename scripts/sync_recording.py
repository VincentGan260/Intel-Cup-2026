"""Align a recorded session to camera frames using monotonic timestamps."""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def nearest(rows: list[dict[str, Any]], timestamps: list[int], target: int) -> tuple[dict[str, Any] | None, float | None]:
    if not rows:
        return None, None
    pos = bisect.bisect_left(timestamps, target)
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(rows)]
    index = min(candidates, key=lambda i: abs(timestamps[i] - target))
    return rows[index], (timestamps[index] - target) / 1_000_000.0


def compact_sensor(row: dict[str, Any] | None, delta_ms: float | None, tolerance_ms: float) -> dict[str, Any]:
    if row is None:
        return {"sample_id": None, "delta_ms": None, "valid": False, "data": None}
    payload = row.get("data", {})
    within = delta_ms is not None and abs(delta_ms) <= tolerance_ms
    return {
        "sample_id": row.get("sample_id"),
        "monotonic_ns": row.get("monotonic_ns"),
        "delta_ms": round(delta_ms, 3) if delta_ms is not None else None,
        "valid": bool(within and payload.get("valid", False)),
        "within_tolerance": within,
        "data": payload,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Align sensor samples to each camera frame")
    ap.add_argument("session", type=Path)
    ap.add_argument("--radar-ms", type=float, default=100.0)
    ap.add_argument("--gps-ms", type=float, default=1000.0)
    ap.add_argument("--output", default="fusion.jsonl")
    args = ap.parse_args()

    session = args.session.resolve()
    frames = read_jsonl(session / "frames.jsonl")
    streams = {name: read_jsonl(session / f"{name}.jsonl") for name in ("radar", "gps")}
    for name in streams:
        streams[name].sort(key=lambda row: int(row["monotonic_ns"]))
    times = {name: [int(row["monotonic_ns"]) for row in rows] for name, rows in streams.items()}
    tolerances = {"radar": args.radar_ms, "gps": args.gps_ms}

    output = session / args.output
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for frame in frames:
            if not frame.get("valid", False):
                continue
            target = int(frame["monotonic_ns"])
            aligned = {}
            for name in streams:
                row, delta = nearest(streams[name], times[name], target)
                aligned[name] = compact_sensor(row, delta, tolerances[name])
            result = {
                "fusion_id": count,
                "frame_id": frame.get("frame_id"),
                "frame_path": frame.get("path"),
                "wall_time_ns": frame.get("wall_time_ns"),
                "monotonic_ns": target,
                "relative_ms": frame.get("relative_ms"),
                **aligned,
            }
            f.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(f"Aligned {count} frames -> {output}")
    return 0 if count else 2


if __name__ == "__main__":
    raise SystemExit(main())
