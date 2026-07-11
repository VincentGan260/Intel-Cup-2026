"""JSONL logging and percentile summaries for actuator-command latency."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class LatencyTracker:
    def __init__(self, output_path: str | Path | None = None) -> None:
        self.values_ms: list[float] = []
        self.output_path = Path(output_path) if output_path else None
        self._file = None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.output_path.open("a", encoding="utf-8")

    def add(self, *, sample_start_ns: int, command_dispatch_ns: int,
            risk_level: int, frame_id: int = -1) -> float | None:
        if sample_start_ns <= 0 or command_dispatch_ns < sample_start_ns:
            return None
        latency_ms = (command_dispatch_ns - sample_start_ns) / 1_000_000.0
        self.values_ms.append(latency_ms)
        if self._file:
            row = {"wall_time_ns": time.time_ns(),
                   "sample_start_monotonic_ns": sample_start_ns,
                   "motor_command_monotonic_ns": command_dispatch_ns,
                   "latency_ms": round(latency_ms, 3),
                   "risk_level": int(risk_level), "frame_id": int(frame_id)}
            self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._file.flush()
        return latency_ms

    def summary(self) -> dict:
        xs = self.values_ms
        return {"count": len(xs),
                "mean_ms": round(statistics.fmean(xs), 3) if xs else 0.0,
                "p50_ms": round(_percentile(xs, 0.50), 3),
                "p95_ms": round(_percentile(xs, 0.95), 3),
                "p99_ms": round(_percentile(xs, 0.99), 3),
                "max_ms": round(max(xs), 3) if xs else 0.0}

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
