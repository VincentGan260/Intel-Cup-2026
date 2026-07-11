"""Analyze JSONL produced by run_dashboard.py --latency-log."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.latency_tracker import LatencyTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sensor-to-motor-command latency")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--warmup", type=int, default=10,
                        help="discard initial samples while models warm up")
    args = parser.parse_args()
    all_rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    discarded = min(max(0, args.warmup), len(all_rows))
    tracker = LatencyTracker()
    tracker.values_ms = [float(row["latency_ms"]) for row in all_rows[discarded:]]
    result = tracker.summary()
    result.update({"source": str(args.jsonl), "warmup_discarded": discarded})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
