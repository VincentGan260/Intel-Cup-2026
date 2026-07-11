"""Regression test: cached radar frames must never inflate latency samples."""
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.latency_tracker import LatencyTracker


with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / "latency.jsonl"
    tracker = LatencyTracker(log, overwrite=True)
    assert tracker.add(sample_start_ns=100, command_dispatch_ns=150,
                       risk_level=2) == 0.00005
    assert tracker.add(sample_start_ns=100, command_dispatch_ns=200,
                       risk_level=2) is None
    assert tracker.add(sample_start_ns=99, command_dispatch_ns=210,
                       risk_level=2) is None
    assert tracker.add(sample_start_ns=300, command_dispatch_ns=400,
                       risk_level=2) == 0.0001
    assert tracker.summary()["count"] == 2
    tracker.close()
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2

print("latency sample dedup: passed")
