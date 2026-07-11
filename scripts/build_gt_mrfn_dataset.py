"""Convert checked Dashboard sessions into grouped GT-MRFN windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_nn.features import load_feature_schema, modality_slices, vectorize_fusion_row

LABELS = {"low": 0, "mid": 1, "high": 2, 0: 0, 1: 1, 2: 2}


def _load_session(path: Path, require_quality: bool) -> tuple[dict, list[dict]]:
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    if require_quality:
        report_path = path / "quality_report.json"
        if not report_path.is_file() or not json.loads(report_path.read_text(encoding="utf-8")).get("passed"):
            raise ValueError(f"session未通过质量检查: {path}")
    rows = [json.loads(line) for line in (path / "samples.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return meta, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build grouped GT-MRFN windows")
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--schema", default="configs/gt_mrfn_features.yaml")
    ap.add_argument("--allow-unchecked", action="store_true")
    args = ap.parse_args()

    schema = load_feature_schema(args.schema)
    slices = modality_slices(schema)
    windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    end_sample_ids: list[int] = []
    for session_path in args.sessions:
        meta, rows = _load_session(session_path.resolve(), not args.allow_unchecked)
        group_id = str(meta.get("group_id") or session_path.name)
        vectors = [vectorize_fusion_row(row, schema) for row in rows]
        for end in range(schema.window_size - 1, len(rows)):
            raw_label = (rows[end].get("label") or {}).get("risk_level")
            if raw_label not in LABELS:
                continue
            window = np.stack(vectors[end - schema.window_size + 1:end + 1]).astype(np.float32)
            windows.append(window)
            labels.append(LABELS[raw_label])
            groups.append(group_id)
            end_sample_ids.append(int(rows[end].get("sample_id", end)))
    if not windows:
        print("没有带有效标签的完整窗口", file=sys.stderr)
        return 2

    x = np.stack(windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": x, "y": np.asarray(labels, dtype=np.int64),
        "group_id": np.asarray(groups), "end_sample_id": np.asarray(end_sample_ids),
        "feature_names": np.asarray(schema.feature_names),
    }
    for modality, sl in slices.items():
        payload[f"X_{modality}"] = x[:, :, sl]
        payload[f"valid_{modality}"] = x[:, :, sl.stop - 1]
    np.savez_compressed(args.output, **payload)
    print(f"Saved {len(windows)} windows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
