"""Validate a Dashboard recording session and write quality_report.json."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


def _stats(values: list[float]) -> dict:
    return {"p50": statistics.median(values) if values else 0.0,
            "p95": _percentile(values, 0.95), "max": max(values) if values else 0.0}


def check_session(session_dir: Path) -> tuple[dict, int]:
    session_dir = session_dir.resolve()
    errors: list[str] = []
    try:
        meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "errors": [f"session.json 无法解析: {exc}"]}, 2
    rows = []
    try:
        with (session_dir / "samples.jsonl").open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception as exc:
                        errors.append(f"samples.jsonl 第 {line_no} 行损坏: {exc}")
    except Exception as exc:
        return {"passed": False, "errors": [f"samples.jsonl 无法读取: {exc}"]}, 2

    ids = [row.get("sample_id") for row in rows]
    if not rows:
        errors.append("录制中没有样本")
    if ids != list(range(len(rows))):
        errors.append("sample_id 不从 0 连续递增")
    if meta.get("sample_count") != len(rows):
        errors.append(f"session sample_count={meta.get('sample_count')}，JSONL 行数={len(rows)}")
    camera_ns = [row.get("timestamps", {}).get("frame_capture_monotonic_ns", 0) for row in rows]
    if any(b <= a for a, b in zip(camera_ns, camera_ns[1:])):
        errors.append("相机主时间戳未严格递增")
    for key in ("radar_read_start_monotonic_ns", "radar_read_end_monotonic_ns",
                "gps_read_start_monotonic_ns", "gps_read_end_monotonic_ns",
                "vision_start_monotonic_ns", "vision_finish_monotonic_ns",
                "record_write_monotonic_ns"):
        values = [row.get("timestamps", {}).get(key, 0) for row in rows]
        nonzero = [value for value in values if value]
        if len(nonzero) != len(values) or any(b <= a for a, b in zip(nonzero, nonzero[1:])):
            errors.append(f"{key} 缺失或未严格递增")

    missing_frames = []
    for row in rows:
        frame = row.get("frame") or {}
        if frame.get("valid") and not (session_dir / frame.get("path", "")).is_file():
            missing_frames.append(row.get("sample_id"))
    if missing_frames:
        errors.append(f"缺少 {len(missing_frames)} 张有效帧图片")

    modalities = {}
    for name in ("frame", "vision", "radar", "gps", "fusion"):
        values = [row.get(name) for row in rows]
        valid = sum(bool(v and (v.get("valid", True) if isinstance(v, dict) else True)) for v in values)
        modalities[name] = {"total": len(rows), "valid": valid,
                            "valid_ratio": valid / len(rows) if rows else 0.0}
    if rows and modalities["vision"]["valid"] == 0:
        errors.append("全部视觉样本无效")
    if rows and modalities["frame"]["valid"] != len(rows):
        errors.append(f"存在 {len(rows) - modalities['frame']['valid']} 个无效图片样本")

    intervals = [(b - a) / 1_000_000.0 for a, b in zip(camera_ns, camera_ns[1:])]
    duration_s = (camera_ns[-1] - camera_ns[0]) / 1e9 if len(camera_ns) > 1 else 0.0
    actual_hz = (len(camera_ns) - 1) / duration_s if duration_s > 0 else 0.0
    vision_latency = [float(row.get("timestamps", {}).get("vision_latency_ms", 0)) for row in rows]
    radar_delta = [float(row.get("timestamps", {}).get("radar_delta_ms", 0)) for row in rows]
    gps_delta = [float(row.get("timestamps", {}).get("gps_delta_ms", 0)) for row in rows]
    thresholds = meta.get("sync_thresholds_ms", {})
    drivable = [float((row.get("vision") or {}).get("drivable_area_ratio", 0)) for row in rows
                if row.get("vision")]
    empty_detections = sum(not (row.get("vision") or {}).get("detections") for row in rows)
    radar_target_frames = sum(bool((row.get("radar") or {}).get("targets")) for row in rows)
    matched_frames = sum((row.get("fusion") or {}).get("vision_radar_count", 0) > 0 for row in rows)
    limits = meta.get("quality_limits", {})
    for modality, limit_name in (("vision", "max_vision_invalid_ratio"),
                                 ("radar", "max_radar_invalid_ratio"),
                                 ("gps", "max_gps_invalid_ratio")):
        invalid_ratio = 1.0 - modalities[modality]["valid_ratio"]
        if invalid_ratio > float(limits.get(limit_name, 1.0)):
            errors.append(f"{modality}无效比例 {invalid_ratio:.1%} 过高")
    radar_stale_ratio = (sum(x > float(thresholds.get("radar_max_delta_ms", 100.0))
                             for x in radar_delta) / len(rows)) if rows else 1.0
    gps_stale_ratio = (sum(x > float(thresholds.get("gps_max_delta_ms", 1000.0))
                           for x in gps_delta) / len(rows)) if rows else 1.0
    if rows and actual_hz < float(limits.get("min_sample_hz", 0.0)):
        errors.append(f"实际采样率 {actual_hz:.2f} Hz 低于最低要求")
    if radar_stale_ratio > float(limits.get("max_radar_stale_ratio", 1.0)):
        errors.append(f"雷达同步超时比例 {radar_stale_ratio:.1%} 过高")
    if gps_stale_ratio > float(limits.get("max_gps_stale_ratio", 1.0)):
        errors.append(f"GPS同步超时比例 {gps_stale_ratio:.1%} 过高")
    if vision_latency and _percentile(vision_latency, 0.95) > float(
            limits.get("max_vision_p95_ms", float("inf"))):
        errors.append("视觉P95推理延迟超过质量阈值")
    for index, row in enumerate(rows):
        ts = row.get("timestamps", {})
        expected = (float(ts.get("vision_finish_monotonic_ns", 0)) -
                    float(ts.get("vision_start_monotonic_ns", 0))) / 1_000_000.0
        if abs(expected - float(ts.get("vision_latency_ms", 0))) > 1.0:
            errors.append(f"样本 {index} 的视觉起止时间与延迟字段不一致")
            break
    report = {
        "passed": not errors, "errors": errors, "sample_count": len(rows),
        "modalities": modalities, "actual_sample_hz": actual_hz,
        "sample_interval_ms": _stats(intervals), "vision_latency_ms": _stats(vision_latency),
        "radar_delta_ms": {**_stats(radar_delta), "over_threshold": sum(
            x > float(thresholds.get("radar_max_delta_ms", 100.0)) for x in radar_delta)},
        "gps_delta_ms": {**_stats(gps_delta), "over_threshold": sum(
            x > float(thresholds.get("gps_max_delta_ms", 1000.0)) for x in gps_delta)},
        "vision": {"empty_detection_ratio": empty_detections / len(rows) if rows else 0.0,
                   "drivable_area_ratio_min": min(drivable) if drivable else 0.0,
                   "drivable_area_ratio_max": max(drivable) if drivable else 0.0,
                   "drivable_area_abnormal_count": sum(x < 0 or x > 1 for x in drivable)},
        "radar_target_frame_ratio": radar_target_frames / len(rows) if rows else 0.0,
        "vision_radar_match_frame_ratio": matched_frames / len(rows) if rows else 0.0,
        "radar_stale_ratio": radar_stale_ratio,
        "gps_stale_ratio": gps_stale_ratio,
    }
    return report, 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    report, code = check_session(args.session_dir)
    report_path = args.session_dir.resolve() / "quality_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[QualityCheck] {'PASS' if code == 0 else 'FAIL'}: {report_path}")
    for error in report.get("errors", []):
        print(f"  - {error}")
    return code


if __name__ == "__main__":
    sys.exit(main())
