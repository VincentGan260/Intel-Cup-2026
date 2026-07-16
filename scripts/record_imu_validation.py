"""WT61C 风险融合前受控动作采集与统计。

默认按 docs/plan/IMU风险融合前行动清单.md 交互式执行全部场景。
每个动作场景包含 3 秒静止开始、12 秒动作和 3 秒静止结束。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sensors.imu_reader import IMUReader  # noqa: E402


SCENARIOS = (
    ("static", "水平静止 20 秒", 20.0, False),
    ("left_tilt", "缓慢向左倾斜并恢复，重复 3 次", 12.0, True),
    ("right_tilt", "缓慢向右倾斜并恢复，重复 3 次", 12.0, True),
    ("forward_tilt", "缓慢前倾并恢复，重复 3 次", 12.0, True),
    ("backward_tilt", "缓慢后仰并恢复，重复 3 次", 12.0, True),
    ("yaw_left", "保持基本竖直，向左转动车头 3 次", 12.0, True),
    ("yaw_right", "保持基本竖直，向右转动车头 3 次", 12.0, True),
    ("bump", "轻微上下振动或模拟小颠簸 3 次", 12.0, True),
    ("push_straight", "安全推行直线", 15.0, True),
    ("normal_turn", "安全推行普通转弯", 15.0, True),
    ("stop", "安全推行并正常停止，不做故意摔车", 12.0, True),
)

FIELDS = (
    "wall_timestamp", "monotonic_timestamp_ns", "scenario", "phase", "valid",
    "roll", "pitch", "yaw", "gyro_x", "gyro_y", "gyro_z",
    "acc_x", "acc_y", "acc_z", "brake_score", "bump_score", "tilt_score",
    "acc_arrival_monotonic_ns", "gyro_arrival_monotonic_ns", "angle_arrival_monotonic_ns",
    "acc_age_ms", "gyro_age_ms", "angle_age_ms", "component_skew_ms",
    "acc_packet_count", "gyro_packet_count", "angle_packet_count",
    "bad_checksum_count", "discarded_byte_count",
)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=PROJECT_ROOT, check=True, text=True,
                              capture_output=True).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "working_tree_changes": run("git", "status", "--short").splitlines(),
    }


def _row(reader: IMUReader, scenario: str, phase: str) -> dict[str, Any]:
    monotonic_ns = time.monotonic_ns()
    data = reader.read_once()
    diagnostics = reader.get_diagnostics()
    arrivals = diagnostics["component_arrival_monotonic_ns"]
    ages = diagnostics["component_age_ms"]
    counts = diagnostics["packet_counts"]
    row: dict[str, Any] = {
        "wall_timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "monotonic_timestamp_ns": monotonic_ns,
        "scenario": scenario,
        "phase": phase,
        "valid": data.valid,
        "acc_arrival_monotonic_ns": arrivals.get("acc", ""),
        "gyro_arrival_monotonic_ns": arrivals.get("gyro", ""),
        "angle_arrival_monotonic_ns": arrivals.get("angle", ""),
        "acc_age_ms": ages.get("acc", ""),
        "gyro_age_ms": ages.get("gyro", ""),
        "angle_age_ms": ages.get("angle", ""),
        "component_skew_ms": diagnostics["component_skew_ms"] if data.valid else "",
        "acc_packet_count": counts["acc"],
        "gyro_packet_count": counts["gyro"],
        "angle_packet_count": counts["angle"],
        "bad_checksum_count": diagnostics["bad_checksum_count"],
        "discarded_byte_count": diagnostics["discarded_byte_count"],
    }
    # 无效样本的数值列留空，不伪装成“有效的 0”。
    for name in ("roll", "pitch", "yaw", "gyro_x", "gyro_y", "gyro_z",
                 "acc_x", "acc_y", "acc_z", "brake_score", "bump_score", "tilt_score"):
        row[name] = getattr(data, name) if data.valid else ""
    return row


def _capture_phase(reader: IMUReader, writer: csv.DictWriter, rows: list[dict[str, Any]],
                   scenario: str, phase: str, duration: float, interval: float) -> tuple[int, int]:
    start_ns = time.monotonic_ns()
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        row = _row(reader, scenario, phase)
        writer.writerow(row)
        rows.append(row)
        remaining = interval - (time.monotonic() - loop_start)
        if remaining > 0:
            time.sleep(remaining)
    return start_ns, time.monotonic_ns()


def _active_rows(rows: list[dict[str, Any]], scenario: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["scenario"] == scenario
            and row["phase"] in ("active", "static") and row["valid"]]


def _signed_peak(values: list[float], baseline: float) -> float | None:
    if not values:
        return None
    deviations = [value - baseline for value in values]
    return max(deviations, key=abs)


def _axis_consistency(rows: list[dict[str, Any]], scenarios: tuple[str, ...],
                      angle_field: str, gyro_field: str) -> dict[str, Any]:
    angle_rates: list[float] = []
    gyro_values: list[float] = []
    for scenario in scenarios:
        scene_rows = _active_rows(rows, scenario)
        for previous, current in zip(scene_rows, scene_rows[1:]):
            elapsed = (int(current["monotonic_timestamp_ns"])
                       - int(previous["monotonic_timestamp_ns"])) / 1e9
            if elapsed <= 0:
                continue
            angle_rate = (float(current[angle_field]) - float(previous[angle_field])) / elapsed
            gyro_value = float(current[gyro_field])
            # 忽略完全静止的噪声段，避免符号一致率被零值主导。
            if abs(angle_rate) < 0.5 and abs(gyro_value) < 0.5:
                continue
            angle_rates.append(angle_rate)
            gyro_values.append(gyro_value)
    correlation = None
    if len(angle_rates) >= 2:
        angle_mean = statistics.fmean(angle_rates)
        gyro_mean = statistics.fmean(gyro_values)
        numerator = sum((x - angle_mean) * (y - gyro_mean)
                        for x, y in zip(angle_rates, gyro_values))
        denominator = math.sqrt(
            sum((x - angle_mean) ** 2 for x in angle_rates)
            * sum((y - gyro_mean) ** 2 for y in gyro_values)
        )
        if denominator > 0:
            correlation = numerator / denominator
    sign_matches = sum((x > 0) == (y > 0) for x, y in zip(angle_rates, gyro_values)
                       if x != 0 and y != 0)
    comparable = sum(x != 0 and y != 0 for x, y in zip(angle_rates, gyro_values))
    return {
        "angle_axis": angle_field,
        "gyro_axis": gyro_field,
        "samples": len(angle_rates),
        "pearson_correlation": correlation,
        "same_sign_fraction": sign_matches / comparable if comparable else None,
    }


def _longest_high_duration(rows: list[dict[str, Any]], threshold: float = 0.5) -> float:
    longest = current = 0.0
    previous_ns: int | None = None
    for row in rows:
        timestamp_ns = int(row["monotonic_timestamp_ns"])
        if float(row["tilt_score"]) >= threshold:
            if previous_ns is not None:
                current += (timestamp_ns - previous_ns) / 1e9
            longest = max(longest, current)
        else:
            current = 0.0
        previous_ns = timestamp_ns
    return longest


def _summarize(rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["valid"]]
    static_rows = _active_rows(rows, "static")
    new_packet_rows: list[dict[str, Any]] = []
    previous_total = -1
    for row in rows:
        total = sum(int(row[name]) for name in
                    ("acc_packet_count", "gyro_packet_count", "angle_packet_count"))
        if total > previous_total:
            new_packet_rows.append(row)
            previous_total = total
    # 场景间的人工回车等待期间未记录数据，不能误算为传感器断流。
    update_intervals_ms = [
        (int(current["monotonic_timestamp_ns"]) - int(previous["monotonic_timestamp_ns"])) / 1e6
        for previous, current in zip(new_packet_rows, new_packet_rows[1:])
        if (current["scenario"], current["phase"]) == (previous["scenario"], previous["phase"])
    ]
    static_roll = [float(row["roll"]) for row in static_rows]
    static_pitch = [float(row["pitch"]) for row in static_rows]
    roll_baseline = statistics.fmean(static_roll) if static_roll else 0.0
    pitch_baseline = statistics.fmean(static_pitch) if static_pitch else 0.0
    directions = {}
    for scenario, field, fallback_baseline in (
        ("left_tilt", "roll", roll_baseline), ("right_tilt", "roll", roll_baseline),
        ("forward_tilt", "pitch", pitch_baseline), ("backward_tilt", "pitch", pitch_baseline),
    ):
        pre_rows = [row for row in rows if row["scenario"] == scenario
                    and row["phase"] == "pre_static" and row["valid"]]
        baseline = (statistics.fmean(float(row[field]) for row in pre_rows)
                    if pre_rows else fallback_baseline)
        peak = _signed_peak([float(row[field]) for row in _active_rows(rows, scenario)], baseline)
        directions[scenario] = {"axis": field, "pre_static_baseline": baseline,
                                "signed_peak_delta": peak,
                                "sign": None if peak is None else ("positive" if peak > 0 else "negative")}
    direction_pairs_passed = {
        "left_right_roll_opposite": (
            directions["left_tilt"]["signed_peak_delta"] is not None
            and directions["right_tilt"]["signed_peak_delta"] is not None
            and directions["left_tilt"]["signed_peak_delta"]
            * directions["right_tilt"]["signed_peak_delta"] < 0
        ),
        "forward_backward_pitch_opposite": (
            directions["forward_tilt"]["signed_peak_delta"] is not None
            and directions["backward_tilt"]["signed_peak_delta"] is not None
            and directions["forward_tilt"]["signed_peak_delta"]
            * directions["backward_tilt"]["signed_peak_delta"] < 0
        ),
    }
    observed_mapping = {}
    for scenario in ("left_tilt", "right_tilt", "forward_tilt", "backward_tilt"):
        pre_rows = [row for row in rows if row["scenario"] == scenario
                    and row["phase"] == "pre_static" and row["valid"]]
        active_rows = _active_rows(rows, scenario)
        axis_peaks = {}
        for field, fallback in (("roll", roll_baseline), ("pitch", pitch_baseline)):
            baseline = (statistics.fmean(float(row[field]) for row in pre_rows)
                        if pre_rows else fallback)
            axis_peaks[field] = _signed_peak(
                [float(row[field]) for row in active_rows], baseline)
        available = {name: value for name, value in axis_peaks.items() if value is not None}
        dominant_axis = max(available, key=lambda name: abs(available[name])) if available else None
        dominant_delta = available.get(dominant_axis) if dominant_axis else None
        observed_mapping[scenario] = {
            "roll_signed_peak_delta": axis_peaks["roll"],
            "pitch_signed_peak_delta": axis_peaks["pitch"],
            "dominant_axis": dominant_axis,
            "dominant_sign": (None if dominant_delta is None else
                              ("positive" if dominant_delta > 0 else "negative")),
        }
    left_axis = observed_mapping["left_tilt"]["dominant_axis"]
    right_axis = observed_mapping["right_tilt"]["dominant_axis"]
    forward_axis = observed_mapping["forward_tilt"]["dominant_axis"]
    backward_axis = observed_mapping["backward_tilt"]["dominant_axis"]
    observed_mapping_checks = {
        "left_right_same_axis": left_axis is not None and left_axis == right_axis,
        "left_right_opposite_sign": (
            observed_mapping["left_tilt"]["dominant_sign"] is not None
            and observed_mapping["right_tilt"]["dominant_sign"] is not None
            and observed_mapping["left_tilt"]["dominant_sign"]
            != observed_mapping["right_tilt"]["dominant_sign"]
        ),
        "forward_backward_same_axis": forward_axis is not None and forward_axis == backward_axis,
        "forward_backward_opposite_sign": (
            observed_mapping["forward_tilt"]["dominant_sign"] is not None
            and observed_mapping["backward_tilt"]["dominant_sign"] is not None
            and observed_mapping["forward_tilt"]["dominant_sign"]
            != observed_mapping["backward_tilt"]["dominant_sign"]
        ),
        "matches_code_axis_assumption": left_axis == "roll" and forward_axis == "pitch",
    }
    gravity = {
        axis: _stats([float(row[axis]) for row in static_rows])
        for axis in ("acc_x", "acc_y", "acc_z")
    }
    dominant_gravity_axis = None
    gravity_means = {axis: abs(values["mean"]) for axis, values in gravity.items()
                     if values["mean"] is not None}
    if gravity_means:
        dominant_gravity_axis = max(gravity_means, key=gravity_means.get)
    sync_skews = [float(row["component_skew_ms"]) for row in new_packet_rows
                  if row["valid"] and row["component_skew_ms"] not in ("", None)]
    tilt_checks = {}
    for scenario in ("bump", "push_straight", "normal_turn", "stop"):
        scene_rows = _active_rows(rows, scenario)
        high_count = sum(float(row["tilt_score"]) >= 0.5 for row in scene_rows)
        tilt_checks[scenario] = {
            "valid_samples": len(scene_rows),
            "fraction_tilt_score_ge_0_5": high_count / len(scene_rows) if scene_rows else None,
            "longest_continuous_high_sec": _longest_high_duration(scene_rows),
            "max_tilt_score": max((float(row["tilt_score"]) for row in scene_rows), default=None),
        }
    return {
        "samples": {"total": len(rows), "valid": len(valid_rows),
                    "invalid": len(rows) - len(valid_rows),
                    "invalid_ratio": (len(rows) - len(valid_rows)) / len(rows) if rows else None},
        "static": {"roll": _stats(static_roll), "pitch": _stats(static_pitch), "gravity": gravity,
                   "dominant_gravity_axis": dominant_gravity_axis},
        "directions": directions,
        "direction_pairs_passed": direction_pairs_passed,
        "observed_installation_mapping": observed_mapping,
        "observed_mapping_checks": observed_mapping_checks,
        "angle_gyro_consistency": {
            "roll_vs_gyro_x": _axis_consistency(
                rows, ("left_tilt", "right_tilt", "forward_tilt", "backward_tilt"),
                "roll", "gyro_x"),
            "pitch_vs_gyro_y": _axis_consistency(
                rows, ("left_tilt", "right_tilt", "forward_tilt", "backward_tilt"),
                "pitch", "gyro_y"),
        },
        "update_interval_ms": _stats(update_intervals_ms),
        "max_outage_ms": max(update_intervals_ms, default=None),
        "component_sync_skew_ms": _stats(sync_skews),
        "sync_candidate_200ms_passed": bool(sync_skews) and _percentile(sync_skews, 99) <= 200.0,
        "tilt_false_positive_checks": tilt_checks,
        "static_tilt_score": {
            "stats": _stats([float(row["tilt_score"]) for row in static_rows]),
            "fraction_ge_0_5": (
                sum(float(row["tilt_score"]) >= 0.5 for row in static_rows) / len(static_rows)
                if static_rows else None
            ),
            "longest_continuous_high_sec": _longest_high_duration(static_rows),
        },
        "packet_counts": diagnostics["packet_counts"],
        "bad_checksum_count": diagnostics["bad_checksum_count"],
        "discarded_byte_count": diagnostics["discarded_byte_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="WT61C 受控动作采集与统计")
    parser.add_argument("--profile", choices=("windows", "dk2500"), default="dk2500")
    parser.add_argument("--mode", choices=("real", "mock"), default="real")
    parser.add_argument("--automatic", action="store_true", help="不等待每个场景的回车确认")
    parser.add_argument("--scenes", default=",".join(scene[0] for scene in SCENARIOS))
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument("--duration-scale", type=float, default=1.0,
                        help="缩放场景时长，仅用于软件自测；真实验收必须为 1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--installation-notes", default="")
    parser.add_argument("--installation-photo", action="append", default=[])
    parser.add_argument("--analyze-existing", type=Path,
                        help="不连接硬件，重新分析已有采集目录")
    args = parser.parse_args()

    if args.analyze_existing is not None:
        output_dir = args.analyze_existing
        with (output_dir / "imu_raw.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["valid"] = str(row["valid"]).lower() == "true"
        last = rows[-1] if rows else {}
        diagnostics = {
            "packet_counts": {
                "acc": int(last.get("acc_packet_count", 0)),
                "gyro": int(last.get("gyro_packet_count", 0)),
                "angle": int(last.get("angle_packet_count", 0)),
            },
            "bad_checksum_count": int(last.get("bad_checksum_count", 0)),
            "discarded_byte_count": int(last.get("discarded_byte_count", 0)),
        }
        summary = _summarize(rows, diagnostics)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已重新分析: {output_dir / 'summary.json'}")
        return 0

    with (PROJECT_ROOT / "configs/sensor_ports.yaml").open(encoding="utf-8") as handle:
        profile_config = yaml.safe_load(handle)[args.profile]
    imu_config = profile_config.get("imu", {})
    selected = {name.strip() for name in args.scenes.split(",") if name.strip()}
    unknown = selected - {scene[0] for scene in SCENARIOS}
    if unknown:
        parser.error(f"未知场景: {', '.join(sorted(unknown))}")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "runs" / "imu_validation" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "imu_raw.csv"
    scenes_path = output_dir / "scenarios.json"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "metadata.json"

    reader = IMUReader(mode=args.mode, config=imu_config)
    reader.start()
    if args.mode == "real" and not reader.get_diagnostics()["connected"]:
        print(f"ERROR: IMU 串口未连接: {imu_config.get('port')}", file=sys.stderr)
        reader.stop()
        return 2

    rows: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    interval = 1.0 / max(1.0, args.sample_hz)
    try:
        with raw_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for name, instruction, active_seconds, has_boundaries in SCENARIOS:
                if name not in selected:
                    continue
                print(f"\n场景 {name}: {instruction}", flush=True)
                if not args.automatic:
                    input("车辆扶稳且操作人员就位后按回车开始...")
                record: dict[str, Any] = {"scenario": name, "instruction": instruction,
                                          "wall_start": datetime.now().astimezone().isoformat()}
                if has_boundaries:
                    print("  前置静止 3 秒", flush=True)
                    record["pre_start_ns"], record["pre_end_ns"] = _capture_phase(
                        reader, writer, rows, name, "pre_static", 3.0, interval)
                    print(f"  现在执行: {instruction}", flush=True)
                    record["active_start_ns"], record["active_end_ns"] = _capture_phase(
                        reader, writer, rows, name, "active", active_seconds * args.duration_scale, interval)
                    print("  结束动作，保持静止 3 秒", flush=True)
                    record["post_start_ns"], record["post_end_ns"] = _capture_phase(
                        reader, writer, rows, name, "post_static", 3.0, interval)
                else:
                    record["active_start_ns"], record["active_end_ns"] = _capture_phase(
                        reader, writer, rows, name, "static", active_seconds * args.duration_scale, interval)
                record["wall_end"] = datetime.now().astimezone().isoformat()
                scene_records.append(record)
                handle.flush()
    except (KeyboardInterrupt, EOFError):
        print("\n采集被中止，已保留已写入数据。", file=sys.stderr)
    finally:
        diagnostics = reader.get_diagnostics()
        reader.stop()

    summary = _summarize(rows, diagnostics)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "profile": args.profile,
        "imu_config": imu_config,
        "installation_notes": args.installation_notes,
        "installation_photos": args.installation_photo,
        "git": _git_metadata(),
    }
    scenes_path.write_text(json.dumps(scene_records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n采集完成: {output_dir}")
    print(f"  原始数据: {raw_path.name}")
    print(f"  场景时段: {scenes_path.name}")
    print(f"  统计摘要: {summary_path.name}")
    print(f"  配置/提交: {metadata_path.name}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
