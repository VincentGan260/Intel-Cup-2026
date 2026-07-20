#!/usr/bin/env python3
"""Generate a deterministic rule-supervised synthetic risk dataset.

The label engine mirrors the competition warning contract:
  low:    score < 0.35
  medium: 0.35 <= score < 0.70
  high:   score >= 0.70

Final rule score is the maximum usable modality score (no veto / no weighted
average).  The acceleration-change modality is an explicit provisional
extension requested for the XGBoost prototype.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "xgb"
SEED = 20260719

ATTENTION_SCORE = 0.35
HIGH_SCORE = 0.70
RADAR_ATTENTION_TTC_S = 4.000469
RADAR_URGENT_TTC_S = 2.500469
VISION_ATTENTION_TAU_S = 4.0
VISION_URGENT_TAU_S = 2.5
VISION_NEAR_BOTTOM_RATIO = 0.61
VISION_VERY_NEAR_BOTTOM_RATIO = 0.82

IMU_ATTENTION_ERROR_DEG = 10.0
IMU_CRITICAL_ERROR_DEG = 25.0
IMU_ATTENTION_OUTWARD_RATE_DEG_S = 5.0
IMU_URGENT_OUTWARD_RATE_DEG_S = 10.0
IMU_ATTENTION_PERSISTENCE_MS = 150.0
IMU_PREDICTION_HORIZON_S = 0.8
IMU_URGENT_MIN_ERROR_DEG = 8.0
IMU_URGENT_CONSISTENT_SAMPLES = 3

MOTION_ATTENTION_DELTA_MPS2 = 1.5
MOTION_HIGH_DELTA_MPS2 = 5.0
MOTION_ATTENTION_JERK_MPS3 = 5.0
MOTION_HIGH_JERK_MPS3 = 10.0

SPLIT_QUOTAS = {
    "train": {0: 2100, 1: 1837, 2: 1313},
    "validation": {0: 450, 1: 394, 2: 281},
    "test": {0: 450, 1: 394, 2: 281},
}

LABEL_NAMES = {0: "低", 1: "中", 2: "高"}

LOW_SCENARIOS = (
    "normal_straight",
    "normal_turn",
    "distant_obstacle_off_path",
    "visual_path_candidate",
    "radar_receding",
    "sensor_gap",
)
MEDIUM_SCENARIOS = (
    "road_bump",
    "moderate_acc_change",
    "sustained_outward_lean",
    "radar_attention",
    "vision_path_near",
    "vision_looming_attention",
)
HIGH_SCENARIOS = (
    "severe_acc_change",
    "predicted_lateral_instability",
    "radar_urgent",
    "vision_looming_urgent",
    "multisensor_high",
)
SCENARIOS_BY_LABEL = {
    0: LOW_SCENARIOS,
    1: MEDIUM_SCENARIOS,
    2: HIGH_SCENARIOS,
}

FIELDNAMES = [
    "sample_id", "split", "scenario_instance_id", "scenario_type",
    "timestamp_s", "window_duration_s", "boundary_case",
    "gps_valid", "gps_speed_kmh", "imu_valid", "pitch_abs_deg", "roll_abs_deg",
    "roll_error_deg", "outward_rate_deg_s", "imu_attention_duration_ms",
    "imu_urgent_consistent_samples", "acc_norm_mean_mps2",
    "acc_delta_signed_mps2", "acc_change_abs_mps2", "jerk_abs_mps3",
    "radar_valid", "radar_target_count", "radar_path_target_count",
    "radar_min_distance_m", "radar_relative_speed_mps",
    "radar_closing_speed_mps", "radar_ttc_s",
    "vision_valid", "object_count", "path_object_count",
    "max_path_bottom_ratio", "box_growth_rate_per_s",
    "growth_duration_s", "visual_tau_s", "vision_confidence",
    "imu_rule_score", "motion_rule_score", "radar_rule_score",
    "vision_rule_score", "rule_score", "risk_label", "risk_label_name",
    "hard_rule_triggered", "trigger_reason",
]

FEATURE_COLUMNS = [
    "gps_valid", "gps_speed_kmh", "imu_valid", "pitch_abs_deg", "roll_abs_deg",
    "roll_error_deg", "outward_rate_deg_s", "imu_attention_duration_ms",
    "imu_urgent_consistent_samples", "acc_norm_mean_mps2",
    "acc_delta_signed_mps2", "acc_change_abs_mps2", "jerk_abs_mps3",
    "radar_valid", "radar_target_count", "radar_path_target_count",
    "radar_min_distance_m", "radar_relative_speed_mps",
    "radar_closing_speed_mps", "radar_ttc_s",
    "vision_valid", "object_count", "path_object_count",
    "max_path_bottom_ratio", "box_growth_rate_per_s",
    "growth_duration_s", "visual_tau_s", "vision_confidence",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def uniform(rng: random.Random, low: float, high: float, digits: int = 4) -> float:
    return round(rng.uniform(low, high), digits)


def near_threshold(
    rng: random.Random, threshold: float, *, below: bool, width: float
) -> float:
    delta = rng.uniform(max(width * 0.05, 1e-5), width)
    return round(threshold - delta if below else threshold + delta, 4)


def base_sample(rng: random.Random) -> dict[str, Any]:
    acc_delta = uniform(rng, -0.9, 0.9)
    return {
        "timestamp_s": uniform(rng, 0.0, 1800.0, 3),
        "window_duration_s": 1.0,
        "boundary_case": 0,
        "gps_valid": 1,
        "gps_speed_kmh": uniform(rng, 5.0, 24.0),
        "imu_valid": 1,
        "pitch_abs_deg": uniform(rng, 0.0, 4.0),
        "roll_abs_deg": uniform(rng, 0.0, 7.0),
        "roll_error_deg": uniform(rng, 0.0, 4.5),
        "outward_rate_deg_s": uniform(rng, 0.0, 3.0),
        "imu_attention_duration_ms": uniform(rng, 0.0, 120.0, 2),
        "imu_urgent_consistent_samples": rng.randint(0, 1),
        "acc_norm_mean_mps2": uniform(rng, 9.55, 10.05),
        "acc_delta_signed_mps2": acc_delta,
        "acc_change_abs_mps2": round(abs(acc_delta), 4),
        "jerk_abs_mps3": uniform(rng, 0.0, 3.8),
        "radar_valid": 1,
        "radar_target_count": 0,
        "radar_path_target_count": 0,
        "radar_min_distance_m": None,
        "radar_relative_speed_mps": 0.0,
        "radar_closing_speed_mps": 0.0,
        "radar_ttc_s": None,
        "vision_valid": 1,
        "object_count": 0,
        "path_object_count": 0,
        "max_path_bottom_ratio": 0.0,
        "box_growth_rate_per_s": 0.0,
        "growth_duration_s": 0.0,
        "visual_tau_s": None,
        "vision_confidence": 0.0,
    }

def set_radar_approach(
    sample: dict[str, Any],
    rng: random.Random,
    ttc_s: float,
    *,
    path: bool,
) -> None:
    closing = uniform(rng, 0.45, 3.2)
    distance = max(0.15, closing * ttc_s)
    sample.update({
        "radar_target_count": rng.randint(1, 4),
        "radar_path_target_count": 1 if path else 0,
        "radar_min_distance_m": round(distance, 4),
        # Project radar code defines approaching velocity as negative.
        "radar_relative_speed_mps": round(-closing, 4),
        "radar_closing_speed_mps": closing,
        "radar_ttc_s": round(ttc_s, 4),
    })


def set_visual_target(
    sample: dict[str, Any],
    rng: random.Random,
    *,
    path: bool,
    bottom_ratio: float,
    visual_tau_s: float | None,
) -> None:
    count = rng.randint(1, 5)
    growth_rate = 0.0
    duration = 0.0
    if visual_tau_s is not None:
        growth_rate = clamp(1.0 / max(visual_tau_s, 0.1), 0.0, 1.5)
        growth_rate = round(growth_rate * rng.uniform(0.85, 1.15), 4)
        duration = uniform(rng, 0.2, 1.0)
    sample.update({
        "object_count": count,
        "path_object_count": rng.randint(1, min(3, count)) if path else 0,
        "max_path_bottom_ratio": round(bottom_ratio if path else 0.0, 4),
        "box_growth_rate_per_s": growth_rate,
        "growth_duration_s": duration,
        "visual_tau_s": round(visual_tau_s, 4) if visual_tau_s is not None else None,
        "vision_confidence": uniform(rng, 0.62, 0.98),
    })


def generate_scenario(
    rng: random.Random, scenario: str, target_label: int, boundary: bool
) -> dict[str, Any]:
    sample = base_sample(rng)
    sample["boundary_case"] = int(boundary)

    if scenario == "normal_straight":
        pass
    elif scenario == "normal_turn":
        sample["roll_abs_deg"] = uniform(rng, 8.0, 27.0)
        sample["roll_error_deg"] = uniform(rng, 0.5, 6.0)
        sample["outward_rate_deg_s"] = uniform(rng, 0.0, 4.5)
        sample["gps_speed_kmh"] = uniform(rng, 8.0, 25.0)
    elif scenario == "distant_obstacle_off_path":
        ttc = uniform(rng, 6.2, 12.0)
        set_radar_approach(sample, rng, ttc, path=False)
        set_visual_target(
            sample, rng, path=False,
            bottom_ratio=uniform(rng, 0.25, 0.55), visual_tau_s=None,
        )
    elif scenario == "visual_path_candidate":
        bottom = (
            near_threshold(rng, VISION_NEAR_BOTTOM_RATIO, below=True, width=0.02)
            if boundary else uniform(rng, 0.41, 0.58)
        )
        set_visual_target(
            sample, rng, path=True, bottom_ratio=bottom, visual_tau_s=None,
        )
    elif scenario == "radar_receding":
        distance = uniform(rng, 1.0, 20.0)
        speed = uniform(rng, 0.2, 2.5)
        sample.update({
            "radar_target_count": rng.randint(1, 3),
            "radar_path_target_count": rng.randint(0, 1),
            "radar_min_distance_m": distance,
            "radar_relative_speed_mps": speed,
            "radar_closing_speed_mps": 0.0,
        })
    elif scenario == "sensor_gap":
        missing = rng.choice(("imu", "radar", "vision"))
        sample[f"{missing}_valid"] = 0
    elif scenario == "road_bump":
        delta = (
            near_threshold(rng, MOTION_ATTENTION_DELTA_MPS2, below=False, width=0.2)
            if boundary else uniform(rng, 1.6, 3.6)
        )
        delta *= rng.choice((-1, 1))
        sample["acc_delta_signed_mps2"] = round(delta, 4)
        sample["acc_change_abs_mps2"] = round(abs(delta), 4)
        sample["jerk_abs_mps3"] = uniform(rng, 5.1, 8.8)
        sample["acc_norm_mean_mps2"] = uniform(rng, 8.7, 10.9)
    elif scenario == "moderate_acc_change":
        delta = (
            near_threshold(rng, MOTION_HIGH_DELTA_MPS2, below=True, width=0.25)
            if boundary else uniform(rng, 1.8, 4.7)
        )
        delta *= rng.choice((-1, 1))
        sample["acc_delta_signed_mps2"] = round(delta, 4)
        sample["acc_change_abs_mps2"] = round(abs(delta), 4)
        sample["jerk_abs_mps3"] = uniform(rng, 5.2, 9.7)
    elif scenario == "sustained_outward_lean":
        sample["roll_error_deg"] = (
            near_threshold(rng, IMU_ATTENTION_ERROR_DEG, below=False, width=0.5)
            if boundary else uniform(rng, 10.5, 18.0)
        )
        sample["roll_abs_deg"] = max(
            sample["roll_abs_deg"], sample["roll_error_deg"] + uniform(rng, 0.0, 5.0)
        )
        sample["outward_rate_deg_s"] = uniform(rng, 5.2, 9.5)
        sample["imu_attention_duration_ms"] = (
            near_threshold(rng, IMU_ATTENTION_PERSISTENCE_MS, below=False, width=8.0)
            if boundary else uniform(rng, 170.0, 650.0)
        )
    elif scenario == "radar_attention":
        ttc = (
            near_threshold(rng, RADAR_ATTENTION_TTC_S, below=True, width=0.08)
            if boundary else uniform(rng, RADAR_URGENT_TTC_S + 0.08, 3.85)
        )
        set_radar_approach(sample, rng, ttc, path=True)
    elif scenario == "vision_path_near":
        bottom = (
            near_threshold(rng, VISION_NEAR_BOTTOM_RATIO, below=False, width=0.015)
            if boundary else uniform(rng, 0.63, 0.80)
        )
        set_visual_target(
            sample, rng, path=True, bottom_ratio=bottom, visual_tau_s=None,
        )
    elif scenario == "vision_looming_attention":
        tau = (
            near_threshold(rng, VISION_ATTENTION_TAU_S, below=True, width=0.08)
            if boundary else uniform(rng, VISION_URGENT_TAU_S + 0.08, 3.85)
        )
        set_visual_target(
            sample, rng, path=True,
            bottom_ratio=uniform(rng, 0.45, 0.79), visual_tau_s=tau,
        )
    elif scenario == "severe_acc_change":
        delta = (
            near_threshold(rng, MOTION_HIGH_DELTA_MPS2, below=False, width=0.25)
            if boundary else uniform(rng, 5.2, 9.0)
        )
        delta *= rng.choice((-1, 1))
        sample["acc_delta_signed_mps2"] = round(delta, 4)
        sample["acc_change_abs_mps2"] = round(abs(delta), 4)
        sample["jerk_abs_mps3"] = uniform(rng, 10.2, 18.0)
    elif scenario == "predicted_lateral_instability":
        error = uniform(rng, 8.2, 22.0)
        min_rate = max(10.2, (IMU_CRITICAL_ERROR_DEG - error) / 0.76)
        sample["roll_error_deg"] = error
        sample["roll_abs_deg"] = error + uniform(rng, 1.0, 8.0)
        sample["outward_rate_deg_s"] = uniform(rng, min_rate, min_rate + 12.0)
        sample["imu_attention_duration_ms"] = uniform(rng, 180.0, 800.0)
        sample["imu_urgent_consistent_samples"] = 3
        sample["gps_speed_kmh"] = uniform(rng, 4.0, 26.0)
    elif scenario == "radar_urgent":
        ttc = (
            near_threshold(rng, RADAR_URGENT_TTC_S, below=True, width=0.08)
            if boundary else uniform(rng, 0.45, 2.35)
        )
        set_radar_approach(sample, rng, ttc, path=True)
    elif scenario == "vision_looming_urgent":
        tau = (
            near_threshold(rng, VISION_URGENT_TAU_S, below=True, width=0.08)
            if boundary else uniform(rng, 0.45, 2.35)
        )
        set_visual_target(
            sample, rng, path=True,
            bottom_ratio=uniform(rng, 0.62, 0.95), visual_tau_s=tau,
        )
    elif scenario == "multisensor_high":
        ttc = uniform(rng, 0.6, 2.2)
        set_radar_approach(sample, rng, ttc, path=True)
        set_visual_target(
            sample, rng, path=True,
            bottom_ratio=uniform(rng, 0.68, 0.96),
            visual_tau_s=uniform(rng, 0.6, 2.2),
        )
        if rng.random() < 0.5:
            delta = uniform(rng, 2.0, 6.5) * rng.choice((-1, 1))
            sample["acc_delta_signed_mps2"] = round(delta, 4)
            sample["acc_change_abs_mps2"] = round(abs(delta), 4)
            sample["jerk_abs_mps3"] = uniform(rng, 6.0, 14.0)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    # GPS availability is independent of the road-risk scenario.  This gives
    # every risk band both valid-GPS and indoor/no-fix examples instead of
    # teaching the model that GPS loss always means low risk.
    if rng.random() < 0.30:
        sample["gps_valid"] = 0
        sample["gps_speed_kmh"] = None

    return sample


def imu_score(sample: dict[str, Any]) -> float:
    if not sample["imu_valid"]:
        return 0.0
    error = abs(float(sample["roll_error_deg"]))
    outward = max(0.0, float(sample["outward_rate_deg_s"]))
    predicted = error + outward * IMU_PREDICTION_HORIZON_S
    time_to_critical = (
        max(0.0, (IMU_CRITICAL_ERROR_DEG - error) / outward)
        if outward > 0 else None
    )
    urgent = (
        bool(sample["gps_valid"])
        and sample["gps_speed_kmh"] is not None
        and sample["gps_speed_kmh"] >= 3.0
        and error >= IMU_URGENT_MIN_ERROR_DEG
        and outward >= IMU_URGENT_OUTWARD_RATE_DEG_S
        and time_to_critical is not None
        and time_to_critical <= IMU_PREDICTION_HORIZON_S
        and sample["imu_urgent_consistent_samples"] >= IMU_URGENT_CONSISTENT_SAMPLES
    )
    if urgent:
        urgency = 1.0 - min(1.0, time_to_critical / IMU_PREDICTION_HORIZON_S)
        return clamp(HIGH_SCORE + (1.0 - HIGH_SCORE) * urgency)
    attention = (
        error >= IMU_ATTENTION_ERROR_DEG
        and outward >= IMU_ATTENTION_OUTWARD_RATE_DEG_S
        and sample["imu_attention_duration_ms"] >= IMU_ATTENTION_PERSISTENCE_MS
    )
    if attention:
        progress = clamp(
            (predicted - IMU_ATTENTION_ERROR_DEG)
            / (IMU_CRITICAL_ERROR_DEG - IMU_ATTENTION_ERROR_DEG)
        )
        return HIGH_SCORE if progress >= 1.0 else ATTENTION_SCORE + 0.35 * progress
    return min(math.nextafter(ATTENTION_SCORE, 0.0), 0.35 * predicted / 10.0)


def motion_score(sample: dict[str, Any]) -> float:
    if not sample["imu_valid"]:
        return 0.0
    delta = float(sample["acc_change_abs_mps2"])
    jerk = float(sample["jerk_abs_mps3"])
    delta_level = (
        2 if delta >= MOTION_HIGH_DELTA_MPS2 else
        1 if delta >= MOTION_ATTENTION_DELTA_MPS2 else 0
    )
    jerk_level = (
        2 if jerk >= MOTION_HIGH_JERK_MPS3 else
        1 if jerk >= MOTION_ATTENTION_JERK_MPS3 else 0
    )
    level = max(delta_level, jerk_level)
    if level == 2:
        progress = max(
            clamp((delta - MOTION_HIGH_DELTA_MPS2) / 5.0),
            clamp((jerk - MOTION_HIGH_JERK_MPS3) / 10.0),
        )
        return HIGH_SCORE + 0.30 * progress
    if level == 1:
        progress = max(
            clamp((delta - MOTION_ATTENTION_DELTA_MPS2)
                  / (MOTION_HIGH_DELTA_MPS2 - MOTION_ATTENTION_DELTA_MPS2)),
            clamp((jerk - MOTION_ATTENTION_JERK_MPS3)
                  / (MOTION_HIGH_JERK_MPS3 - MOTION_ATTENTION_JERK_MPS3)),
        )
        return min(math.nextafter(HIGH_SCORE, 0.0), ATTENTION_SCORE + 0.35 * progress)
    progress = max(
        delta / MOTION_ATTENTION_DELTA_MPS2,
        jerk / MOTION_ATTENTION_JERK_MPS3,
    )
    return min(math.nextafter(ATTENTION_SCORE, 0.0), ATTENTION_SCORE * progress)


def radar_ttc_score(ttc_s: float) -> float:
    if ttc_s <= RADAR_URGENT_TTC_S:
        urgency = 1.0 - ttc_s / RADAR_URGENT_TTC_S
        return clamp(HIGH_SCORE + 0.30 * urgency)
    if ttc_s <= RADAR_ATTENTION_TTC_S:
        progress = (
            (RADAR_ATTENTION_TTC_S - ttc_s)
            / (RADAR_ATTENTION_TTC_S - RADAR_URGENT_TTC_S)
        )
        return ATTENTION_SCORE + 0.35 * progress
    return ATTENTION_SCORE * RADAR_ATTENTION_TTC_S / ttc_s


def radar_score(sample: dict[str, Any]) -> float:
    if not sample["radar_valid"] or sample["radar_target_count"] <= 0:
        return 0.0
    ttc = sample["radar_ttc_s"]
    if ttc is None or sample["radar_closing_speed_mps"] <= 0:
        return 0.0
    if sample["radar_path_target_count"] > 0:
        return clamp(radar_ttc_score(float(ttc)))
    return min(math.nextafter(ATTENTION_SCORE, 0.0), radar_ttc_score(float(ttc)))


def vision_score(sample: dict[str, Any]) -> float:
    if not sample["vision_valid"] or sample["path_object_count"] <= 0:
        return 0.0
    bottom = clamp(float(sample["max_path_bottom_ratio"]))
    if bottom < VISION_NEAR_BOTTOM_RATIO:
        span = VISION_NEAR_BOTTOM_RATIO - 0.40
        proximity = ATTENTION_SCORE * clamp((bottom - 0.40) / span)
    else:
        span = VISION_VERY_NEAR_BOTTOM_RATIO - VISION_NEAR_BOTTOM_RATIO
        progress = clamp((bottom - VISION_NEAR_BOTTOM_RATIO) / span)
        proximity = min(math.nextafter(HIGH_SCORE, 0.0),
                        ATTENTION_SCORE + 0.35 * progress)
    tau_score = 0.0
    tau = sample["visual_tau_s"]
    if tau is not None and tau > 0:
        if tau <= VISION_URGENT_TAU_S:
            urgency = 1.0 - tau / VISION_URGENT_TAU_S
            tau_score = HIGH_SCORE + 0.30 * urgency
        elif tau <= VISION_ATTENTION_TAU_S:
            progress = (
                (VISION_ATTENTION_TAU_S - tau)
                / (VISION_ATTENTION_TAU_S - VISION_URGENT_TAU_S)
            )
            tau_score = ATTENTION_SCORE + 0.35 * progress
        else:
            tau_score = ATTENTION_SCORE * VISION_ATTENTION_TAU_S / tau
    return clamp(max(proximity, tau_score))


def apply_label(sample: dict[str, Any]) -> dict[str, Any]:
    scores = {
        "imu": imu_score(sample),
        "motion": motion_score(sample),
        "radar": radar_score(sample),
        "vision": vision_score(sample),
    }
    final_score = max(scores.values())
    label = 2 if final_score >= HIGH_SCORE else 1 if final_score >= ATTENTION_SCORE else 0
    max_sources = [name for name, value in scores.items()
                   if math.isclose(value, final_score, abs_tol=1e-9)]
    reason = "+".join(max_sources)
    sample.update({
        "imu_rule_score": round(scores["imu"], 6),
        "motion_rule_score": round(scores["motion"], 6),
        "radar_rule_score": round(scores["radar"], 6),
        "vision_rule_score": round(scores["vision"], 6),
        "rule_score": round(final_score, 6),
        "risk_label": label,
        "risk_label_name": LABEL_NAMES[label],
        "hard_rule_triggered": int(label == 2),
        "trigger_reason": reason,
    })
    return sample


def make_split(
    rng: random.Random, split: str, quotas: dict[int, int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boundary_probability = 0.12 if split == "train" else 0.34
    for target_label, count in quotas.items():
        accepted = 0
        attempts = 0
        while accepted < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError(f"could not fill {split} label {target_label}")
            scenario = rng.choice(SCENARIOS_BY_LABEL[target_label])
            boundary = rng.random() < boundary_probability
            sample = apply_label(generate_scenario(
                rng, scenario, target_label, boundary
            ))
            if sample["risk_label"] != target_label:
                continue
            sample["split"] = split
            sample["scenario_type"] = scenario
            sample["scenario_instance_id"] = (
                f"{split[:3]}-{target_label}-{accepted + 1:05d}"
            )
            rows.append(sample)
            accepted += 1
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["sample_id"] = f"{split[:3]}-{index:05d}"
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def validate(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids: set[str] = set()
    report: dict[str, Any] = {"seed": SEED, "splits": {}, "checks": {}}
    for split, rows in splits.items():
        expected = SPLIT_QUOTAS[split]
        actual = Counter(int(row["risk_label"]) for row in rows)
        assert actual == Counter(expected), (split, actual, expected)
        assert len(rows) == sum(expected.values())
        for row in rows:
            assert row["sample_id"] not in ids
            ids.add(row["sample_id"])
            score = float(row["rule_score"])
            expected_label = 2 if score >= 0.70 else 1 if score >= 0.35 else 0
            assert expected_label == int(row["risk_label"])
            assert math.isclose(
                score,
                max(float(row[key]) for key in (
                    "imu_rule_score", "motion_rule_score",
                    "radar_rule_score", "vision_rule_score",
                )),
                abs_tol=1e-6,
            )
        report["splits"][split] = {
            "rows": len(rows),
            "class_counts": {
                LABEL_NAMES[label]: actual[label] for label in (0, 1, 2)
            },
            "boundary_cases": sum(int(row["boundary_case"]) for row in rows),
            "scenario_counts": dict(sorted(Counter(
                row["scenario_type"] for row in rows
            ).items())),
        }

    all_rows = [row for rows in splits.values() for row in rows]
    normal = [row for row in all_rows if row["scenario_type"] == "normal_straight"]
    severe_acc = [row for row in all_rows if row["scenario_type"] == "severe_acc_change"]
    no_radar = [row for row in all_rows if row["radar_target_count"] == 0]
    urgent_path_radar = [row for row in all_rows
                         if row["radar_path_target_count"] > 0
                         and row["radar_ttc_s"] is not None
                         and row["radar_ttc_s"] <= RADAR_URGENT_TTC_S]
    no_path = [row for row in all_rows if row["path_object_count"] == 0]
    looming = [row for row in all_rows
               if row["path_object_count"] > 0
               and row["visual_tau_s"] is not None
               and row["visual_tau_s"] <= VISION_URGENT_TAU_S]
    checks = {
        "all_score_bands_match_labels": True,
        "all_final_scores_equal_modality_max": True,
        "sample_ids_unique": len(ids) == len(all_rows),
        "positive_and_negative_severe_acceleration_present": (
            any(row["acc_delta_signed_mps2"] > 0 for row in severe_acc)
            and any(row["acc_delta_signed_mps2"] < 0 for row in severe_acc)
        ),
        "severe_acceleration_mean_risk_exceeds_normal": (
            mean(row["rule_score"] for row in severe_acc)
            > mean(row["rule_score"] for row in normal)
        ),
        "urgent_path_radar_mean_risk_exceeds_no_radar_target": (
            mean(row["rule_score"] for row in urgent_path_radar)
            > mean(row["rule_score"] for row in no_radar)
        ),
        "urgent_visual_looming_mean_risk_exceeds_no_path_object": (
            mean(row["rule_score"] for row in looming)
            > mean(row["rule_score"] for row in no_path)
        ),
    }
    assert all(checks.values()), checks
    report["checks"] = checks
    report["totals"] = {
        "rows": len(all_rows),
        "class_counts": {
            LABEL_NAMES[label]: sum(
                int(row["risk_label"]) == label for row in all_rows
            )
            for label in (0, 1, 2)
        },
    }
    return report


def write_support_files(report: dict[str, Any]) -> None:
    feature_config = {
        "target": "risk_label",
        "target_names": LABEL_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": [
            name for name in FIELDNAMES
            if name not in FEATURE_COLUMNS and name != "risk_label"
        ],
        "missing_value_fields": ["radar_min_distance_m", "radar_ttc_s", "visual_tau_s"],
        "categorical_metadata_not_for_training": ["scenario_type"],
        "random_seed": SEED,
    }
    (OUTPUT_DIR / "feature_config.json").write_text(
        json.dumps(feature_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "feature_columns.txt").write_text(
        "\n".join(FEATURE_COLUMNS) + "\n", encoding="utf-8"
    )

    dictionary_rows = [
        ("sample_id", "元数据", "字符串", "样本唯一编号", "不输入模型"),
        ("split", "元数据", "字符串", "train/validation/test", "不输入模型"),
        ("scenario_instance_id", "元数据", "字符串", "合成场景实例编号", "不输入模型"),
        ("scenario_type", "元数据", "字符串", "场景类型", "不输入模型"),
        ("timestamp_s", "元数据", "s", "场景内时间戳", "不输入模型"),
        ("boundary_case", "元数据", "0/1", "是否为阈值边界样本", "不输入模型"),
        ("gps_valid", "特征", "0/1", "GPS定位与速度是否可用", "输入模型"),
        ("gps_speed_kmh", "特征", "km/h", "GPS速度，用于IMU转弯补偿", "输入模型"),
        ("roll_error_deg", "特征", "deg", "转弯补偿后的横滚误差", "输入模型"),
        ("outward_rate_deg_s", "特征", "deg/s", "横滚误差向外增大速度", "输入模型"),
        ("acc_delta_signed_mps2", "特征", "m/s²", "加速度突变量；正负方向均保留", "输入模型"),
        ("acc_change_abs_mps2", "特征", "m/s²", "加速度突变量绝对值", "输入模型"),
        ("jerk_abs_mps3", "特征", "m/s³", "加速度变化率绝对值", "输入模型"),
        ("radar_relative_speed_mps", "特征", "m/s", "项目原始约定：靠近为负", "输入模型"),
        ("radar_closing_speed_mps", "特征", "m/s", "接近速度，靠近为正", "输入模型"),
        ("radar_ttc_s", "特征", "s", "路径目标TTC；无法计算时缺失", "输入模型"),
        ("path_object_count", "特征", "个", "可行驶路径中的视觉目标数量", "输入模型"),
        ("max_path_bottom_ratio", "特征", "0-1", "路径目标框底部相对图像高度", "输入模型"),
        ("box_growth_rate_per_s", "特征", "1/s", "目标框尺度增长率", "输入模型"),
        ("growth_duration_s", "特征", "s", "目标持续增大的时间", "输入模型"),
        ("visual_tau_s", "特征", "s", "视觉目标尺度增长推算的时间尺度", "输入模型"),
        ("rule_score", "标签辅助", "0-1", "各模态规则分数最大值", "禁止输入模型"),
        ("risk_label", "标签", "0/1/2", "0低、1中、2高", "训练目标"),
        ("trigger_reason", "标签辅助", "字符串", "产生最大规则分数的模态", "禁止输入模型"),
    ]
    lines = [
        "# XGBoost风险数据字段字典",
        "",
        "| 字段 | 类型 | 单位/取值 | 含义 | 训练用途 |",
        "|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in dictionary_rows)
    lines.extend([
        "",
        "完整训练字段以 `feature_config.json` 中的 `feature_columns` 为准。",
        "空白的 GPS速度、TTC或距离表示当前窗口无法形成有效估计，不得替换为 0。",
    ])
    (OUTPUT_DIR / "data_dictionary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    rules = f"""# 合成标签规则

本数据集依据 `configs/warning_rules.yaml` 和当前融合代码生成，标签仅表示规则监督，
不表示真实道路事故概率。

## 统一等级与仲裁

- 低风险：`score < {ATTENTION_SCORE}`
- 中风险：`{ATTENTION_SCORE} <= score < {HIGH_SCORE}`
- 高风险：`score >= {HIGH_SCORE}`
- 最终分数：`max(imu_score, motion_score, radar_score, vision_score)`
- 不使用加权平均，不允许一个低风险模态否决另一个高风险模态。

## 雷达

- 项目原始相对速度约定为靠近负值，数据同时提供正向的 `radar_closing_speed_mps`。
- 路径门内目标 TTC <= {RADAR_URGENT_TTC_S:.6f}s 为高风险。
- 路径门内目标 TTC <= {RADAR_ATTENTION_TTC_S:.6f}s 为中风险。
- 雷达目标统一按障碍物处理，不使用视觉原始类别改变风险。

## 视觉

- 所有检测目标在风险语义上统一为 `obstacle`，原始类别不输入模型。
- 仅可行驶路径内目标参与视觉告警。
- 目标底部比例 >= {VISION_NEAR_BOTTOM_RATIO} 进入中风险。
- 视觉尺度增长时间 `visual_tau_s <= {VISION_ATTENTION_TAU_S}s` 进入中风险。
- `visual_tau_s <= {VISION_URGENT_TAU_S}s` 进入高风险。

## IMU侧倾

- 使用转弯补偿后的 `roll_error_deg`，避免正常转弯被直接判危险。
- 中风险需要误差 >= {IMU_ATTENTION_ERROR_DEG}°、向外速度 >=
  {IMU_ATTENTION_OUTWARD_RATE_DEG_S}°/s，并持续 >=
  {IMU_ATTENTION_PERSISTENCE_MS}ms。
- 高风险需要GPS速度可用于转弯补偿、预测在
  {IMU_PREDICTION_HORIZON_S}s 内达到 {IMU_CRITICAL_ERROR_DEG}°，
  且连续满足 {IMU_URGENT_CONSISTENT_SAMPLES} 个样本。

## GPS可用性

- `gps_valid=1` 时输入有效速度。
- `gps_valid=0` 时 `gps_speed_kmh` 保持缺失，不以 0 冒充静止。
- GPS缺失样本独立叠加在低、中、高风险场景中，风险标签继续由可用模态决定。

## 加速度突变（原型扩展）

- 正负突变对称处理，保留有符号变化量用于模型区分急加速与急减速。
- `|delta_acc| >= {MOTION_ATTENTION_DELTA_MPS2}m/s²` 或
  `|jerk| >= {MOTION_ATTENTION_JERK_MPS3}m/s³`：至少中风险。
- `|delta_acc| >= {MOTION_HIGH_DELTA_MPS2}m/s²` 或
  `|jerk| >= {MOTION_HIGH_JERK_MPS3}m/s³`：高风险。
- 这些阈值是合成数据用工程起点，尚未经过车辆实测标定。

## 防止标签泄漏

训练时必须排除 `rule_score`、四个模态规则分数、`trigger_reason`、
`hard_rule_triggered`、`risk_label_name`、场景名称和样本编号。
"""
    (OUTPUT_DIR / "risk_rules.md").write_text(rules, encoding="utf-8")

    readme = """# XGBoost合成风险数据集

这是一个无真实样本阶段的规则监督数据集，用于跑通 XGBoost 三分类流程。

文件：

- `train.csv`：5250条
- `validation.csv`：1125条
- `test.csv`：1125条
- `feature_config.json`：训练目标、特征白名单和泄漏字段
- `risk_rules.md`：标签规则
- `data_dictionary.md`：字段说明
- `qa_report.json`：生成后的自动校验结果

风险标签：`0=低，1=中，2=高`。所有数据均为合成数据，不能用于证明真实道路泛化能力。
验证集和测试集提高了阈值边界样本比例，用于检查模型在 0.35/0.70、
TTC、视觉增长和IMU阈值附近的稳定性。
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    splits = {
        split: make_split(rng, split, quotas)
        for split, quotas in SPLIT_QUOTAS.items()
    }
    for split, rows in splits.items():
        write_csv(OUTPUT_DIR / f"{split}.csv", rows)
    report = validate(splits)
    write_support_files(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
