"""端到端 mock demo：检测+分割 → 视觉-雷达融合 → risk_model → 风险等级。

合成 VisionResult（检测框+路面掩码）+ mock RadarData，跑通整条风险链，
无需摄像头/模型/硬件。验证融合产出的 R_obs 正确流入 risk_model 并影响最终等级，
并对比「纯视觉 R_obs」vs「融合后 R_obs」体现雷达带来的差异。

运行：
    D:/Anaconda_envs/envs/intel/python.exe scripts/test_vision_radar_pipeline_mock.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.fusion.data_types import (
    FusionInput, GPSData, IMUData, RadarData, RadarTarget, VisionData, now,
)
from src.fusion.risk_level import RiskLevelClassifier
from src.fusion.risk_model import RiskModel
from src.fusion.vision_radar_fusion import VisionRadarFusion
from src.vision.common.types import DetectionResult, SegmentationResult, VisionResult

W, H = 1280, 720


def make_vision() -> VisionResult:
    car = DetectionResult("car", "motor_vehicle", 0.8, (560, 300, 720, 520), True, 0.50)
    person = DetectionResult("person", "pedestrian", 0.7, (900, 320, 960, 500), True, 0.42)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[360:, 400:700] = 1
    seg = SegmentationResult(drivable_mask=mask, drivable_ratio=float(mask.mean()))
    vis_only = max(d.visual_risk for d in (car, person))
    return VisionResult(detections=[car, person], segmentation=seg,
                        drivable_mask=mask, max_visual_risk=vis_only)


def make_radar() -> RadarData:
    targets = [
        RadarTarget(0, distance_m=8.0, relative_speed_mps=-4.0, angle_deg=0.0, confidence=0.9),   # 对齐车
        RadarTarget(1, distance_m=6.0, relative_speed_mps=-3.0, angle_deg=-10.0, confidence=0.8),  # 未知障碍(在路面)
    ]
    return RadarData(valid=True, targets=targets, nearest_distance_m=6.0, min_ttc=2.0)


def main() -> None:
    vision, radar = make_vision(), make_radar()
    fuser = VisionRadarFusion()
    risk_model = RiskModel()
    classifier = RiskLevelClassifier()

    # 跑几帧让持续性 ramp 起来
    fused = None
    for _ in range(3):
        fused = fuser.fuse_vision_result(vision, radar)

    print("=" * 66)
    print("逐目标融合结果")
    print("=" * 66)
    for o in fused.objects:
        d = f" d={o.distance_m:.0f}m ttc={o.ttc_sec:.1f}s" if o.distance_m > 0 else ""
        print(f"  [{o.source:<12}] {o.risk_class:<14} risk={o.risk:.3f}{d}")
    print(f"\n纯视觉 R_obs = {vision.max_visual_risk:.3f}  →  融合后 R_obs = {fused.max_risk:.3f}"
          f"  (雷达兜底未知障碍 ×{fused.n_radar_only})")

    # 构造 FusionInput：用融合后的 R_obs 覆盖视觉项，再走 risk_model
    fin = FusionInput(
        timestamp=now(),
        gps=GPSData(valid=True, speed_kmh=15.0),
        imu=IMUData(valid=True),
        radar=radar,
        vision=VisionData(valid=True, max_visual_risk=fused.max_risk),
        vision_enabled=True,
    )
    items, weights = risk_model.compute(fin)
    level, label = classifier.classify(items["risk_score"])

    print("\n" + "=" * 66)
    print("风险融合（risk_model）")
    print("=" * 66)
    print(f"  R_obs={items['R_obs']:.3f}  R_dist={items['R_dist']:.3f}  "
          f"R_pose={items['R_pose']:.3f}  R_speed={items['R_speed']:.3f}")
    print(f"  权重 obs/dist/pose/speed = "
          f"{weights['obs']:.2f}/{weights['dist']:.2f}/{weights['pose']:.2f}/{weights['speed']:.2f}")
    print(f"  综合风险 score = {items['risk_score']:.3f}  →  等级 {level} ({label})")

    # —— 断言整条链贯通且融合生效 ——
    assert fused.max_risk >= vision.max_visual_risk - 1e-6, "融合 R_obs 不应低于纯视觉(此场景雷达增益)"
    assert fused.n_radar_only == 1, "应保留 1 个仅雷达未知障碍"
    assert 0.0 <= items["risk_score"] <= 1.0
    print("\n✅ 端到端贯通：检测+分割 → 融合(R_obs) → risk_model → 风险等级，均正常")


if __name__ == "__main__":
    main()
