"""视觉-雷达融合 smoke test（合成数据，无需硬件/模型）。

验证三类目标 + 杠杆1 路面门控 + 杠杆3 持续性：
  1. 视觉∩雷达：检测框与雷达目标方位对齐 → source=vision_radar，用雷达 TTC。
  2. 仅雷达未知障碍（在路面、在走廊）→ source=radar，risk 随持续帧数上升。
  3. 仅雷达但投影到路面外 → 被门控丢弃。
  4. 仅视觉（无雷达回波）→ source=vision，沿用 visual_risk。

运行：
    D:/Anaconda_envs/envs/intel/python.exe scripts/test_vision_radar_fusion_mock.py
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

from src.fusion.data_types import RadarData, RadarTarget
from src.fusion.vision_radar_fusion import VisionRadarFusion, bearing_to_col
from src.vision.common.types import DetectionResult, SegmentationResult, VisionResult

W, H = 1280, 720


def make_vision() -> VisionResult:
    # 一个正前方的车（bearing≈0°），visual_risk 预设 0.5
    car = DetectionResult(
        class_name="car", risk_class="motor_vehicle", confidence=0.8,
        bbox=(560, 300, 720, 520), in_drivable_area=True, visual_risk=0.5)
    # 一个右侧的人（bearing≈+24°，超出雷达 ±15° 视场）→ 雷达看不到 → 仅视觉
    person = DetectionResult(
        class_name="person", risk_class="pedestrian", confidence=0.7,
        bbox=(900, 320, 960, 500), in_drivable_area=True, visual_risk=0.42)
    # 路面掩码：下半部、列 400~700 为路面（道路略偏左）；右侧非路面
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[360:, 400:700] = 1
    seg = SegmentationResult(drivable_mask=mask, drivable_ratio=float(mask.mean()))
    return VisionResult(detections=[car, person], segmentation=seg,
                        drivable_mask=mask, max_visual_risk=0.5)


def make_radar() -> RadarData:
    # 角度均在 LD2451 实际 ±15° 视场内
    targets = [
        # ① 与车对齐（angle≈0），8m 接近 → 匹配检测 → vision_radar
        RadarTarget(target_id=0, distance_m=8.0, relative_speed_mps=-4.0, angle_deg=0.0, confidence=0.9),
        # ② 未知障碍：-10°（在走廊边缘、投影在路面），6m 接近 → radar-only 保留（lateral 略衰减）
        RadarTarget(target_id=1, distance_m=6.0, relative_speed_mps=-3.0, angle_deg=-10.0, confidence=0.8),
        # ③ 未知障碍：+12°（投影到路面外，col~776 > 700）→ 路面门控丢弃
        RadarTarget(target_id=2, distance_m=5.0, relative_speed_mps=-2.0, angle_deg=12.0, confidence=0.7),
    ]
    nearest = min(t.distance_m for t in targets)
    return RadarData(valid=True, targets=targets, nearest_distance_m=nearest, min_ttc=2.0)


def main() -> None:
    fusion = VisionRadarFusion()
    vision, radar = make_vision(), make_radar()

    print(f"角度→列校验: -10°→col {bearing_to_col(-10,W,90)}（路面内 400~700）, "
          f"+12°→col {bearing_to_col(12,W,90)}（路面外 >700 → 被门控）\n")

    # 跑 3 帧观察持续性 ramp
    for frame in range(1, 4):
        out = fusion.update(vision, radar, W, H)
        print(f"=== 帧 {frame} ===  场景 max_risk={out.max_risk:.3f}  "
              f"(vision_radar={out.n_vision_radar}, vision={out.n_vision_only}, radar={out.n_radar_only})")
        for o in out.objects:
            extra = f" d={o.distance_m:.0f}m ttc={o.ttc_sec:.1f}s" if o.distance_m > 0 else ""
            road = f" on_road={o.on_road}" if o.source == "radar" else ""
            print(f"  [{o.source:<12}] {o.risk_class:<14} risk={o.risk:.3f} "
                  f"persist={o.persist:.2f}{extra}{road}")
        print()

    # —— 断言三类目标行为正确 ——
    out = fusion.update(vision, radar, W, H)
    srcs = [o.source for o in out.objects]
    assert "vision_radar" in srcs, "应有视觉∩雷达目标"
    assert "vision" in srcs, "应有仅视觉目标（右侧的人无雷达回波）"
    assert out.n_radar_only == 1, f"应恰好 1 个仅雷达未知障碍（-10°在途在路面），实际 {out.n_radar_only}"
    radar_obj = next(o for o in out.objects if o.source == "radar")
    assert radar_obj.on_road is True, "保留的未知障碍应在路面上"
    assert radar_obj.persist >= 1.0, "跑满 3 帧后持续可信度应达 1.0"
    print("✅ 断言通过：三类目标 + 路面门控（+12°被丢弃）+ 持续性 ramp 均正确")


if __name__ == "__main__":
    main()
