"""人工注入视觉风险测试脚本。

不依赖摄像头和 OpenVINO，直接构造 VisionData 注入风险融合模型。
验证 R_obs 能否正确影响综合风险分数和等级。

测试 5 个 case：
  1. vision_enabled=False, max_visual_risk=0    → R_obs=0, 视觉不参与
  2. vision_enabled=True,  valid=True,  risk=0.2 → R_obs≈0.2
  3. vision_enabled=True,  valid=True,  risk=0.6 → R_obs≈0.6
  4. vision_enabled=True,  valid=True,  risk=0.9 → R_obs≈0.9
  5. vision_enabled=True,  valid=False, risk=0.9 → R_obs=0 (无效帧不参与)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.data_types import (
    FusionInput,
    GPSData,
    IMUData,
    RadarData,
    RadarTarget,
    VisionData,
    now,
)
from src.fusion.risk_model import RiskModel
from src.fusion.risk_level import determine_risk_level


def make_vision(valid: bool, max_risk: float) -> VisionData:
    """构造视觉数据，只关注 risk 相关字段。"""
    return VisionData(
        timestamp=now(),
        valid=valid,
        max_visual_risk=max_risk,
        objects=[],
        person_count=0,
        vehicle_count=0,
    )


def make_gps(speed: float = 0.0) -> GPSData:
    return GPSData(timestamp=now(), valid=True, speed_kmh=speed, speed_mps=speed / 3.6)


def make_imu() -> IMUData:
    return IMUData(timestamp=now(), valid=True)


def make_radar() -> RadarData:
    """无目标雷达，R_dist=0。"""
    return RadarData(timestamp=now(), valid=True, targets=[], nearest_distance_m=-1.0, min_ttc=-1.0)


def run_case(
    case_id: int,
    desc: str,
    vision_enabled: bool,
    vision_valid: bool,
    max_visual_risk: float,
    gps_speed: float = 0.0,
) -> dict:
    """运行单个测试 case，返回风险项字典。"""
    fusion = FusionInput(
        timestamp=now(),
        vision_enabled=vision_enabled,
        vision=make_vision(vision_valid, max_visual_risk),
        gps=make_gps(gps_speed),
        imu=make_imu(),
        radar=make_radar(),
    )

    model = RiskModel()
    risk_items, weights = model.compute(fusion)
    level, label = determine_risk_level(
        risk_items["risk_score"],
        low_threshold=model.thresholds["low"],
        high_threshold=model.thresholds["high"],
    )

    result = {
        "case": case_id,
        "desc": desc,
        "vision_enabled": vision_enabled,
        "vision_valid": vision_valid,
        "injected_max_risk": max_visual_risk,
        "R_obs": risk_items["R_obs"],
        "R_dist": risk_items["R_dist"],
        "R_pose": risk_items["R_pose"],
        "R_speed": risk_items["R_speed"],
        "risk_score": risk_items["risk_score"],
        "risk_level": level,
        "risk_label": label,
        "weights": weights,
    }
    return result


def print_case(r: dict) -> None:
    """格式化输出。"""
    print(f"  [{r['case']}] {r['desc']}")
    print(f"       vision_enabled={r['vision_enabled']}, "
          f"vision_valid={r['vision_valid']}, "
          f"injected_max_risk={r['injected_max_risk']}")
    print(f"       R_obs={r['R_obs']:.3f}  R_dist={r['R_dist']:.3f}  "
          f"R_pose={r['R_pose']:.3f}  R_speed={r['R_speed']:.3f}")
    print(f"       risk_score={r['risk_score']:.3f}  "
          f"risk_level={r['risk_level']}({r['risk_label']})")
    print(f"       权重: obs={r['weights']['obs']:.3f}  "
          f"dist={r['weights']['dist']:.3f}  "
          f"pose={r['weights']['pose']:.3f}  "
          f"speed={r['weights']['speed']:.3f}")
    print()


def verify_assertions(results: list) -> None:
    """验证所有 case 的预期结果。"""
    print("=" * 55)
    print("  断言检查")
    print("-" * 55)

    # case 1: vision_enabled=False → R_obs=0
    r1 = results[0]
    assert r1["R_obs"] == 0.0, f"Case 1: R_obs 应为 0, 实际={r1['R_obs']}"
    assert r1["risk_score"] < 0.30, f"Case 1: 应为低风险, risk={r1['risk_score']:.3f}"
    print("  [OK] Case 1: vision_enabled=False → R_obs=0")

    # case 2: low visual risk → R_obs ≈ 0.2
    r2 = results[1]
    assert abs(r2["R_obs"] - 0.2) < 0.01, f"Case 2: R_obs 应 ≈ 0.2, 实际={r2['R_obs']:.3f}"
    print(f"  [OK] Case 2: R_obs={r2['R_obs']:.3f} (注入=0.2)")

    # case 3: medium visual risk → R_obs ≈ 0.6
    r3 = results[2]
    assert abs(r3["R_obs"] - 0.6) < 0.01, f"Case 3: R_obs 应 ≈ 0.6, 实际={r3['R_obs']:.3f}"
    print(f"  [OK] Case 3: R_obs={r3['R_obs']:.3f} (注入=0.6)")

    # case 4: high visual risk → R_obs ≈ 0.9
    r4 = results[3]
    assert abs(r4["R_obs"] - 0.9) < 0.01, f"Case 4: R_obs 应 ≈ 0.9, 实际={r4['R_obs']:.3f}"
    print(f"  [OK] Case 4: R_obs={r4['R_obs']:.3f} (注入=0.9)")

    # 验证风险分数随 visual_risk 上升而上升
    assert results[1]["risk_score"] < results[2]["risk_score"] < results[3]["risk_score"], \
        "风险分数应随 visual_risk 单调递增"
    print("  [OK] risk_score 随 visual_risk 单调递增")

    # case 5: vision_valid=False → R_obs=0 即使 max_visual_risk=0.9
    r5 = results[4]
    assert r5["R_obs"] == 0.0, f"Case 5: vision_valid=False, R_obs 应为 0, 实际={r5['R_obs']}"
    print("  [OK] Case 5: vision_valid=False → R_obs=0 (无效帧不参与)")

    # R_obs 对风险等级的定量影响
    assert r1["risk_level"] == 0, "Case 1: 应为 level=0"
    assert r4["risk_level"] >= 1, "Case 4: max_risk=0.9, 至少中风险"
    print(f"  [OK] 风险等级从 level={r1['risk_level']} 上升到 level={r4['risk_level']}")
    print("\n  [PASS] 全部断言通过!")


def main():
    LOW = 0.30
    HIGH = 0.70

    # 用 GPS 速度 0 排除 R_speed 干扰，IMU/Radar 无风险
    cases = [
        (1, "视觉未启用 (vision_enabled=False)", False, False, 0.0, 0.0),
        (2, "低视觉风险 (max_risk=0.2)", True, True, 0.2, 0.0),
        (3, "中视觉风险 (max_risk=0.6)", True, True, 0.6, 0.0),
        (4, "高视觉风险 (max_risk=0.9)", True, True, 0.9, 0.0),
        (5, "视觉无效帧 (valid=False, risk=0.9)", True, False, 0.9, 0.0),
    ]

    print("=" * 55)
    print("  视觉风险注入测试")
    print("=" * 55)

    results = []
    for cid, desc, v_enabled, v_valid, m_risk, speed in cases:
        r = run_case(cid, desc, v_enabled, v_valid, m_risk, gps_speed=speed)
        results.append(r)
        print_case(r)

    verify_assertions(results)

    print("=" * 55)
    print("  全部测试通过!")
    print("=" * 55)


if __name__ == "__main__":
    main()
