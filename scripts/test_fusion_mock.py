"""风险融合模块 mock 测试脚本。

覆盖 5 个测试用例：
  1. 低速 + 雷达无目标 + 姿态稳定             → 低风险 level=0
  2. 中速 + 雷达目标 TTC 较小               → 中/高风险
  3. 高速 + 急刹姿态 + 雷达接近目标           → 高风险 level=2
  4. 雷达异常，但 GPS 和 IMU 正常             → 不崩溃，输出合理风险
  5. 所有输入来自 mock reader 连续 10 次       → 稳定输出

运行方式：
  python scripts/test_fusion_mock.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random

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
from src.fusion.risk_level import RiskLevelClassifier, determine_risk_level
from src.sensors.gps_reader import GPSReader
from src.sensors.imu_reader import IMUReader
from src.sensors.radar_reader import RadarReader
from src.fusion.synchronizer import Synchronizer

# 默认阈值
LOW = 0.30
HIGH = 0.70


def make_gps(speed_kmh: float, valid: bool = True) -> GPSData:
    return GPSData(timestamp=now(), valid=valid, speed_kmh=speed_kmh, speed_mps=speed_kmh / 3.6)


def make_imu(brake: float = 0.0, bump: float = 0.0, tilt: float = 0.0, valid: bool = True) -> IMUData:
    return IMUData(
        timestamp=now(), valid=valid,
        brake_score=brake, bump_score=bump, tilt_score=tilt,
    )


def make_radar(targets: list, valid: bool = True) -> RadarData:
    """targets: [(distance_m, speed_mps, angle_deg, confidence), ...]"""
    rt_list = []
    nearest = -1.0
    min_ttc = -1.0
    for i, (d, s, a, c) in enumerate(targets):
        rt = RadarTarget(target_id=i, distance_m=d, relative_speed_mps=s, angle_deg=a, confidence=c)
        rt_list.append(rt)
        if nearest < 0 or d < nearest:
            nearest = d
        if s < 0:  # 接近
            ttc = d / (-s)
            if min_ttc < 0 or ttc < min_ttc:
                min_ttc = round(ttc, 1)
    return RadarData(
        timestamp=now(), valid=valid,
        targets=rt_list, nearest_distance_m=nearest, min_ttc=min_ttc,
    )


def make_vision(max_risk: float = 0.0, valid: bool = True) -> VisionData:
    return VisionData(timestamp=now(), valid=valid, max_visual_risk=max_risk)


def test1_low_speed_no_target_stable():
    """测试 1：低速(10km/h) + 雷达无目标 + 姿态稳定 → 低风险 level=0"""
    print("=" * 55)
    print("测试 1：低速 + 雷达无目标 + 姿态稳定")
    print("-" * 55)

    fusion = FusionInput(
        timestamp=now(),
        gps=make_gps(10.0),
        imu=make_imu(),
        radar=make_radar([]),
        vision=VisionData(timestamp=now()),
        vision_enabled=False,
    )

    model = RiskModel()
    risk_items, weights = model.compute(fusion)
    level, label = determine_risk_level(risk_items["risk_score"], LOW, HIGH)

    print(f"  R_obs   = {risk_items['R_obs']:.3f}")
    print(f"  R_dist  = {risk_items['R_dist']:.3f}")
    print(f"  R_pose  = {risk_items['R_pose']:.3f}")
    print(f"  R_speed = {risk_items['R_speed']:.3f}")
    print(f"  risk    = {risk_items['risk_score']:.3f}")
    print(f"  level   = {level} ({label})")
    print(f"  权重: {weights}")

    assert risk_items["risk_score"] < LOW, f"应为低风险, 实际={risk_items['risk_score']:.3f}"
    assert level == 0, f"level 应为 0, 实际={level}"
    print("[PASS] 低风险确认\n")


def test2_medium_speed_approaching_target():
    """测试 2：中速(20km/h) + 雷达目标 TTC ≈ 2s → 中/高风险"""
    print("=" * 55)
    print("测试 2：中速 + 雷达目标 TTC 较小")
    print("-" * 55)

    # 目标距离 6m，速度 -3m/s → TTC = 2.0s
    fusion = FusionInput(
        timestamp=now(),
        gps=make_gps(20.0),
        imu=make_imu(),
        radar=make_radar([(6.0, -3.0, 0.0, 0.85)]),
        vision=VisionData(timestamp=now()),
        vision_enabled=False,
    )

    model = RiskModel()
    risk_items, weights = model.compute(fusion)
    level, label = determine_risk_level(risk_items["risk_score"], LOW, HIGH)

    print(f"  R_obs   = {risk_items['R_obs']:.3f}")
    print(f"  R_dist  = {risk_items['R_dist']:.3f}")
    print(f"  R_pose  = {risk_items['R_pose']:.3f}")
    print(f"  R_speed = {risk_items['R_speed']:.3f}")
    print(f"  risk    = {risk_items['risk_score']:.3f}")
    print(f"  level   = {level} ({label})")

    assert risk_items["R_dist"] > 0.3, f"R_dist 应 > 0.3 (TTC=2s), 实际={risk_items['R_dist']:.3f}"
    assert level >= 1, f"至少中风险, 实际 level={level}"
    print("[PASS] 中/高风险确认\n")


def test3_high_speed_braking_approaching():
    """测试 3：高速(40km/h) + 急刹姿态 + 雷达接近目标(TTC≈0.4s) → 高风险 level=2"""
    print("=" * 55)
    print("测试 3：高速 + 急刹姿态 + 雷达接近目标")
    print("-" * 55)

    fusion = FusionInput(
        timestamp=now(),
        gps=make_gps(40.0),  # 40km/h → R_speed=1.0
        imu=make_imu(brake=0.9, bump=0.2, tilt=0.1),  # R_pose=0.45
        radar=make_radar([(3.0, -8.0, 0.0, 0.95)]),  # TTC≈0.375s → R_dist≈0.925
        vision=VisionData(timestamp=now()),
        vision_enabled=False,
    )

    model = RiskModel()
    risk_items, weights = model.compute(fusion)
    level, label = determine_risk_level(risk_items["risk_score"], LOW, HIGH)

    print(f"  R_obs   = {risk_items['R_obs']:.3f}")
    print(f"  R_dist  = {risk_items['R_dist']:.3f}")
    print(f"  R_pose  = {risk_items['R_pose']:.3f}")
    print(f"  R_speed = {risk_items['R_speed']:.3f}")
    print(f"  risk    = {risk_items['risk_score']:.3f}")
    print(f"  level   = {level} ({label})")

    assert risk_items["R_dist"] > 0.8, f"TTC≈0.375s, R_dist 应 > 0.8, 实际={risk_items['R_dist']:.3f}"
    assert risk_items["R_pose"] > 0.3, f"急刹, R_pose 应 > 0.3, 实际={risk_items['R_pose']:.3f}"
    assert level == 2, f"应为高风险 level=2, 实际 level={level}"
    print("[PASS] 高风险确认\n")


def test4_radar_abnormal():
    """测试 4：雷达异常 (valid=False)，但 GPS 和 IMU 正常 → 不崩溃，输出合理"""
    print("=" * 55)
    print("测试 4：雷达异常，GPS/IMU 正常")
    print("-" * 55)

    fusion = FusionInput(
        timestamp=now(),
        gps=make_gps(20.0),
        imu=make_imu(brake=0.0, bump=0.0, tilt=0.0),
        radar=RadarData(timestamp=now(), valid=False),  # 雷达异常
        vision=VisionData(timestamp=now()),
        vision_enabled=False,
    )

    model = RiskModel()
    risk_items, weights = model.compute(fusion)
    level, label = determine_risk_level(risk_items["risk_score"], LOW, HIGH)

    print(f"  R_obs   = {risk_items['R_obs']:.3f}")
    print(f"  R_dist  = {risk_items['R_dist']:.3f}")
    print(f"  R_pose  = {risk_items['R_pose']:.3f}")
    print(f"  R_speed = {risk_items['R_speed']:.3f}")
    print(f"  risk    = {risk_items['risk_score']:.3f}")
    print(f"  level   = {level} ({label})")
    print(f"  权重: {weights}")

    # 雷达无效 → R_dist=0，且雷达权重应被降低
    assert risk_items["R_dist"] == 0.0, f"雷达异常, R_dist 应为 0, 实际={risk_items['R_dist']}"
    assert weights["dist"] < 0.30, f"雷达权重应降低, 实际={weights['dist']:.3f}"
    print("[PASS] 雷达异常处理正确，未崩溃\n")


def test5_mock_reader_integration():
    """测试 5：从 mock reader 读取 + 同步 + 融合，连续 10 次，稳定输出"""
    print("=" * 55)
    print("测试 5：mock reader 集成测试 (10 次)")
    print("-" * 55)

    gps_reader = GPSReader(mode="mock")
    imu_reader = IMUReader(mode="mock")
    radar_reader = RadarReader(mode="mock")
    gps_reader.start()
    imu_reader.start()
    radar_reader.start()

    sync = Synchronizer(vision_enabled=False)
    model = RiskModel()
    classifier = RiskLevelClassifier()

    print(f"  {'#':>3s} | {'R_obs':>5s} {'R_dist':>5s} {'R_pose':>5s} {'R_speed':>5s} | {'risk':>5s} | {'level':>6s}")
    print("-" * 55)

    for i in range(10):
        gps = gps_reader.read_once()
        imu = imu_reader.read_once()
        radar = radar_reader.read_once()

        sync.update_gps(gps)
        sync.update_imu(imu)
        sync.update_radar(radar)

        fusion = sync.build_frame()
        risk_items, weights = model.compute(fusion)
        level, label = classifier.classify(risk_items["risk_score"])

        print(f"  {i + 1:3d} | {risk_items['R_obs']:.3f}  {risk_items['R_dist']:.3f}  "
              f"{risk_items['R_pose']:.3f}  {risk_items['R_speed']:.3f} | "
              f"{risk_items['risk_score']:.3f} | {level:6d}")

        # 验证所有字段在合法范围
        for k, v in risk_items.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} 超出 [0,1]"
        assert level in (0, 1, 2), f"level={level} 无效"

    gps_reader.stop()
    imu_reader.stop()
    radar_reader.stop()
    print("[PASS] 集成测试: 10 次均稳定输出\n")


if __name__ == "__main__":
    random.seed(42)
    test1_low_speed_no_target_stable()
    test2_medium_speed_approaching_target()
    test3_high_speed_braking_approaching()
    test4_radar_abnormal()
    test5_mock_reader_integration()
    print("=" * 55)
    print("全部 5 个测试通过!")
    print("=" * 55)
