"""main_integrated.py 整合测试脚本。

测试 3 个 case：
  1. mock + vision false + loops=20
     预期：完整闭环运行成功，R_obs=0，日志生成，马达 mock 正常输出

  2. mock + vision true + loops=10
     预期：视觉不可用时 graceful degradation；R_obs=0；程序不崩溃

  3. 人工视觉风险注入
     预期：注入非零 max_visual_risk 后 R_obs 非零，risk_score 上升
"""

import sys
import os
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.data_types import (
    FusionInput,
    GPSData,
    IMUData,
    RadarData,
    VisionData,
    now,
)
from src.fusion.risk_model import RiskModel
from src.fusion.risk_level import determine_risk_level


def verify_log(path: str, min_lines: int = 1) -> bool:
    """验证日志文件存在且行数 >= min_lines。"""
    if not os.path.exists(path):
        print(f"  [FAIL] 日志文件不存在: {path}")
        return False
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) < min_lines:
        print(f"  [FAIL] 日志行数不足: 期望 >= {min_lines}, 实际 {len(lines)}")
        return False
    print(f"  [OK] 日志文件 {path} ({len(lines)} 行)")
    return True


def run_main_integrated(args: list, timeout: float = 30.0) -> tuple[int, str, str]:
    """运行 main_integrated.py，返回 (returncode, stdout, stderr)。"""
    cmd = [sys.executable, "main_integrated.py"] + args
    print(f"  运行: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        print("  [WARN] 子进程超时，已被终止")

    return proc.returncode, stdout, stderr


# ============================================================
#  Case 1: mock + vision false + loops=20
# ============================================================


def test_case1():
    print("=" * 65)
    print("  Case 1: mock + vision false + loops=20")
    print("-" * 65)

    ret, stdout, stderr = run_main_integrated(
        ["--mode", "mock", "--vision", "false", "--loops", "20"],
        timeout=30.0,
    )

    # 检查退出码
    assert ret == 0, f"退出码应为 0, 实际={ret}"
    print("  [OK] 正常退出 (exit code=0)")

    # 检查 R_obs=0 (视觉未启用)
    assert "R_obs=0.000" in stdout, "R_obs 应为 0.000"
    print("  [OK] R_obs=0.000 (视觉未启用)")

    # 检查马达 mock 输出
    assert "MotorController" in stdout, "应有马达输出"
    print("  [OK] 马达控制正常")

    # 检查日志文件
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "main_integrated_mock.csv")
    verify_log(log_path, min_lines=21)

    # 检查日志字段
    with open(log_path, "r") as f:
        header = f.readline().strip()
    assert "R_obs" in header, f"日志缺少 R_obs 字段: {header}"
    assert "risk_score" in header, f"日志缺少 risk_score 字段: {header}"
    print("  [OK] 日志字段完整")

    print("\n  [PASS] Case 1 通过")
    return True


# ============================================================
#  Case 2: mock + vision true + loops=10
# ============================================================


def test_case2():
    print("=" * 65)
    print("  Case 2: mock + vision true + loops=10")
    print("-" * 65)

    ret, stdout, stderr = run_main_integrated(
        ["--mode", "mock", "--vision", "true", "--loops", "10"],
        timeout=30.0,
    )

    # 检查退出码
    assert ret == 0, f"退出码应为 0, 实际={ret}"
    print("  [OK] 正常退出 (exit code=0)")

    # 视觉可能不可用（无 openvino），应 graceful degradation
    if "降级" in stdout or "不可用" in stdout:
        print("  [OK] 视觉 graceful degradation (当前环境无 openvino)")
        assert "R_obs=0.000" in stdout, "R_obs 应为 0.000"
        print("  [OK] R_obs=0.000")
    else:
        print("  [OK] 视觉正常加载")
        # 检查 R_obs 是否参与融合（视觉可能有数据也可能没有）
        if "vision_valid=True" in stdout:
            print("  [OK] 视觉帧有效")
        else:
            print("  [INFO] 视觉帧无效（预期降级行为）")

    # 检查日志文件
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "main_integrated_mock.csv")
    verify_log(log_path, min_lines=11)

    print("\n  [PASS] Case 2 通过")
    return True


# ============================================================
#  Case 3: 人工视觉风险注入（不依赖 main_integrated.py）
# ============================================================


def test_case3():
    """直接构造 FusionInput 注入视觉风险，验证 R_obs 影响 risk_score。"""
    print("=" * 65)
    print("  Case 3: 人工视觉风险注入")
    print("-" * 65)

    def make_vision(valid: bool, max_risk: float) -> VisionData:
        return VisionData(timestamp=now(), valid=valid, max_visual_risk=max_risk)

    model = RiskModel()

    # 构造 4 组测试（仅 vision 不同，其他传感器全 mock 且有效）
    cases = [
        ("vision_enabled=False", False, True, 0.0),
        ("低视觉风险 (0.2)", True, True, 0.2),
        ("中视觉风险 (0.6)", True, True, 0.6),
        ("高视觉风险 (0.9)", True, True, 0.9),
    ]

    scores = []
    for desc, enabled, v_valid, m_risk in cases:
        fusion = FusionInput(
            timestamp=now(),
            vision_enabled=enabled,
            vision=make_vision(v_valid, m_risk),
            gps=GPSData(timestamp=now(), valid=True, speed_kmh=5.0),
            imu=IMUData(timestamp=now(), valid=True),
            radar=RadarData(timestamp=now(), valid=True, targets=[]),
        )
        risk_items, weights = model.compute(fusion)
        level, label = determine_risk_level(
            risk_items["risk_score"],
            low_threshold=model.thresholds["low"],
            high_threshold=model.thresholds["high"],
        )
        scores.append((m_risk, risk_items["R_obs"], risk_items["risk_score"], level))

        print(f"  [{desc}]")
        print(f"       R_obs={risk_items['R_obs']:.3f}  "
              f"risk_score={risk_items['risk_score']:.3f}  level={level}")

    # 断言
    _, r_obs_0, score_0, _ = scores[0]
    assert r_obs_0 == 0.0, "vision_enabled=False → R_obs=0"
    for i in range(1, 4):
        injected = scores[i][0]
        actual_r = scores[i][1]
        assert abs(actual_r - injected) < 0.01, \
            f"R_obs({actual_r:.3f}) 应等于注入值({injected})"

    # 验证 risk_score 单调递增
    for i in range(1, 4):
        assert scores[i][2] > scores[i - 1][2], \
            f"risk_score 应递增: {[s[2] for s in scores]}"

    print("  [OK] R_obs = 注入的 max_visual_risk")
    print("  [OK] risk_score 随视觉风险增加而上升")
    print("  [OK] vision_enabled=False 时 R_obs=0")

    print("\n  [PASS] Case 3 通过")
    return True


# ============================================================
#  主入口
# ============================================================


def main():
    results = []
    cases = [
        ("Case 1: mock + vision false", test_case1),
        ("Case 2: mock + vision true", test_case2),
        ("Case 3: 人工视觉风险注入", test_case3),
    ]

    for name, fn in cases:
        print()
        try:
            fn()
            results.append((name, True))
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 汇总
    print("\n" + "=" * 65)
    print("  测试汇总")
    print("=" * 65)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    print("-" * 65)
    if all_pass:
        print("  全部通过!")
    else:
        print("  存在失败项")
    print("=" * 65)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
