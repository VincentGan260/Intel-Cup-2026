"""震动马达控制模块 mock 测试脚本。

验证以下 6 项：
  1. level=0 不震动（静默，不打印）
  2. level=1 输出中风险震动模式
  3. level=2 输出高风险震动模式
  4. 连续重复 level=1 时冷却机制生效
  5. level=2 可以打断 level=1
  6. mock 模式下没有硬件也能运行

运行方式：
  python scripts/test_motor_mock.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.actuator.motor_controller import MotorController
from src.fusion.data_types import MotorCommand


def test1_level0_silent():
    """测试 1：level=0 不震动"""
    print("=" * 55)
    print("测试 1：level=0 静默")
    print("-" * 55)

    motor = MotorController(mode="mock")
    motor.start()

    motor.alert_low()
    cmd = motor.get_latest()
    assert cmd is not None, "get_latest() 不应为 None"
    assert cmd.risk_level == 0, f"level 应为 0, 实际={cmd.risk_level}"
    assert cmd.pattern == "silent", f"pattern 应为 silent, 实际={cmd.pattern}"
    print(f"  cmd: level={cmd.risk_level}, pattern={cmd.pattern}")
    print("  [PASS] level=0 为静默模式")

    # 重复发送静默，不应重复触发
    motor.alert_low()
    motor.alert_low()
    print("  [PASS] 重复 level=0 未重复触发")

    motor.stop()
    print()


def test2_level1_medium():
    """测试 2：level=1 中风险震动"""
    print("=" * 55)
    print("测试 2：level=1 中风险震动")
    print("-" * 55)

    motor = MotorController(mode="mock")
    motor.start()

    motor.alert_medium(risk_score=0.45)
    cmd = motor.get_latest()
    assert cmd is not None
    assert cmd.risk_level == 1, f"level 应为 1, 实际={cmd.risk_level}"
    assert cmd.pattern == "short_pulse", f"pattern 应为 short_pulse, 实际={cmd.pattern}"
    assert cmd.duration_ms > 0, f"duration_ms 应 > 0, 实际={cmd.duration_ms}"
    print(f"  cmd: level={cmd.risk_level}, pattern={cmd.pattern}, duration={cmd.duration_ms}ms, score={cmd.risk_score:.2f}")
    print("  [PASS] level=1 输出短促轻震")

    motor.stop()
    print()


def test3_level2_high():
    """测试 3：level=2 高风险震动"""
    print("=" * 55)
    print("测试 3：level=2 高风险震动")
    print("-" * 55)

    motor = MotorController(mode="mock")
    motor.start()

    motor.alert_high(risk_score=0.85)
    cmd = motor.get_latest()
    assert cmd is not None
    assert cmd.risk_level == 2, f"level 应为 2, 实际={cmd.risk_level}"
    assert cmd.pattern == "continuous_strong", f"pattern 应为 continuous_strong, 实际={cmd.pattern}"
    assert cmd.duration_ms >= 400, f"高风险 duration_ms 应 >= 400ms, 实际={cmd.duration_ms}"
    print(f"  cmd: level={cmd.risk_level}, pattern={cmd.pattern}, duration={cmd.duration_ms}ms, score={cmd.risk_score:.2f}")
    print("  [PASS] level=2 输出持续强震")

    motor.stop()
    print()


def test4_cooldown():
    """测试 4：连续重复 level=1 时冷却机制生效"""
    print("=" * 55)
    print("测试 4：冷却机制")
    print("-" * 55)

    motor = MotorController(mode="mock")
    motor.start()

    # 第一次 alert_medium
    motor.alert_medium(risk_score=0.50)
    cmd1_time = motor._last_level_time
    print(f"  第 1 次: level=1, time={cmd1_time:.3f}")

    # 立即重复（冷却期内，应被忽略）
    motor.alert_medium(risk_score=0.55)
    cmd2_time = motor._last_level_time
    same = abs(cmd2_time - cmd1_time) < 0.001
    print(f"  第 2 次(无间隔): time={cmd2_time:.3f}, 被冷却={'是' if same else '否'}")
    assert same, "冷却期内重复触发应被忽略"

    # 等待冷却期后再触发
    time.sleep(0.6)  # min_interval_sec=0.5
    motor.alert_medium(risk_score=0.60)
    cmd3_time = motor._last_level_time
    different = abs(cmd3_time - cmd1_time) > 0.5
    print(f"  第 3 次(等待 0.6s): time={cmd3_time:.3f}, 已通过冷却={'是' if different else '否'}")
    assert different, "冷却期后应允许触发"

    print("  [PASS] 冷却机制正常工作")

    motor.stop()
    print()


def test5_interrupt():
    """测试 5：level=2 可以打断 level=1"""
    print("=" * 55)
    print("测试 5：高风险打断中风险")
    print("-" * 55)

    motor = MotorController(mode="mock")
    motor.start()

    # 先触发中风险
    motor.alert_medium(risk_score=0.45)
    assert motor._last_level == 1
    print("  中风险已触发, level=1")

    # 立即触发高风险（应打断）
    motor.alert_high(risk_score=0.80)
    assert motor._last_level == 2, f"打断后 level 应为 2, 实际={motor._last_level}"
    assert not motor._is_medium_playing, "打断后 _is_medium_playing 应为 False"
    print("  高风险已打断中风险, level=2")

    print("  [PASS] 高风险可打断中风险")

    motor.stop()
    print()


def test6_mock_no_hardware():
    """测试 6：mock 模式没有硬件也能运行"""
    print("=" * 55)
    print("测试 6：mock 模式无硬件运行")
    print("-" * 55)

    # 即使硬件不可用也不应崩溃
    motor = MotorController(mode="mock", i2c_bus=999)
    try:
        motor.start()
        motor.alert_low()
        motor.alert_medium(risk_score=0.50)
        motor.alert_high(risk_score=0.85)
        cmd = motor.get_latest()
        assert cmd is not None
        assert cmd.risk_level == 2
        motor.stop()
        print("  [PASS] mock 模式可在无硬件环境下正常运行")
    except Exception as e:
        print(f"  [FAIL] 出现异常: {e}")
        raise

    print()


if __name__ == "__main__":
    test1_level0_silent()
    test2_level1_medium()
    test3_level2_high()
    test4_cooldown()
    test5_interrupt()
    test6_mock_no_hardware()

    print("=" * 55)
    print("全部 6 个测试通过!")
    print("=" * 55)
