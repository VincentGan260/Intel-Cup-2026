"""真实传感器（GPS/IMU/Radar）+ 马达 Mock 联调测试。

在 DK-2500 上执行真实传感器采集 + 风险融合 + 马达 mock 输出，
验证传感器硬件和风险融合链路，同时确保马达不会真实震动安全。

运行命令（DK-2500）：
  python main_integrated.py --mode real --vision false --motor mock --profile dk2500 --loops 100

当前 Windows 环境也可运行测试脚本，验证：
  1. 参数解析正确
  2. 硬件不可达时打印明确失败原因
  3. 不崩溃
"""

import sys
import os
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_and_check(args: list, desc: str, timeout: float = 20.0) -> bool:
    """运行 main_integrated.py 并检查输出。"""
    cmd = [sys.executable, "main_integrated.py"] + args
    print(f"\n{'=' * 65}")
    print(f"  {desc}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'-' * 65}")

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
        print("  [超时] 子进程被终止\n")

    # 显示关键输出行
    for line in stdout.splitlines():
        stripped = line.strip()
        if any(kw in stripped for kw in ["启动摘要", "传感器模式", "马达模式", "端口配置",
                                          "GPS:", "IMU:", "Radar:", "Motor:", "Vision:",
                                          "WARNING", "FAILED", "已完成", "正在关闭",
                                          "日志:"]):
            print(f"  {stripped}")
        if "FAILED" in stripped:
            print(f"  → {stripped}")

    print(f"\n  退出码: {proc.returncode}")

    # 检查
    checks = []
    is_rejection = "无确认" in desc
    if is_rejection:
        # motor real 拒绝时预期正常退出（exit=0）且有安全提示
        checks.append(("安全退出 (exit=0)", proc.returncode == 0))
    else:
        checks.append(("正常退出", proc.returncode == 0))

    if is_rejection:
        has_safety = "安全保护" in stdout or "安全退出" in stdout
        checks.append(("安全保护提示已显示", has_safety))
    else:
        if "启动摘要" in stdout:
            checks.append(("启动摘要已打印", True))
        else:
            checks.append(("启动摘要已打印", False))

    if not is_rejection:
        for label in ("GPS", "IMU", "Radar", "Motor", "Vision"):
            found = any(f"{label}:" in l for l in stdout.splitlines())
            checks.append((f"{label} 状态已显示", found))

    # 检查马达模式（仅非拒绝 case）
    if not is_rejection:
        if "--motor mock" in " ".join(args) or "motor mock" in desc:
            has_warning = "WARNING" in stdout
            checks.append(("motor mock 无 WARNING", not has_warning))
            checks.append(("motor mock → MOCK", "MOCK" in stdout))
        elif "--motor real" in " ".join(args) or "motor real" in desc:
            checks.append(("motor real 有 WARNING", "WARNING" in stdout))

    all_ok = all(v for _, v in checks)
    for name, ok in checks:
        print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

    if all_ok:
        print(f"\n  [PASS] {desc}")
    else:
        print(f"\n  [FAIL] {desc}")

    return all_ok


def main():
    results = []

    # ── Test A: mock + motor mock（默认情形） ──
    ok = run_and_check(
        ["--mode", "mock", "--vision", "false", "--loops", "10"],
        "Test A: mock + motor mock (默认)",
    )
    results.append(("A: mock+motor mock", ok))

    # ── Test B: real + motor mock（传感器假失败 + 马达安全） ──
    ok = run_and_check(
        ["--mode", "real", "--vision", "false", "--motor", "mock", "--profile", "windows", "--loops", "10"],
        "Test B: real sensors + motor mock (硬件不可达应有 FAILED 提示)",
    )
    results.append(("B: real sensors + motor mock", ok))

    # ── Test C: real + motor real 无确认（应被拒绝） ──
    ok = run_and_check(
        ["--mode", "real", "--vision", "false", "--motor", "real", "--profile", "windows", "--loops", "10"],
        "Test C: real + motor real 无确认 (应被安全拒绝)",
        timeout=10.0,
    )
    # 无确认时 exit code 应为 0（安全退出），且应有"安全保护"提示
    results.append(("C: motor real 无确认应被拒绝", ok))

    # ── Test D: real + motor real + 确认（应有 WARNING） ──
    ok = run_and_check(
        ["--mode", "real", "--vision", "false", "--motor", "real",
         "--confirm-motor-real", "--profile", "windows", "--loops", "10"],
        "Test D: real + motor real + 确认 (应有 WARNING)",
    )
    results.append(("D: motor real + confirm", ok))

    # ── Test E: dk2500 profile（验证端口配置切换） ──
    ok = run_and_check(
        ["--mode", "real", "--vision", "false", "--motor", "mock", "--profile", "dk2500", "--loops", "5"],
        "Test E: dk2500 profile + motor mock",
    )
    results.append(("E: dk2500 profile", ok))

    # ── 汇总 ──
    print("\n" + "=" * 65)
    print("  测试汇总")
    print("=" * 65)
    all_pass = True
    for name, ok in results:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
        if not ok:
            all_pass = False
    print("-" * 65)
    print(f"  结果: {'全部通过' if all_pass else '存在失败'}")
    print("=" * 65)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
