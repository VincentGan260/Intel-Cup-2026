"""RiderGuardian实地测试一键入口（相机 + LD2451 + GPS，不使用IMU）。

流程：配置/设备预检 -> 可选等待GPS定位 -> 启动三路录制 ->
安全停止 -> 自动离线同步 -> 自动质量检查 -> 显示session路径。

示例：
  python scripts/field_record.py --scene low_clear --duration 60
  python scripts/field_record.py --scene high_crossing --duration 30 --skip-gps-fix
  python scripts/field_record.py --scene desk_test --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"


def load_config(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "sensor_ports.yaml").read_text(encoding="utf-8"))
    if profile not in raw:
        raise ValueError(f"profile不存在: {profile}")
    return raw[profile], raw.get("camera", {})


def serial_port_exists(port: str) -> bool:
    if os.name == "nt":
        try:
            from serial.tools import list_ports
            return port.upper() in {p.device.upper() for p in list_ports.comports()}
        except ImportError:
            return False
    return Path(port).exists()


def preflight(profile: str, ports: dict[str, Any], camera: dict[str, Any], check_camera: bool = True) -> list[str]:
    problems: list[str] = []
    selected = {name: str(ports.get(name, {}).get("port", "")).strip() for name in ("gps", "radar")}
    if any(not port for port in selected.values()):
        problems.append("GPS或雷达串口配置为空")
    if len({p.lower() for p in selected.values() if p}) != len(selected):
        problems.append(f"GPS与雷达不能共用串口: {selected}")
    for name, port in selected.items():
        if port and not serial_port_exists(port):
            problems.append(f"{name}端口不存在: {port}")

    if check_camera:
        try:
            import cv2
            camera_id = int(camera.get("device_id", 0))
            cap = cv2.VideoCapture(camera_id)
            ok, _ = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if not ok:
                problems.append(f"相机无法读取: device_id={camera_id}")
        except ImportError as exc:
            problems.append(f"OpenCV不可用: {exc}")

    print("\n设备预检")
    print(f"  profile: {profile}")
    print(f"  GPS:   {selected['gps']} @ {ports.get('gps', {}).get('baudrate')}")
    print(f"  雷达:  {selected['radar']} @ {ports.get('radar', {}).get('baudrate')}")
    print(f"  相机:  device_id={camera.get('device_id', 0)} "
          f"{camera.get('width', 640)}x{camera.get('height', 480)}@{camera.get('fps', 30)}")
    print("  IMU:   不使用")
    print("  马达:  不启动（采集安全策略）")
    for problem in problems:
        print(f"  [FAIL] {problem}")
    if not problems:
        print("  [PASS] 配置、串口和相机预检通过")
    return problems


def wait_for_gps(config: dict[str, Any], timeout_sec: int) -> bool:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.sensors.gps_reader import GPSReader

    reader = GPSReader("real", config)
    reader.start()
    if getattr(reader, "_serial", None) is None:
        print("[FAIL] GPS串口无法打开")
        return False
    deadline = time.monotonic() + timeout_sec
    last_print = 0.0
    try:
        while time.monotonic() < deadline:
            data = reader.read_once()
            if data.valid:
                print(f"[PASS] GPS定位：sat={data.satellites}, fix={data.fix_quality}, "
                      f"lat={data.latitude:.6f}, lon={data.longitude:.6f}")
                return True
            if time.monotonic() - last_print >= 2.0:
                print(f"  等待GPS：sat={data.satellites}, fix={data.fix_quality}")
                last_print = time.monotonic()
        print(f"[FAIL] {timeout_sec}s内GPS未定位")
        return False
    finally:
        reader.stop()


def newest_session(scene: str, started_after: float) -> Path | None:
    if not RECORDINGS_DIR.exists():
        return None
    candidates = [p for p in RECORDINGS_DIR.glob(f"*_{scene}*") if p.is_dir() and p.stat().st_mtime >= started_after]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def run_command(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def record(args: argparse.Namespace) -> tuple[int, Path | None]:
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "record_multimodal.py"),
        "--mode", "real", "--profile", args.profile, "--scene", args.scene,
    ]
    if args.duration > 0:
        command.extend(["--duration", str(args.duration)])
    if args.session_name:
        command.extend(["--session-name", args.session_name])

    started = time.time() - 1.0
    print("\n开始录制；按Ctrl+C安全停止。")
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\n正在通知录制进程安全停止...")
        process.send_signal(signal.SIGINT)
        try:
            return_code = process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            print("[WARN] 录制进程未及时退出，执行终止")
            process.terminate()
            return_code = process.wait(timeout=5)
    return return_code, newest_session(args.scene, started)


def postprocess(session: Path) -> int:
    sync_code = run_command([sys.executable, str(PROJECT_ROOT / "scripts" / "sync_recording.py"), str(session)])
    if sync_code:
        return sync_code
    return run_command([
        sys.executable, str(PROJECT_ROOT / "scripts" / "check_recording.py"),
        str(session), "--require-fusion",
    ])


def print_report(session: Path) -> None:
    session_json = session / "session.json"
    quality_json = session / "quality_report.json"
    metadata = json.loads(session_json.read_text(encoding="utf-8")) if session_json.exists() else {}
    quality = json.loads(quality_json.read_text(encoding="utf-8")) if quality_json.exists() else {}
    print("\n" + "=" * 64)
    print(f"Session: {session}")
    print(f"Counts:  {metadata.get('counts', {})}")
    print(f"Quality problems: {quality.get('problems', ['质量报告未生成'])}")
    print("=" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description="RiderGuardian实地测试一键启动（无IMU）")
    ap.add_argument("--scene", required=True, help="如 low_clear/mid_approach/high_crossing/high_braking")
    ap.add_argument("--profile", choices=["windows", "dk2500"], default="dk2500")
    ap.add_argument("--duration", type=float, default=0.0, help="录制秒数；0表示Ctrl+C停止")
    ap.add_argument("--session-name", default="")
    ap.add_argument("--gps-timeout", type=int, default=90)
    ap.add_argument("--skip-gps-fix", action="store_true", help="允许GPS未定位时开始（桌面测试用）")
    ap.add_argument("--dry-run", action="store_true", help="只做预检，不开始录制")
    ap.add_argument("--no-postprocess", action="store_true")
    args = ap.parse_args()

    try:
        ports, camera = load_config(args.profile)
    except Exception as exc:
        print(f"[FAIL] 配置读取失败: {exc}", file=sys.stderr)
        return 2
    problems = preflight(args.profile, ports, camera)
    if problems:
        return 2
    if args.dry_run:
        return 0
    if not args.skip_gps_fix and not wait_for_gps(ports["gps"], args.gps_timeout):
        print("为避免产生无位置训练数据，本次不开始录制；桌面测试可显式传--skip-gps-fix。")
        return 3

    code, session = record(args)
    if session is None:
        print("[FAIL] 未找到本次session目录", file=sys.stderr)
        return code or 4
    if code:
        print(f"[WARN] 录制进程退出码={code}，仍保留session用于排查")
    post_code = 0 if args.no_postprocess else postprocess(session)
    print_report(session)
    return code or post_code


if __name__ == "__main__":
    raise SystemExit(main())
