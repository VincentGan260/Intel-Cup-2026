"""RiderGuardian 现场录制一键启动脚本

功能：
  1. 启动前自动检测所有设备连接状态
  2. 等待GPS获得定位（可选）
  3. 实时显示各传感器采集状态
  4. 支持交互式场景命名
  5. 录制结束后显示统计报告

使用方式：
  python scripts/field_record.py --scene road_test
  python scripts/field_record.py --scene intersection --wait-gps
  python scripts/field_record.py --scene campus --duration 120
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_device(port: str, name: str) -> tuple[bool, str]:
    """检查设备端口是否存在"""
    if port.startswith("/dev/"):
        exists = os.path.exists(port)
        return exists, "✓" if exists else "✗"
    return True, "?"


def print_device_status(config: dict) -> None:
    """打印所有设备状态"""
    print("\n" + "=" * 50)
    print("       RiderGuardian 设备检测")
    print("=" * 50)
    
    devices = [
        ("GPS", config.get("gps", {}).get("port", "")),
        ("雷达", config.get("radar", {}).get("port", "")),
        ("IMU", config.get("imu", {}).get("port", "")),
    ]
    
    for name, port in devices:
        exists, icon = check_device(port, name)
        status = "已连接" if exists else "未连接"
        print(f"  {icon} {name}: {port} ({status})")
    
    cam_id = config.get("camera", {}).get("device_id", 0)
    print(f"  ? 摄像头: device_id={cam_id}")
    print("=" * 50 + "\n")


def wait_gps_fix(timeout: int = 60) -> bool:
    """等待GPS获得定位信号"""
    print(f"等待GPS定位中（最多{timeout}秒）...")
    
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.sensors.gps_reader import GPSReader
    
    gps = GPSReader(mode="real", config={"port": "/dev/ttyGPSNEO", "baudrate": 9600, "timeout": 1.0})
    try:
        gps.start()
        start = time.time()
        while time.time() - start < timeout:
            data = gps.read_once()
            if data.valid and data.fix_quality >= 1:
                print(f"✓ GPS定位成功! 卫星数: {data.satellites}, 精度: {data.hdop:.1f}")
                gps.stop()
                return True
            print(f"  等待中... fix={data.fix_quality}, 卫星={data.satellites}", end="\r")
            time.sleep(0.5)
        print(f"\n✗ GPS定位超时({timeout}秒)，继续录制")
        gps.stop()
        return False
    except Exception as e:
        print(f"✗ GPS读取失败: {e}，继续录制")
        return False


def run_recording(scene: str, profile: str, duration: float, session_name: str = "") -> None:
    """运行录制命令并实时显示状态"""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "record_multimodal.py"),
        "--mode", "real",
        "--profile", profile,
        "--scene", scene,
    ]
    if duration > 0:
        cmd.extend(["--duration", str(duration)])
    if session_name:
        cmd.extend(["--session-name", session_name])
    
    print(f"\n启动录制: {' '.join(cmd)}")
    print("=" * 50)
    print("正在录制... (按 Ctrl+C 停止)")
    print("=" * 50)
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    try:
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"  {line}")
    except KeyboardInterrupt:
        print("\n\n正在停止录制...")
        process.terminate()
    
    process.wait()
    print("\n" + "=" * 50)
    print("录制结束")
    print("=" * 50)


def show_recording_report(session_dir: Path) -> None:
    """显示录制统计报告"""
    session_file = session_dir / "session.json"
    if not session_file.exists():
        print("✗ 未找到 session.json")
        return
    
    with open(session_file, "r") as f:
        metadata = json.load(f)
    
    counts = metadata.get("counts", {})
    duration_ms = metadata.get("ended_wall_time_ns", 0) - metadata.get("started_wall_time_ns", 0)
    duration_sec = duration_ms / 1_000_000_000.0 if duration_ms > 0 else 0
    
    print(f"\n📊 录制报告")
    print(f"┌─────────────────────────────────────────────┐")
    print(f"│ 场景: {metadata.get('scene', '')}")
    print(f"│ 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata.get('started_wall_time_ns', 0)/1_000_000_000))}")
    print(f"│ 时长: {duration_sec:.1f}秒")
    print(f"├─────────────────────────────────────────────┤")
    print(f"│ 摄像头: {counts.get('camera', 0)}帧 ({counts.get('camera', 0)/duration_sec:.1f} fps)")
    print(f"│ 雷达: {counts.get('radar', 0)}帧 ({counts.get('radar', 0)/duration_sec:.1f} hz)")
    print(f"│ GPS: {counts.get('gps', 0)}帧")
    print(f"│ IMU: {counts.get('imu', 0)}帧")
    print(f"├─────────────────────────────────────────────┤")
    print(f"│ 保存路径: {session_dir}")
    print(f"└─────────────────────────────────────────────┘")


def main() -> int:
    ap = argparse.ArgumentParser(description="RiderGuardian 现场录制一键启动")
    ap.add_argument("--scene", required=True, help="场景名称（如：road_test, intersection, campus）")
    ap.add_argument("--profile", choices=["windows", "dk2500"], default="dk2500")
    ap.add_argument("--duration", type=float, default=0.0, help="录制时长（秒），0为无限")
    ap.add_argument("--session-name", default="", help="会话名称后缀")
    ap.add_argument("--wait-gps", action="store_true", help="等待GPS获得定位后再开始录制")
    ap.add_argument("--no-check", action="store_true", help="跳过设备检测")
    args = ap.parse_args()
    
    print("\n🚀 RiderGuardian 现场录制系统")
    
    if not args.no_check:
        import yaml
        with open(PROJECT_ROOT / "configs" / "sensor_ports.yaml", "r") as f:
            config = yaml.safe_load(f)
        print_device_status(config)
    
    if args.wait_gps:
        wait_gps_fix()
    
    run_recording(args.scene, args.profile, args.duration, args.session_name)
    
    latest_dir = None
    recordings_dir = PROJECT_ROOT / "data" / "recordings"
    if recordings_dir.exists():
        dirs = sorted(recordings_dir.glob(f"*_{args.scene}*"), key=lambda x: x.stat().st_mtime)
        if dirs:
            latest_dir = dirs[-1]
    
    if latest_dir:
        show_recording_report(latest_dir)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
