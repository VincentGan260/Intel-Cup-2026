"""DK-2500 上车前硬件预检脚本。

功能：
  - 逐个检测 GPS、IMU、雷达、马达、视觉模块
  - 每个模块读取 N 帧，统计 valid_count / fail_count
  - 输出最近一次有效数据
  - 马达仅检测 I2C 通信，不触发真实震动
  - 不进入风险融合闭环，只做硬件可用性验证

运行方式：
  python scripts/preflight_dk2500.py
  python scripts/preflight_dk2500.py --profile dk2500 --loops 10

参数：
  --profile {windows,dk2500}   端口配置 profile（默认从 config.yaml 读取）
  --loops N                    每个传感器采样帧数（默认 10）
  --vision                     是否检测视觉模块（默认 false）
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion.data_types import now

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


def _load_sensor_ports(profile: str) -> dict:
    cfg = _load_config("configs/sensor_ports.yaml")
    return cfg.get(profile, {})


def _load_system_config() -> dict:
    cfg = _load_config("configs/config.yaml")
    return cfg.get("system", {})


# ============================================================
#  模块检测函数
# ============================================================


def check_gps(ports: dict, loops: int) -> dict:
    """检测 GPS 模块。"""
    from src.sensors.gps_reader import GPSReader

    result = {"mode": "real", "status": "N/A", "valid_count": 0, "fail_count": 0,
              "last_valid": None, "error": None}

    cfg = ports.get("gps", {})
    result["port"] = cfg.get("port", "N/A")
    result["baudrate"] = cfg.get("baudrate", 9600)

    reader = GPSReader(mode="real", config=cfg)
    try:
        reader.start()
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        return result

    # 检查串口是否真正打开（有些 reader 在 start 中静默失败后回退到 mock 数据）
    if hasattr(reader, "_serial") and reader._serial is None:
        result["status"] = "FAILED"
        result["error"] = f"串口 {result['port']} 未能打开"
        reader.stop()
        return result

    deadline = time.time() + 30  # 每模块总超时 30 秒
    for i in range(loops):
        if time.time() > deadline:
            result["error"] = f"模块超时（已采样 {i}/{loops} 帧）"
            break
        sys.stdout.write(f"    [{i+1}/{loops}] ")
        sys.stdout.flush()
        try:
            data = reader.read_once()
            if data.valid:
                result["valid_count"] += 1
                result["last_valid"] = {
                    "speed_kmh": round(data.speed_kmh, 2),
                    "lat": round(data.latitude, 6),
                    "lon": round(data.longitude, 6),
                }
                sys.stdout.write("valid\n")
            else:
                result["fail_count"] += 1
                sys.stdout.write("-\n")
        except Exception as e:
            result["fail_count"] += 1
            result["error"] = str(e)
            sys.stdout.write(f"ERR: {e}\n")
        sys.stdout.flush()
        time.sleep(0.1)

    result["status"] = "OK" if result["valid_count"] > 0 else "FAILED"
    if result["status"] == "FAILED" and result["error"] is None:
        result["error"] = f"0/{loops} 帧有效"

    reader.stop()
    return result


def check_imu(ports: dict, loops: int) -> dict:
    """检测 IMU 模块。"""
    from src.sensors.imu_reader import IMUReader

    result = {"mode": "real", "status": "N/A", "valid_count": 0, "fail_count": 0,
              "last_valid": None, "error": None}

    cfg = ports.get("imu", {})
    result["port"] = cfg.get("port", "N/A")
    result["baudrate"] = cfg.get("baudrate", 115200)

    reader = IMUReader(mode="real", config=cfg)
    try:
        reader.start()
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        return result

    # 检查串口是否真正打开
    if hasattr(reader, "_serial") and reader._serial is None:
        result["status"] = "FAILED"
        result["error"] = f"串口 {result['port']} 未能打开"
        reader.stop()
        return result

    deadline = time.time() + 30  # 每模块总超时 30 秒
    for i in range(loops):
        if time.time() > deadline:
            result["error"] = f"模块超时（已采样 {i}/{loops} 帧）"
            break
        sys.stdout.write(f"    [{i+1}/{loops}] ")
        sys.stdout.flush()
        try:
            data = reader.read_once()
            if data.valid:
                result["valid_count"] += 1
                result["last_valid"] = {
                    "roll": round(data.roll, 2),
                    "pitch": round(data.pitch, 2),
                    "yaw": round(data.yaw, 2),
                }
                sys.stdout.write("valid\n")
            else:
                result["fail_count"] += 1
                sys.stdout.write("-\n")
        except Exception as e:
            result["fail_count"] += 1
            result["error"] = str(e)
            sys.stdout.write(f"ERR: {e}\n")
        sys.stdout.flush()
        time.sleep(0.1)

    result["status"] = "OK" if result["valid_count"] > 0 else "FAILED"
    if result["status"] == "FAILED" and result["error"] is None:
        result["error"] = f"0/{loops} 帧有效"

    reader.stop()
    return result


def check_radar(ports: dict, loops: int) -> dict:
    """检测雷达模块。"""
    from src.sensors.radar_reader import RadarReader

    result = {"mode": "real", "status": "N/A", "valid_count": 0, "fail_count": 0,
              "last_valid": None, "error": None}

    cfg = ports.get("radar", {})
    result["port"] = cfg.get("port", "N/A")
    result["baudrate"] = cfg.get("baudrate", 115200)

    reader = RadarReader(mode="real", config=cfg)
    try:
        reader.start()
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        return result

    # 检查串口是否真正打开
    if hasattr(reader, "_serial") and reader._serial is None:
        result["status"] = "FAILED"
        result["error"] = f"串口 {result['port']} 未能打开"
        reader.stop()
        return result

    deadline = time.time() + 30  # 每模块总超时 30 秒
    for i in range(loops):
        if time.time() > deadline:
            result["error"] = f"模块超时（已采样 {i}/{loops} 帧）"
            break
        sys.stdout.write(f"    [{i+1}/{loops}] ")
        sys.stdout.flush()
        try:
            data = reader.read_once()
            if data.valid:
                result["valid_count"] += 1
                result["last_valid"] = {
                    "target_count": len(data.targets),
                    "nearest_m": round(data.nearest_distance_m, 2),
                    "min_ttc": round(data.min_ttc, 2),
                }
                sys.stdout.write("valid\n")
            else:
                result["fail_count"] += 1
                sys.stdout.write("-\n")
        except Exception as e:
            result["fail_count"] += 1
            result["error"] = str(e)
            sys.stdout.write(f"ERR: {e}\n")
        sys.stdout.flush()
        time.sleep(0.1)

    result["status"] = "OK" if result["valid_count"] > 0 else "FAILED"
    if result["status"] == "FAILED" and result["error"] is None:
        result["error"] = f"0/{loops} 帧有效"

    reader.stop()
    return result


def check_motor(ports: dict) -> dict:
    """检测马达 I2C 通信（不触发震动）。

    尝试初始化 I2C 总线并检测 DRV2605 地址，
    如果初始化成功但不想触发震动，则立即关闭。
    """
    result = {"mode": "real", "status": "N/A", "error": None}

    cfg = ports.get("motor", {})
    result["i2c_bus"] = cfg.get("i2c_bus", "N/A")
    result["address"] = cfg.get("driver_address", "N/A")

    try:
        from src.actuator.motor_controller import MotorController

        motor = MotorController(
            mode="real",
            i2c_bus=int(cfg.get("i2c_bus", 1)),
            i2c_addr=int(cfg.get("driver_address", "0x5A"), 16),
        )
        motor.start()
        # 检测是否自动回退到了 mock 模式
        if not motor.is_real:
            result["status"] = "FAILED"
            result["error"] = "I2C 初始化失败，已自动回退至 mock 模式"
        else:
            result["status"] = "OK"
        motor.stop()
    except ImportError as e:
        result["status"] = "SKIPPED"
        result["error"] = f"smbus2 库不可用: {e}"
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)

    return result


def check_vision() -> dict:
    """检测视觉模块是否可用（不加载模型）。"""
    result = {"enabled": True, "status": "N/A", "error": None}

    try:
        from src.fusion.vision_adapter import VisionAdapter

        adapter = VisionAdapter(vision_enabled=True)
        adapter.start()
        if adapter.vision_enabled:
            result["status"] = "OK"
        else:
            result["status"] = "DEGRADED"
            result["error"] = "模型加载失败（openvino 可能不可用）"
        adapter.stop()
    except ImportError as e:
        result["status"] = "DEGRADED"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "DEGRADED"
        result["error"] = str(e)

    return result


# ============================================================
#  输出
# ============================================================


def print_gps(r: dict) -> None:
    print(f"  GPS:")
    print(f"    mode:         {r['mode']}")
    print(f"    port:         {r.get('port', 'N/A')}")
    print(f"    baudrate:     {r.get('baudrate', 'N/A')}")
    print(f"    status:       {r['status']}")
    print(f"    valid_count:  {r['valid_count']}")
    print(f"    fail_count:   {r['fail_count']}")
    if r.get("last_valid"):
        lv = r["last_valid"]
        print(f"    last_speed:   {lv.get('speed_kmh', 'N/A')} km/h")
        print(f"    last_pos:     ({lv.get('lat', 'N/A')}, {lv.get('lon', 'N/A')})")
    if r.get("error"):
        print(f"    error:        {r['error']}")
    print()


def print_imu(r: dict) -> None:
    print(f"  IMU:")
    print(f"    mode:         {r['mode']}")
    print(f"    port:         {r.get('port', 'N/A')}")
    print(f"    baudrate:     {r.get('baudrate', 'N/A')}")
    print(f"    status:       {r['status']}")
    print(f"    valid_count:  {r['valid_count']}")
    print(f"    fail_count:   {r['fail_count']}")
    if r.get("last_valid"):
        lv = r["last_valid"]
        print(f"    last_rpy:     ({lv.get('roll', 'N/A')}, "
              f"{lv.get('pitch', 'N/A')}, {lv.get('yaw', 'N/A')})")
    if r.get("error"):
        print(f"    error:        {r['error']}")
    print()


def print_radar(r: dict) -> None:
    print(f"  Radar:")
    print(f"    mode:         {r['mode']}")
    print(f"    port:         {r.get('port', 'N/A')}")
    print(f"    baudrate:     {r.get('baudrate', 'N/A')}")
    print(f"    status:       {r['status']}")
    print(f"    valid_count:  {r['valid_count']}")
    print(f"    fail_count:   {r['fail_count']}")
    if r.get("last_valid"):
        lv = r["last_valid"]
        print(f"    target_count:  {lv.get('target_count', 'N/A')}")
        print(f"    nearest_dist:  {lv.get('nearest_m', 'N/A')} m")
        print(f"    min_ttc:       {lv.get('min_ttc', 'N/A')} s")
    if r.get("error"):
        print(f"    error:        {r['error']}")
    print()


def print_motor(r: dict) -> None:
    print(f"  Motor:")
    print(f"    mode:         {r['mode']}")
    print(f"    i2c_bus:      {r.get('i2c_bus', 'N/A')}")
    print(f"    address:      {r.get('address', 'N/A')}")
    print(f"    status:       {r['status']}")
    if r.get("error"):
        print(f"    error:        {r['error']}")
    print()


def print_vision(r: dict) -> None:
    print(f"  Vision:")
    print(f"    enabled:      {r.get('enabled', False)}")
    print(f"    status:       {r['status']}")
    if r.get("error"):
        print(f"    error:        {r['error']}")
    print()


# ============================================================
#  主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="DK-2500 上车前硬件预检")
    parser.add_argument("--profile", type=str, default=None,
                        choices=["windows", "dk2500"], help="端口配置 profile")
    parser.add_argument("--loops", type=int, default=10,
                        help="每个传感器采样帧数（默认 10）")
    parser.add_argument("--vision", action="store_true",
                        help="是否检测视觉模块")
    args = parser.parse_args()

    sys_cfg = _load_system_config()
    profile = args.profile if args.profile is not None else sys_cfg.get("platform", "windows")
    loops = max(1, args.loops)

    ports = _load_sensor_ports(profile)

    print("=" * 65)
    print("  DK-2500 上车前硬件预检")
    print(f"  profile: {profile}  |  每模块采样: {loops} 帧")
    print("=" * 65)
    print()
    print("  注意: 本预检不会触发马达真实震动。")
    print("  视觉模块仅在 --vision 参数下检测。")
    print()

    results = {}

    # ── GPS ──
    print("-" * 65)
    r_gps = check_gps(ports, loops)
    results["GPS"] = r_gps
    print_gps(r_gps)

    # ── IMU ──
    print("-" * 65)
    r_imu = check_imu(ports, loops)
    results["IMU"] = r_imu
    print_imu(r_imu)

    # ── Radar ──
    print("-" * 65)
    r_radar = check_radar(ports, loops)
    results["Radar"] = r_radar
    print_radar(r_radar)

    # ── Motor ──
    print("-" * 65)
    r_motor = check_motor(ports)
    results["Motor"] = r_motor
    print_motor(r_motor)

    # ── Vision（可选） ──
    if args.vision:
        print("-" * 65)
        r_vision = check_vision()
        results["Vision"] = r_vision
        print_vision(r_vision)

    # ── 汇总 ──
    print("=" * 65)
    print("  预检汇总")
    print("-" * 65)
    all_ok = True
    for label in ("GPS", "IMU", "Radar", "Motor"):
        r = results.get(label, {})
        st = r.get("status", "N/A")
        valid = r.get("valid_count", 0)
        fail = r.get("fail_count", 0)
        ok = st == "OK"
        print(f"  {label + ':':8s}  status={st:10s}  valid={valid}  fail={fail}")
        if not ok:
            all_ok = False

    if args.vision:
        r = results.get("Vision", {})
        st = r.get("status", "N/A")
        print(f"  {'Vision:':8s}  status={st}")
        if st != "OK":
            all_ok = False

    print("-" * 65)
    if all_ok:
        print("  所有模块 OK，可以上车联调。")
    else:
        print("  部分模块异常，请检查上述错误信息后再联调。")
    print("=" * 65)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
