"""雷达实时监视器（桌面 bring-up 用）：连 LD2451，实时打印每帧目标。

★ 接上雷达即可跑。用来：确认有没有数据、挥手/走动看读数变化、
  桌面就能验证「接近=相对速度为负」(speed_dir) 和「角度左右符号」。

运行：
    python scripts/monitor_radar.py --port COM5
    python scripts/monitor_radar.py --port /dev/ttyUSB0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.sensors.radar_reader import RadarReader


def main() -> None:
    ap = argparse.ArgumentParser(description="LD2451 实时监视（桌面 bring-up）")
    ap.add_argument("--port", default="/dev/ttyUSB1", help="串口号（Win:COM5 / Linux:/dev/ttyUSB1）")
    ap.add_argument("--baud", type=int, default=256000,
                    help="当前实测LD2451为256000；恢复出厂后可能为115200")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--approaching-code", type=int, choices=[0, 1], default=1,
                    help="靠近方向码；当前LD2451真机标定值为1")
    ap.add_argument("--angle-sign", type=int, choices=[-1, 1], default=-1,
                    help="角度符号；当前安装标定-1后满足左负右正")
    args = ap.parse_args()

    reader = RadarReader(mode="real", config={"port": args.port, "baudrate": args.baud,
                                                "timeout": 0.5,
                                                "approaching_direction_code": args.approaching_code,
                                                "angle_sign": args.angle_sign})
    reader.start()
    print("实时监视中（Ctrl+C 退出）。挥手/走动应看到读数变化；")
    print("  走近 → 相对速度应为「负(接近)」；站右侧 → 看角度正负是否符合预期。\n")
    period = 1.0 / max(1.0, args.hz)
    invalid_count = 0
    try:
        while True:
            rd = reader.read_once()
            if not rd.valid:
                invalid_count += 1
                if invalid_count % max(1, int(args.hz * 3)) == 0:
                    print("  [等待完整帧] 无目标时雷达约1秒上报一次；持续无数据再检查连接")
                time.sleep(period); continue
            invalid_count = 0
            if not rd.targets:
                print("  无目标"); time.sleep(period); continue
            parts = []
            for t in rd.targets:
                d = "接近" if t.relative_speed_mps < 0 else "远离"
                side = "右" if t.angle_deg > 0 else ("左" if t.angle_deg < 0 else "正前")
                parts.append(f"[{t.angle_deg:+.0f}°({side}) {t.distance_m:.0f}m "
                             f"{abs(t.relative_speed_mps)*3.6:.0f}km/h {d}]")
            print(f"  目标×{len(rd.targets)}: " + " ".join(parts))
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
