"""雷达实时监视器（桌面 bring-up 用）：连 LD2451，实时打印每帧目标。

★ 接上雷达即可跑。用来：确认有没有数据、挥手/走动看读数变化、
  桌面就能验证「接近=相对速度为负」(speed_dir) 和「角度左右符号」。

运行：
    python scripts/monitor_radar.py --port COM7
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
    ap.add_argument("--port", default="COM7", help="串口号（Win:COM7 / Linux:/dev/ttyUSB0）")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    reader = RadarReader(mode="real", config={"port": args.port, "baudrate": args.baud, "timeout": 0.5})
    reader.start()
    print("实时监视中（Ctrl+C 退出）。挥手/走动应看到读数变化；")
    print("  走近 → 相对速度应为「负(接近)」；站右侧 → 看角度正负是否符合预期。\n")
    period = 1.0 / max(1.0, args.hz)
    try:
        while True:
            rd = reader.read_once()
            if not rd.valid:
                print("  [无效帧 / 未连接]"); time.sleep(period); continue
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
