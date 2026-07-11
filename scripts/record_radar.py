"""雷达数据录制器：连 LD2451，把每帧 RadarData 存成 jsonl（供离线回放调融合）。

★ 需要真实硬件 + 串口。软件侧逻辑已就绪，接上雷达即可录。

运行（在接了雷达的机器上）：
    python scripts/record_radar.py --port COM5 --duration 60
    python scripts/record_radar.py --port /dev/ttyUSB0 --frames 600 --out logs/radar_run1.jsonl

录完用 scripts/replay_radar.py 回放。
"""
from __future__ import annotations

import argparse
import json
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
from src.sensors.radar_replay import radar_to_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="LD2451 雷达录制器（存 jsonl）")
    ap.add_argument("--port", default="/dev/ttyUSB1", help="串口号（Win:COM5 / Linux:/dev/ttyUSB1）")
    ap.add_argument("--baud", type=int, default=256000,
                    help="当前实测LD2451为256000；恢复出厂后可能为115200")
    ap.add_argument("--duration", type=float, default=0.0, help="录制秒数（>0 时按时长）")
    ap.add_argument("--frames", type=int, default=0, help="录制帧数（>0 时按帧数）")
    ap.add_argument("--out", default=None, help="输出 jsonl 路径；默认 logs/radar_<时间戳>.jsonl")
    ap.add_argument("--all", action="store_true", help="连无效帧也录（默认只录 valid 帧）")
    ap.add_argument("--hz", type=float, default=20.0, help="读取频率（雷达 10Hz 输出，默认 20Hz 轮询）")
    ap.add_argument("--approaching-code", type=int, choices=[0, 1], default=1,
                    help="靠近方向码；当前LD2451真机标定值为1")
    ap.add_argument("--angle-sign", type=int, choices=[-1, 1], default=-1,
                    help="当前安装标定-1后满足左负右正")
    args = ap.parse_args()

    out = Path(args.out) if args.out else PROJECT_ROOT / "logs" / f"radar_{int(time.time())}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = RadarReader(mode="real", config={"port": args.port, "baudrate": args.baud,
                                               "timeout": 0.5,
                                               "approaching_direction_code": args.approaching_code,
                                               "angle_sign": args.angle_sign})
    reader.start()

    n_total = n_valid = n_with_target = 0
    period = 1.0 / max(1.0, args.hz)
    t0 = time.time()
    print(f"开始录制 → {out}（Ctrl+C 结束）")
    try:
        with out.open("w", encoding="utf-8") as f:
            while True:
                rd = reader.read_once()
                if args.all or rd.valid:
                    f.write(json.dumps(radar_to_dict(rd), ensure_ascii=False) + "\n")
                    f.flush()
                    n_total += 1
                    if rd.valid:
                        n_valid += 1
                    if rd.targets:
                        n_with_target += 1
                if args.duration > 0 and time.time() - t0 >= args.duration:
                    break
                if args.frames > 0 and n_total >= args.frames:
                    break
                time.sleep(period)
    except KeyboardInterrupt:
        print("\n手动结束")
    finally:
        reader.stop()

    print(f"录制完成：{out}")
    print(f"  共写入 {n_total} 帧（有效 {n_valid}，含目标 {n_with_target}）")


if __name__ == "__main__":
    main()
