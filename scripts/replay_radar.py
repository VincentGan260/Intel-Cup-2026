"""雷达回放/检查器。

- 查看一段录制：逐帧/统计打印，确认录制是否正常。
- --make-sample：用 mock 雷达生成一段**合成录制**，便于没硬件时测「录制→回放→融合」整条链。

运行：
    D:/Anaconda_envs/envs/intel/python.exe scripts/replay_radar.py --make-sample logs/radar_sample.jsonl --frames 30
    D:/Anaconda_envs/envs/intel/python.exe scripts/replay_radar.py --input logs/radar_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.sensors.radar_replay import RadarReplayReader, radar_to_dict


def make_sample(out: Path, frames: int) -> None:
    """用 mock 雷达生成合成录制（供离线测试链路）。"""
    from src.sensors.radar_reader import RadarReader

    out.parent.mkdir(parents=True, exist_ok=True)
    reader = RadarReader(mode="mock")
    reader.start()
    with out.open("w", encoding="utf-8") as f:
        for _ in range(frames):
            rd = reader.read_once()
            f.write(json.dumps(radar_to_dict(rd), ensure_ascii=False) + "\n")
    reader.stop()
    print(f"已生成合成录制：{out}（{frames} 帧）")


def inspect(path: Path, limit: int) -> None:
    r = RadarReplayReader(str(path))
    r.start()
    n = len(r._frames)
    n_with_target = sum(1 for fr in r._frames if fr.targets)
    print(f"总帧 {n}，含目标 {n_with_target}\n")
    for i in range(min(limit, n)):
        rd = r.read_once()
        tg = ", ".join(
            f"{t.angle_deg:+.0f}°/{t.distance_m:.0f}m/{t.relative_speed_mps:+.1f}mps"
            for t in rd.targets
        ) or "(无目标)"
        print(f"  帧{i:>3} valid={rd.valid} nearest={rd.nearest_distance_m:.1f} "
              f"ttc={rd.min_ttc:.1f}  目标: {tg}")


def main() -> None:
    ap = argparse.ArgumentParser(description="雷达回放/检查器")
    ap.add_argument("--input", help="要查看的录制 jsonl")
    ap.add_argument("--make-sample", help="生成合成录制到此路径")
    ap.add_argument("--frames", type=int, default=30, help="--make-sample 的帧数")
    ap.add_argument("--limit", type=int, default=20, help="--input 最多打印帧数")
    args = ap.parse_args()

    if args.make_sample:
        make_sample(Path(args.make_sample), args.frames)
    elif args.input:
        inspect(Path(args.input), args.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
