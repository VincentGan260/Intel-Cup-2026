"""雷达数据序列化 + 回放器。

- 录制：把每帧 RadarData 序列化成 jsonl（一行一帧），由 scripts/record_radar.py 写入。
- 回放：RadarReplayReader 读 jsonl，逐帧返回 RadarData，接口与 RadarReader 一致
  （start/stop/read_once），可直接喂给融合/管线做**离线、无硬件**的调试与演示。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.fusion.data_types import RadarData, RadarTarget, now


def radar_to_dict(rd: RadarData) -> dict:
    """RadarData → 可 JSON 序列化的 dict。"""
    return {
        "timestamp": rd.timestamp,
        "valid": rd.valid,
        "nearest_distance_m": rd.nearest_distance_m,
        "min_ttc": rd.min_ttc,
        "targets": [
            {
                "target_id": t.target_id,
                "distance_m": t.distance_m,
                "relative_speed_mps": t.relative_speed_mps,
                "angle_deg": t.angle_deg,
                "confidence": t.confidence,
            }
            for t in rd.targets
        ],
    }


def dict_to_radar(d: dict) -> RadarData:
    """dict → RadarData。"""
    targets = [
        RadarTarget(
            target_id=t.get("target_id", i),
            distance_m=t.get("distance_m", 0.0),
            relative_speed_mps=t.get("relative_speed_mps", 0.0),
            angle_deg=t.get("angle_deg", 0.0),
            confidence=t.get("confidence", 0.0),
        )
        for i, t in enumerate(d.get("targets", []))
    ]
    return RadarData(
        timestamp=d.get("timestamp", 0.0),
        valid=d.get("valid", False),
        targets=targets,
        nearest_distance_m=d.get("nearest_distance_m", -1.0),
        min_ttc=d.get("min_ttc", -1.0),
    )


class RadarReplayReader:
    """从 jsonl 回放雷达帧，接口对齐 RadarReader（start/stop/read_once）。

    用法：
        r = RadarReplayReader("logs/radar_xxx.jsonl"); r.start()
        while True:
            data = r.read_once()   # 逐帧返回；耗尽后返回最后一帧（或空），见 loop
    """

    def __init__(self, path: str, loop: bool = False) -> None:
        self.path = Path(path)
        self.loop = loop
        self._frames: List[RadarData] = []
        self._idx = 0

    def start(self) -> None:
        if not self.path.is_file():
            print(f"[RadarReplay] 找不到回放文件: {self.path}")
            return
        with self.path.open(encoding="utf-8") as f:
            self._frames = [dict_to_radar(json.loads(line)) for line in f if line.strip()]
        self._idx = 0
        print(f"[RadarReplay] 载入 {len(self._frames)} 帧：{self.path}")

    def stop(self) -> None:
        self._frames = []
        self._idx = 0

    @property
    def exhausted(self) -> bool:
        return not self.loop and self._idx >= len(self._frames)

    def read_once(self) -> RadarData:
        if not self._frames:
            return RadarData(timestamp=now(), valid=False)
        if self._idx >= len(self._frames):
            if self.loop:
                self._idx = 0
            else:
                return self._frames[-1]   # 耗尽后保持最后一帧
        rd = self._frames[self._idx]
        self._idx += 1
        return rd
