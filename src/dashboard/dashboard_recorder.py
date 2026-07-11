"""Session writer used by the real-sensor Dashboard.

Each JSONL row is one training sample containing aligned sensor readings,
vision features and vision/radar association output. Raw frames are retained so
features can be regenerated after a model update.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


class DashboardRecorder:
    def __init__(self, output: Path, scene: str, profile: str) -> None:
        label = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = (output / f"{label}_{scene}_dashboard").resolve()
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self._file = (self.session_dir / "samples.jsonl").open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._started_mono_ns = time.monotonic_ns()
        self._count = 0
        self._meta = {"schema_version": 1, "scene": scene, "profile": profile,
                      "host": platform.node(), "started_wall_time_ns": time.time_ns(),
                      "alignment_clock": "monotonic_ns",
                      "contains": ["raw_frame", "radar", "gps", "vision_features", "vision_radar_fusion"]}
        self._write_meta()

    def _write_meta(self) -> None:
        (self.session_dir / "session.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def write(self, frame, radar, gps, vision, fusion, inference_ms: float) -> None:
        import cv2

        mono_ns = time.monotonic_ns()
        wall_ns = time.time_ns()
        frame_path = ""
        frame_valid = frame is not None
        if frame_valid:
            name = f"frame_{self._count:06d}_{wall_ns}.jpg"
            frame_valid = bool(cv2.imwrite(str(self.frames_dir / name), frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, 92]))
            if frame_valid:
                frame_path = f"frames/{name}"
        vision_features = None
        if vision is not None:
            vision_features = {
                "valid": bool(vision.valid),
                "obstacle_count": len(vision.objects),
                "person_count": int(vision.person_count),
                "vehicle_count": int(vision.vehicle_count),
                "drivable_area_ratio": round(float(vision.drivable_area_ratio), 4),
                "detections": [{
                    "class_name": o.class_name,
                    "risk_class": o.risk_class,
                    "confidence": round(float(o.confidence), 4),
                    "bbox": list(o.bbox),
                    "in_drivable_area": o.in_drivable_area,
                } for o in vision.objects],
            }
        gps_features = {
            "valid": bool(gps.valid), "timestamp": float(gps.timestamp),
            "speed_kmh": round(float(gps.speed_kmh), 3),
            "latitude": round(float(gps.latitude), 7),
            "longitude": round(float(gps.longitude), 7),
            "fix_quality": int(gps.fix_quality),
        }
        fusion_features = None
        if fusion is not None:
            fusion_features = {
                "vision_radar_count": int(fusion.n_vision_radar),
                "vision_only_count": int(fusion.n_vision_only),
                "radar_only_count": int(fusion.n_radar_only),
                "objects": [{
                    "source": o.source, "risk_class": o.risk_class,
                    "bbox": list(o.bbox) if o.bbox is not None else None,
                    "distance_m": round(float(o.distance_m), 3),
                    "ttc_s": round(float(o.ttc_sec), 3),
                    "angle_deg": round(float(o.angle_deg), 3) if o.angle_deg is not None else None,
                    "on_road": o.on_road,
                } for o in fusion.objects],
            }
        row = {
            "sample_id": self._count, "wall_time_ns": wall_ns, "monotonic_ns": mono_ns,
            "relative_ms": (mono_ns - self._started_mono_ns) / 1_000_000.0,
            "frame": {"valid": frame_valid, "path": frame_path,
                      "width": int(frame.shape[1]) if frame is not None else 0,
                      "height": int(frame.shape[0]) if frame is not None else 0},
            "radar": _jsonable(radar), "gps": gps_features,
            "vision": vision_features, "fusion": fusion_features,
            "vision_inference_ms": round(float(inference_ms), 3),
        }
        with self._lock:
            self._file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._count += 1

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()
            self._meta.update({"ended_wall_time_ns": time.time_ns(), "sample_count": self._count})
            self._write_meta()
