"""Session writer used by the real-sensor Dashboard.

Each JSONL row is one training sample containing aligned sensor readings,
vision features and vision/radar association output. Raw frames are retained so
features can be regenerated after a model update.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        files = [path] if path.is_file() else []
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest() if files else ""


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


class DashboardRecorder:
    def __init__(self, output: Path, scene: str, profile: str, *,
                 recording_config: dict | None = None, session_fields: dict | None = None,
                 model_path: str = "", vision_config: str = "",
                 vision_runtime: dict | None = None,
                 risk_label: str | None = None) -> None:
        label = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = (output / f"{label}_{scene}_dashboard").resolve()
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self._file = (self.session_dir / "samples.jsonl").open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._started_mono_ns = time.monotonic_ns()
        self._count = 0
        self._risk_label = risk_label
        cfg = recording_config or {}
        project_root = Path(__file__).resolve().parents[2]
        model_abs = project_root / model_path
        vision_abs = project_root / vision_config
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
                capture_output=True, check=False).stdout.strip()
        except Exception:
            git_commit = ""
        self._meta = {"schema_version": int(cfg.get("schema_version", 2)),
                      "scene": scene, "profile": profile,
                      "host": platform.node(), "started_wall_time_ns": time.time_ns(),
                      "started_monotonic_ns": self._started_mono_ns,
                      "alignment_clock": "monotonic_ns",
                      "primary_timeline": "camera",
                      "sync_thresholds_ms": cfg.get("sync", {}),
                      "quality_limits": cfg.get("quality", {}),
                      "model_path": model_path, "vision_config": vision_config,
                      "vision_runtime": vision_runtime or {},
                      "git_commit": git_commit,
                      "model_sha256": _sha256(model_abs),
                      "vision_config_sha256": _sha256(vision_abs),
                      "versions": cfg.get("versions", {}), "labels": cfg.get("labels", {}),
                      "contains": ["raw_frame", "radar", "gps", "vision_features", "vision_radar_fusion"]}
        self._meta.update(session_fields or {})
        (self.session_dir / "events.jsonl").touch()
        self._write_meta()

    def _write_meta(self) -> None:
        (self.session_dir / "session.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def write(self, frame, radar, gps, vision, fusion, inference_ms: float,
              timestamps: dict, *, camera_frame_id: int,
              radar_valid: bool, gps_valid: bool, risk_decision=None) -> None:
        wall_ns = time.time_ns()
        frame_path = ""
        frame_valid = frame is not None
        if frame_valid:
            import cv2

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
            "valid": gps_valid, "sensor_timestamp": float(gps.timestamp),
            "speed_kmh": round(float(gps.speed_kmh), 3),
            "latitude": round(float(gps.latitude), 7),
            "longitude": round(float(gps.longitude), 7),
            "fix_quality": int(gps.fix_quality),
        }
        fusion_features = None
        if fusion is not None:
            fusion_features = {
                "valid": True,
                "vision_radar_count": int(fusion.n_vision_radar),
                "vision_only_count": int(fusion.n_vision_only),
                "radar_only_count": int(fusion.n_radar_only),
                "objects": [{
                    "source": o.source, "risk_class": o.risk_class,
                    "bbox": list(o.bbox) if o.bbox is not None else None,
                    "distance_m": round(float(o.distance_m), 3),
                    "relative_speed_mps": round(float(o.relative_speed_mps), 3),
                    "ttc_s": round(float(o.ttc_sec), 3),
                    "angle_deg": round(float(o.angle_deg), 3) if o.angle_deg is not None else None,
                    "on_road": o.on_road,
                } for o in fusion.objects],
            }
        radar_features = _jsonable(radar)
        if isinstance(radar_features, dict):
            radar_features["valid"] = radar_valid
        record_write_ns = time.monotonic_ns()
        timestamps = dict(timestamps)
        timestamps["record_write_monotonic_ns"] = record_write_ns
        timestamps["end_to_end_latency_ms"] = (
            record_write_ns - timestamps["frame_capture_monotonic_ns"]) / 1_000_000.0
        row = {
            "sample_id": self._count, "wall_time_ns": wall_ns,
            "monotonic_ns": timestamps["frame_capture_monotonic_ns"],
            "relative_ms": (timestamps["frame_capture_monotonic_ns"] - self._started_mono_ns) / 1_000_000.0,
            "frame": {"valid": frame_valid, "path": frame_path,
                      "camera_frame_id": int(camera_frame_id),
                      "width": int(frame.shape[1]) if frame is not None else 0,
                      "height": int(frame.shape[0]) if frame is not None else 0},
            "radar": radar_features, "gps": gps_features,
            "vision": vision_features, "fusion": fusion_features,
            "vision_inference_ms": round(float(inference_ms), 3),
            "vision_module_inference_ms": {
                "detection": round(float(getattr(vision, "detection_inference_ms", 0.0)), 3),
                "segmentation": round(float(getattr(vision, "segmentation_inference_ms", 0.0)), 3),
            },
            "timestamps": timestamps,
            "risk_decision": _jsonable(risk_decision) if risk_decision is not None else None,
            "label": {"risk_level": self._risk_label, "event_id": None, "note": ""},
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
            if self._risk_label is not None and self._count > 0:
                event = {"event_id": "session-000", "start_sample_id": 0,
                         "end_sample_id": self._count - 1, "risk_level": self._risk_label,
                         "event_type": self._meta.get("scene"), "reviewer": "unreviewed"}
                (self.session_dir / "events.jsonl").write_text(
                    json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
