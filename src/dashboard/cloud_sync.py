"""Upload live ride samples and minute-long raw video segments to the cloud."""

from __future__ import annotations

import queue
import re
import threading
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _finite_optional(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_optional(value):
    number = _finite_optional(value)
    return number if number is not None and number >= 0.0 else None


def build_ride_payload(state: dict, device_id: str) -> dict:
    """Map the Dashboard state contract to the compact cloud schema."""
    gps = state.get("gps_data", {})
    radar = state.get("radar_data", {})
    vision = state.get("vision_details", {})
    gps_valid = bool(gps.get("valid", False))
    radar_valid = bool(radar.get("valid", False))
    vision_valid = bool(vision.get("valid", False))
    timestamp = float(state.get("timestamp", time.time()))
    return {
        "device_id": device_id,
        "collected_at": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
        "gps_valid": gps_valid,
        "latitude": _finite_optional(gps.get("latitude")) if gps_valid else None,
        "longitude": _finite_optional(gps.get("longitude")) if gps_valid else None,
        "speed_kmh": max(0.0, float(gps.get("speed_kmh", 0.0) or 0.0)),
        "radar_valid": radar_valid,
        "target_count": max(0, int(radar.get("target_count", 0) or 0)),
        # Radar uses -1 internally for "no target/no approaching target";
        # the cloud schema represents that state as NULL.
        "nearest_distance_m": _nonnegative_optional(radar.get("nearest_distance_m")) if radar_valid else None,
        "min_ttc_s": _nonnegative_optional(radar.get("min_ttc_s")) if radar_valid else None,
        "vision_valid": vision_valid,
        "obstacle_count": max(0, int(vision.get("object_count", 0) or 0)),
        "drivable_area_ratio": _finite_optional(vision.get("drivable_area_ratio")) if vision_valid else None,
        "risk_score": min(1.0, max(0.0, float(state.get("risk_score", 0.0) or 0.0))),
        "risk_level": min(2, max(0, int(state.get("risk_level", 0) or 0))),
    }


class CloudSyncClient:
    """Non-blocking cloud uploader with a persistent local video spool."""

    def __init__(self, *, base_url: str, device_id: str, camera, spool_dir: Path,
                 state_hz: float = 1.0, video_fps: float = 10.0,
                 segment_seconds: float = 60.0, request_timeout: float = 15.0) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", device_id):
            raise ValueError("device_id must match [A-Za-z0-9_-]{1,32}")
        self.base_url = base_url.rstrip("/")
        self.device_id = device_id
        self.camera = camera
        self.spool_dir = Path(spool_dir).resolve()
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.state_interval = 1.0 / max(0.1, float(state_hz))
        self.video_fps = max(1.0, float(video_fps))
        self.segment_seconds = max(1.0, float(segment_seconds))
        self.request_timeout = max(1.0, float(request_timeout))
        self._stop = threading.Event()
        self._state_queue: queue.Queue = queue.Queue(maxsize=2)
        self._video_queue: queue.Queue = queue.Queue()
        self._last_state_queued = 0.0
        self._state_uploaded = 0
        self._record_done = threading.Event()
        self._threads = []

    def start(self) -> None:
        for path in self.spool_dir.glob("*.partial.mp4"):
            path.unlink(missing_ok=True)
        for path in sorted(self.spool_dir.glob("*.mp4")):
            self._video_queue.put((path, self._started_at_from_path(path)))
        self._threads = [
            threading.Thread(target=self._state_upload_loop, daemon=True,
                             name="cloud-state-upload"),
            threading.Thread(target=self._video_record_loop, daemon=True,
                             name="cloud-video-record"),
            threading.Thread(target=self._video_upload_loop, daemon=True,
                             name="cloud-video-upload"),
        ]
        for thread in self._threads:
            thread.start()
        print(f"[CloudSync] started url={self.base_url} device={self.device_id}")

    def publish_state(self, state: dict) -> None:
        now = time.monotonic()
        if now - self._last_state_queued < self.state_interval:
            return
        self._last_state_queued = now
        payload = build_ride_payload(state, self.device_id)
        try:
            self._state_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._state_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._state_queue.put_nowait(payload)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        print("[CloudSync] stopped")

    def _state_upload_loop(self) -> None:
        import requests
        url = f"{self.base_url}/api/ride-samples"
        while not self._stop.is_set():
            try:
                payload = self._state_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                response = requests.post(url, json=payload, timeout=self.request_timeout)
                if not response.ok:
                    print(f"[CloudSync] state rejected status={response.status_code} "
                          f"body={response.text} payload={payload}")
                response.raise_for_status()
                self._state_uploaded += 1
                if self._state_uploaded == 1 or self._state_uploaded % 10 == 0:
                    print(f"[CloudSync] state uploaded count={self._state_uploaded} "
                          f"id={response.json().get('id', '?')}")
            except Exception as exc:
                print(f"[CloudSync] state upload failed: {exc}")

    def _open_writer(self, frame, path: Path):
        import cv2
        height, width = frame.shape[:2]
        # MPEG-4 Part 2 is substantially lighter than software H.264 on DK2500.
        # Prefer it so the recorder can sustain the requested wall-clock FPS.
        for codec in ("mp4v", "avc1", "H264"):
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*codec), self.video_fps, (width, height))
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None, ""

    def _video_record_loop(self) -> None:
        writer = None
        current_path: Optional[Path] = None
        partial_path: Optional[Path] = None
        started_at: Optional[datetime] = None
        segment_start = 0.0
        frame_interval = 1.0 / self.video_fps
        while not self._stop.is_set():
            loop_start = time.monotonic()
            frame = self.camera.get_bgr_frame()
            if frame is not None:
                if writer is None:
                    started_at = datetime.now(timezone.utc)
                    name = f"{self.device_id}_{started_at.strftime('%Y%m%dT%H%M%S%fZ')}.mp4"
                    current_path = self.spool_dir / name
                    partial_path = current_path.with_name(current_path.stem + ".partial.mp4")
                    writer, codec = self._open_writer(frame, partial_path)
                    segment_start = time.monotonic()
                    if writer is None:
                        print("[CloudSync] no usable MP4 codec")
                        self._stop.wait(2.0)
                        continue
                    print(f"[CloudSync] recording {name} codec={codec}")
                writer.write(frame)
                if time.monotonic() - segment_start >= self.segment_seconds:
                    writer.release()
                    writer = None
                    if (current_path is not None and partial_path is not None
                            and started_at is not None):
                        partial_path.replace(current_path)
                        self._video_queue.put((current_path, started_at))
                    current_path = None
                    partial_path = None
                    started_at = None
            remaining = frame_interval - (time.monotonic() - loop_start)
            if remaining > 0:
                self._stop.wait(remaining)
        if writer is not None:
            writer.release()
            if (current_path is not None and partial_path is not None
                    and started_at is not None and partial_path.exists()):
                partial_path.replace(current_path)
                self._video_queue.put((current_path, started_at))
        self._record_done.set()

    def _video_upload_loop(self) -> None:
        import requests
        url = f"{self.base_url}/api/video-segments"
        while not self._record_done.is_set() or not self._video_queue.empty():
            try:
                path, started_at = self._video_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not path.is_file():
                continue
            try:
                duration_s = self._probe_duration(path)
                with path.open("rb") as source:
                    response = requests.post(
                        url,
                        data={"device_id": self.device_id,
                              "started_at": started_at.isoformat(),
                              "duration_s": str(duration_s)},
                        files={"file": (path.name, source, "video/mp4")},
                        timeout=max(60.0, self.request_timeout),
                    )
                response.raise_for_status()
                path.unlink()
                print(f"[CloudSync] video uploaded {path.name} duration={duration_s:.1f}s")
            except Exception as exc:
                print(f"[CloudSync] video upload failed {path.name}: {exc}")
                if not self._stop.wait(10.0):
                    self._video_queue.put((path, started_at))

    def _probe_duration(self, path: Path) -> float:
        """Return the playable MP4 duration instead of assuming segment length."""
        try:
            import cv2
            capture = cv2.VideoCapture(str(path))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            capture.release()
            if fps > 0.0 and frames > 0.0:
                return max(0.1, frames / fps)
        except Exception:
            pass
        return self.segment_seconds

    def _started_at_from_path(self, path: Path) -> datetime:
        try:
            stamp = path.stem.rsplit("_", 1)[1]
            return datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
