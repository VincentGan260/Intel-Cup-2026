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


def describe_imu_posture(state: dict, deadband_deg: float = 5.0) -> Optional[str]:
    """Convert calibrated body roll/pitch into a stable display description."""
    imu = state.get("imu_data", {})
    if not bool(imu.get("valid", False)):
        return None
    roll = _finite_optional(imu.get("roll"))
    pitch = _finite_optional(imu.get("pitch"))
    if roll is None or pitch is None:
        return None
    posture = []
    if roll <= -deadband_deg:
        posture.append("向左倾斜")
    elif roll >= deadband_deg:
        posture.append("向右倾斜")
    if pitch >= deadband_deg:
        posture.append("向前倾斜")
    elif pitch <= -deadband_deg:
        posture.append("向后仰")
    return "并".join(posture) if posture else "姿态平稳"


def build_ride_payload(state: dict, device_id: str) -> dict:
    """Map the Dashboard state contract to the compact cloud schema."""
    gps = state.get("gps_data", {})
    radar = state.get("radar_data", {})
    vision = state.get("vision_details", {})
    gps_valid = bool(gps.get("valid", False))
    radar_valid = bool(radar.get("valid", False))
    vision_valid = bool(vision.get("valid", False))
    timestamp = float(state.get("timestamp", time.time()))
    risk_score = _finite_optional(state.get("risk_score"))
    risk_score = min(1.0, max(0.0, risk_score)) if risk_score is not None else None
    risk_level = state.get("risk_level")
    risk_level = int(risk_level) if risk_level in (0, 1, 2) else None
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
        "imu_posture": describe_imu_posture(state),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "system_status": str(state.get("system_status", "unknown")),
        "warning_reason": str(state.get("warning_reason", "")),
        "radar_level": state.get("radar_level"),
        "vision_level": state.get("vision_level"),
    }


class CloudSyncClient:
    """Non-blocking cloud uploader with a persistent local video spool."""

    def __init__(self, *, base_url: str, device_id: str, camera, spool_dir: Path,
                 state_hz: float = 1.0, video_fps: float = 10.0,
                 segment_seconds: float = 60.0, request_timeout: float = 15.0,
                 video_queue_size: int = 8, spool_max_gb: float = 2.0) -> None:
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
        self.spool_max_bytes = max(64 * 1024 * 1024, int(float(spool_max_gb) * 1024 ** 3))
        self._stop = threading.Event()
        self._state_queue: queue.Queue = queue.Queue(maxsize=2)
        self._video_queue: queue.Queue = queue.Queue(maxsize=max(1, int(video_queue_size)))
        self._video_queue_lock = threading.Lock()
        self._queued_video_paths: set[Path] = set()
        self._video_retry_after: dict[Path, float] = {}
        self._video_failures: dict[Path, int] = {}
        self._last_state_queued = 0.0
        self._state_uploaded = 0
        self._record_done = threading.Event()
        self._threads = []
        self._last_spool_warning = 0.0

    def start(self) -> None:
        for path in self.spool_dir.glob("*.partial.mp4"):
            path.unlink(missing_ok=True)
        self._fill_video_queue()
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

    def _spool_usage_bytes(self) -> int:
        total = 0
        for pattern in ("*.mp4", "*.partial.mp4"):
            for path in self.spool_dir.glob(pattern):
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    def _spool_has_capacity(self) -> bool:
        usage = self._spool_usage_bytes()
        if usage < self.spool_max_bytes:
            return True
        now_mono = time.monotonic()
        if now_mono - self._last_spool_warning >= 30.0:
            print(f"[CloudSync] video spool full ({usage / 1024 ** 3:.2f} GiB); "
                  "pausing new cloud video recording until uploads free space")
            self._last_spool_warning = now_mono
        return False

    def _enqueue_video(self, path: Path, started_at: datetime) -> bool:
        path = path.resolve()
        if path.name.endswith(".partial.mp4"):
            return False
        with self._video_queue_lock:
            if path in self._queued_video_paths:
                return True
            if time.monotonic() < self._video_retry_after.get(path, 0.0):
                return False
            try:
                self._video_queue.put_nowait((path, started_at))
            except queue.Full:
                return False
            self._queued_video_paths.add(path)
            return True

    def _fill_video_queue(self) -> None:
        for path in sorted(self.spool_dir.glob("*.mp4")):
            if path.name.endswith(".partial.mp4"):
                continue
            if not self._enqueue_video(path, self._started_at_from_path(path)):
                if self._video_queue.full():
                    break

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
        # Browser playback requires AVC/H.264 inside MP4. MPEG-4 Part 2
        # (mp4v) is deliberately not used as a fallback because Chrome/Edge
        # cannot reliably decode it even though the container is named .mp4.
        for codec in ("avc1", "H264"):
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
                    if not self._spool_has_capacity():
                        self._stop.wait(1.0)
                        continue
                    started_at = datetime.now(timezone.utc)
                    name = f"{self.device_id}_{started_at.strftime('%Y%m%dT%H%M%S%fZ')}.mp4"
                    current_path = self.spool_dir / name
                    partial_path = current_path.with_name(current_path.stem + ".partial.mp4")
                    writer, codec = self._open_writer(frame, partial_path)
                    segment_start = time.monotonic()
                    if writer is None:
                        print("[CloudSync] no browser-compatible H.264 MP4 codec")
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
                        self._enqueue_video(current_path, started_at)
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
                self._enqueue_video(current_path, started_at)
        self._record_done.set()

    def _video_upload_loop(self) -> None:
        import requests
        url = f"{self.base_url}/api/video-segments"
        while not self._stop.is_set() or not self._record_done.is_set():
            self._fill_video_queue()
            try:
                path, started_at = self._video_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not path.is_file():
                with self._video_queue_lock:
                    self._queued_video_paths.discard(path)
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
                with self._video_queue_lock:
                    self._queued_video_paths.discard(path)
                    self._video_retry_after.pop(path, None)
                    self._video_failures.pop(path, None)
                print(f"[CloudSync] video uploaded {path.name} duration={duration_s:.1f}s")
            except ValueError as exc:
                path.unlink(missing_ok=True)
                with self._video_queue_lock:
                    self._queued_video_paths.discard(path)
                    self._video_retry_after.pop(path, None)
                    self._video_failures.pop(path, None)
                print(f"[CloudSync] discarded invalid video {path.name}: {exc}")
            except Exception as exc:
                with self._video_queue_lock:
                    failures = self._video_failures.get(path, 0) + 1
                    self._video_failures[path] = failures
                    self._video_retry_after[path] = (
                        time.monotonic() + min(60.0, 5.0 * (2 ** min(failures - 1, 4))))
                    self._queued_video_paths.discard(path)
                print(f"[CloudSync] video upload failed {path.name}; retained for retry: {exc}")

    def _probe_duration(self, path: Path) -> float:
        """Return playable MP4 duration, rejecting truncated/unreadable files."""
        size_bytes = path.stat().st_size
        if size_bytes < 1024:
            raise ValueError(f"file is too small ({size_bytes} bytes)")
        try:
            import cv2
            capture = cv2.VideoCapture(str(path))
            opened = capture.isOpened()
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            readable, _ = capture.read() if opened else (False, None)
            capture.release()
            if opened and readable and math.isfinite(fps) and math.isfinite(frames) \
                    and fps > 0.0 and frames > 0.0:
                return max(0.1, frames / fps)
        except (ImportError, OSError) as exc:
            raise ValueError(f"cannot inspect MP4: {exc}") from exc
        raise ValueError("MP4 has no readable video frames")

    def _started_at_from_path(self, path: Path) -> datetime:
        try:
            stamp = path.stem.rsplit("_", 1)[1]
            return datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
