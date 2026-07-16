"""Focused regressions for conservative Dashboard resilience helpers."""
from __future__ import annotations

import shutil
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard import server
from src.dashboard.cloud_sync import CloudSyncClient
from src.dashboard.frame_producer import CameraFrameProducer
from src.sensors.radar_reader import RadarReader


def _det(bbox, risk):
    return NS(
        bbox=bbox,
        visual_risk=risk,
        risk_class="obstacle",
        class_name="obstacle",
        confidence=0.9,
    )


def test_display_stability() -> None:
    server._overlay_detection_cache = None
    server._overlay_previous_detections = []
    first = NS(detections=[_det((100, 100, 200, 200), 0.4)])
    second = NS(detections=[_det((110, 100, 210, 200), 0.4)])
    high = NS(detections=[_det((140, 100, 240, 200), 0.8)])

    assert server._get_display_detections(first, 1)[0][1] == (100.0, 100.0, 200.0, 200.0)
    assert server._get_display_detections(second, 2)[0][1] == (108.5, 100.0, 208.5, 200.0)
    assert server._get_display_detections(high, 3)[0][1] == (140.0, 100.0, 240.0, 200.0)
    assert server._get_display_detections(NS(detections=[]), 4) == []
    server._vision_result_cache = first
    server.update_vision_result_cache(None)
    assert server._overlay_previous_detections == []


def test_mask_stability() -> None:
    server._overlay_mask_cache = None
    server._overlay_previous_mask = None
    base = np.zeros((10, 10), dtype=np.uint8)
    base[5:, :] = 1
    changed = base.copy()
    changed[4, 5] = 1

    server._get_overlay_mask_blend(
        NS(drivable_mask=base, segmentation=NS(drivable_ratio=0.5)), 1, 10, 10)
    _keep, color_term, _ratio = server._get_overlay_mask_blend(
        NS(drivable_mask=changed, segmentation=NS(drivable_ratio=0.51)), 2, 10, 10)
    assert np.isclose(color_term[4, 5, 1], 220.0 * 0.35 * 0.85)

    major = np.ones((10, 10), dtype=np.uint8)
    _keep, major_color, _ratio = server._get_overlay_mask_blend(
        NS(drivable_mask=major, segmentation=NS(drivable_ratio=1.0)), 3, 10, 10)
    assert np.isclose(major_color[0, 0, 1], 220.0 * 0.35)


def test_bounded_cloud_queue() -> None:
    spool = ROOT / "tmp" / "test_cloud_resilience"
    shutil.rmtree(spool, ignore_errors=True)
    spool.mkdir(parents=True)
    try:
        for index in range(4):
            (spool / f"bike-001_20260715T00000{index}000000Z.mp4").write_bytes(b"x")
        partial = spool / "bike-001_20260715T000009000000Z.partial.mp4"
        partial.write_bytes(b"still recording")
        client = CloudSyncClient(
            base_url="http://127.0.0.1",
            device_id="bike-001",
            camera=NS(),
            spool_dir=spool,
            video_queue_size=2,
        )
        client._fill_video_queue()
        assert client._video_queue.qsize() == 2
        assert partial.resolve() not in client._queued_video_paths
        client.spool_max_bytes = 2
        assert client._spool_has_capacity() is False
        assert len([
            path for path in spool.glob("*.mp4")
            if not path.name.endswith(".partial.mp4")
        ]) == 4
        try:
            client._probe_duration(spool / "bike-001_20260715T000000000000Z.mp4")
        except ValueError as exc:
            assert "too small" in str(exc)
        else:
            raise AssertionError("truncated MP4 must be rejected")
    finally:
        shutil.rmtree(spool, ignore_errors=True)


def test_cloud_player_keeps_selection() -> None:
    index = (ROOT / "deploy" / "cloud" / "index.html").read_text(encoding="utf-8")
    cloud_data = (ROOT / "deploy" / "cloud" / "cloud-data.html").read_text(encoding="utf-8")
    assert index == cloud_data
    assert "selectedVideoId" in index
    assert "Number(x.file_size_bytes)>=1024" in index
    assert "selectedVideoId=''" not in index
    assert "selectedVideoId=null" in index
    assert "if(i===0&&selectedVideoId===null)" in index
    assert 'id="high-risk-rows"' in index
    assert 'id="all-rows"' in index
    assert "Number(x.risk_level)===2" in index
    assert "x.imu_posture||'--'" in index


def test_serial_retry() -> None:
    attempts = {"count": 0}

    class FakeSerial:
        def __init__(self, *_args, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise OSError("temporarily unavailable")
            self.is_open = True

        def close(self):
            self.is_open = False

    original = sys.modules.get("serial")
    sys.modules["serial"] = types.SimpleNamespace(Serial=FakeSerial)
    try:
        reader = RadarReader(mode="real", config={"reconnect_interval_sec": 0.5})
        reader.start()
        assert reader._serial is None
        assert reader._connect_serial() is False
        assert attempts["count"] == 1
        reader._serial_next_reconnect_at = time.monotonic() - 0.01
        assert reader._connect_serial() is True
        assert reader._serial is not None and reader._serial.is_open
        reader.stop()
    finally:
        if original is None:
            sys.modules.pop("serial", None)
        else:
            sys.modules["serial"] = original


def test_camera_recovery_loop() -> None:
    class FailedCapture:
        def __init__(self):
            self.released = False

        def read(self):
            return False, None

        def release(self):
            self.released = True

    class RecoveredCapture:
        def __init__(self, producer):
            self.producer = producer

        def read(self):
            self.producer._capture_stop.set()
            return True, np.ones((4, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    producer = CameraFrameProducer.__new__(CameraFrameProducer)
    producer._lock = threading.Lock()
    producer._cap = FailedCapture()
    failed_capture = producer._cap
    producer._available = True
    producer._latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    producer._latest_capture_ns = 1
    producer._latest_frame_id = 0
    producer._capture_stop = threading.Event()
    producer._reconnect_interval_sec = 0.01
    producer._reconnect_after_failures = 3

    def recover(*, initial=False):
        producer._cap = RecoveredCapture(producer)
        producer._available = True
        return True

    producer._try_open_capture = recover
    producer._capture_loop()
    assert failed_capture.released is True
    assert producer._available is True
    assert producer._latest_frame_id == 1
    assert np.all(producer._latest_frame == 1)


def main() -> None:
    test_display_stability()
    test_mask_stability()
    test_bounded_cloud_queue()
    test_cloud_player_keeps_selection()
    test_serial_retry()
    test_camera_recovery_loop()
    print("dashboard resilience regressions passed")


if __name__ == "__main__":
    main()
