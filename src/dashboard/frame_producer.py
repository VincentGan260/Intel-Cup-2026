"""摄像头帧读取器。

零依赖（除 opencv-python / numpy），提供线程安全的摄像头帧读取与 JPEG 编码。
摄像头不可用时自动生成 Camera Not Available 提示图，不抛异常、不崩溃。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional


class CameraFrameProducer:
    """线程安全的摄像头帧读取器。

    Args:
        camera_id: 摄像头设备编号，默认 0
        width: 输出帧宽度
        height: 输出帧高度
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        reconnect_interval_sec: float = 2.0,
        reconnect_after_failures: int = 15,
    ) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._cap = None  # type: Optional["cv2.VideoCapture"]  # noqa
        self._lock = threading.Lock()
        self._fallback_jpeg: Optional[bytes] = None
        self._available = False
        self._latest_frame = None
        self._latest_capture_ns = 0
        self._latest_frame_id = -1
        self._capture_stop = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._reconnect_interval_sec = max(0.5, float(reconnect_interval_sec))
        self._reconnect_after_failures = max(3, int(reconnect_after_failures))

        self._build_fallback_image()
        self._try_open_capture(initial=True)
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="dashboard-camera-capture")
        self._capture_thread.start()

    # ---- 内部 ----

    def _build_fallback_image(self) -> None:
        """Build the fallback JPEG once; reconnect attempts do not re-encode it."""
        try:
            import numpy as np

            self._build_fallback(np)
        except ImportError:
            self._fallback_jpeg = b""

    def _try_open_capture(self, *, initial: bool = False) -> bool:
        """Open and validate a capture without spawning another worker thread."""
        try:
            import cv2

            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                cap.release()
                if initial:
                    print(f"[CameraFrameProducer] 摄像头 {self.camera_id} 未能打开，将自动重试")
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # 验证可读取
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                if initial:
                    print(f"[CameraFrameProducer] 摄像头 {self.camera_id} 暂时无法读取，将自动重试")
                return False

            with self._lock:
                self._cap = cap
                self._latest_frame = frame.copy()
                self._latest_capture_ns = time.monotonic_ns()
                self._latest_frame_id += 1
                self._available = True
            action = "已就绪" if initial else "已自动恢复"
            print(f"[CameraFrameProducer] 摄像头 {self.camera_id} {action} ({self.width}x{self.height})")
            return True

        except ImportError:
            if initial:
                print("[CameraFrameProducer] opencv-python 未安装，无法打开摄像头")
            return False
        except Exception as exc:
            if initial:
                print(f"[CameraFrameProducer] 摄像头打开异常，将自动重试: {exc}")
            return False

    def _disconnect_capture(self) -> None:
        """Mark the stream unavailable and release the failed handle."""
        with self._lock:
            cap = self._cap
            self._cap = None
            self._available = False
            self._latest_frame = None
            self._latest_capture_ns = 0
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _build_fallback(self, np_module) -> None:
        """生成黑底白字回退图。"""
        import cv2

        img = np_module.zeros((self.height, self.width, 3), dtype=np_module.uint8)
        cv2.putText(
            img,
            "Camera Not Available",
            (self.width // 2 - 180, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ret, buf = cv2.imencode(".jpg", img)
        if ret:
            self._fallback_jpeg = buf.tobytes()
        self._available = False

    # ---- 公共接口 ----

    def _capture_loop(self) -> None:
        """唯一的底层摄像头消费者；页面、推理和录制只读取最新缓存。"""
        consecutive_failures = 0
        while not self._capture_stop.is_set():
            cap = self._cap
            if cap is None:
                if self._capture_stop.wait(self._reconnect_interval_sec):
                    break
                self._try_open_capture()
                consecutive_failures = 0
                continue
            try:
                ret, frame = cap.read()
                capture_ns = time.monotonic_ns()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= self._reconnect_after_failures:
                        print("[CameraFrameProducer] 摄像头连续读帧失败，开始自动恢复")
                        self._disconnect_capture()
                        consecutive_failures = 0
                    self._capture_stop.wait(0.02)
                    continue
                consecutive_failures = 0
                with self._lock:
                    self._latest_frame = frame.copy()
                    self._latest_capture_ns = capture_ns
                    self._latest_frame_id += 1
                    self._available = True
            except Exception as exc:
                print(f"[CameraFrameProducer] 摄像头读取异常，开始自动恢复: {exc}")
                self._disconnect_capture()
                consecutive_failures = 0
                self._capture_stop.wait(0.02)

    def get_jpeg_frame(self) -> bytes:
        """读取一帧并编码为 JPEG bytes（线程安全）。

        摄像头不可用时返回回退提示图。
        """
        with self._lock:
            if not self._available or self._latest_frame is None:
                return self._fallback_jpeg or b""

            try:
                import cv2

                ret, buf = cv2.imencode(".jpg", self._latest_frame)
                if not ret:
                    return self._fallback_jpeg or b""

                return buf.tobytes()
            except Exception:
                return self._fallback_jpeg or b""

    def get_bgr_frame(self):
        """读取原始 BGR 帧（线程安全，返回副本）。

        供 VisionAdapter.process(frame) 使用，不重复打开摄像头。

        Returns:
            numpy.ndarray (H, W, 3) 或 None（摄像头不可用时）
        """
        with self._lock:
            if not self._available or self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_bgr_frame_with_timestamp(self):
        """读取一帧，并返回成功采集后立即取得的单调时钟时间戳。"""
        with self._lock:
            if not self._available or self._latest_frame is None:
                return None, 0, -1
            return self._latest_frame.copy(), self._latest_capture_ns, self._latest_frame_id

    def release(self) -> None:
        """释放摄像头资源。"""
        self._capture_stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        self._disconnect_capture()
        print("[CameraFrameProducer] 摄像头已释放")

    @property
    def is_available(self) -> bool:
        """摄像头是否可用。"""
        return self._available
