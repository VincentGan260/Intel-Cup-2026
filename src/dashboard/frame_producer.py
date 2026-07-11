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

        self._open()

    # ---- 内部 ----

    def _open(self) -> None:
        """尝试打开摄像头，失败则生成回退提示图。"""
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                print(f"[CameraFrameProducer] 摄像头 {self.camera_id} 未能打开")
                self._build_fallback(np)
                return

            # 验证可读取
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[CameraFrameProducer] 摄像头 {self.camera_id} 打开成功但无法读取帧")
                cap.release()
                self._build_fallback(np)
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self._cap = cap
            self._latest_frame = frame.copy()
            self._latest_capture_ns = time.monotonic_ns()
            self._latest_frame_id = 0
            self._available = True
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="dashboard-camera-capture")
            self._capture_thread.start()
            print(f"[CameraFrameProducer] 摄像头 {self.camera_id} 已就绪 ({self.width}x{self.height})")

        except ImportError:
            print("[CameraFrameProducer] opencv-python 未安装，无法打开摄像头")
            self._available = False

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
        while not self._capture_stop.is_set():
            cap = self._cap
            if cap is None:
                break
            try:
                ret, frame = cap.read()
                capture_ns = time.monotonic_ns()
                if not ret or frame is None:
                    self._capture_stop.wait(0.02)
                    continue
                with self._lock:
                    self._latest_frame = frame.copy()
                    self._latest_capture_ns = capture_ns
                    self._latest_frame_id += 1
            except Exception:
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
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._available = False
            self._latest_frame = None
        print("[CameraFrameProducer] 摄像头已释放")

    @property
    def is_available(self) -> bool:
        """摄像头是否可用。"""
        return self._available
