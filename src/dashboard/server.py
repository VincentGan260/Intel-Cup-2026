"""FastAPI Web Dashboard 服务。

提供：
  GET  /            → 返回 index.html
  GET  /video_feed  → MJPEG 实时视频流
  GET  /api/state   → 当前系统状态 JSON（从 DashboardStateStore 读取）
  GET  /api/health  → 健康检查
  StaticFiles 挂载 /static

本模块只负责 Web 服务，不负责生成状态数据。
状态数据由 run_dashboard.py 中的后台线程写入 DashboardStateStore。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── 静态文件目录 ──
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── 外部注入实例（由 run_dashboard.py 注入） ──
_camera = None       # type: Optional[object]  # CameraFrameProducer
_state_store = None  # type: Optional[object]  # DashboardStateStore

# ── VisionResult 缓存（状态线程推理后写入，video_annotated_feed 每帧读取绘制） ──
_vision_result_cache: Optional[object] = None  # VisionResult
_vr_lock = threading.Lock()

# ── FastAPI 应用 ──
app = FastAPI(title="Rider Warning Dashboard", version="0.1.0")

# 静态文件挂载（必须在路由前生效）
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ============================================================
#  路由
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 Dashboard 首页。"""
    index_path = _STATIC_DIR / "index.html"
    if index_path.is_file():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/video_feed")
async def video_feed():
    """MJPEG 视频流端点。

    浏览器 <img src="/video_feed"> 自动刷新。
    """

    def _generate():
        while True:
            frame_bytes = b""
            if _camera is not None:
                try:
                    frame_bytes = _camera.get_jpeg_frame()
                except Exception:
                    pass

            if not frame_bytes:
                frame_bytes = b""

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(0.05)  # ~20 FPS

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video_annotated_feed")
async def video_annotated_feed():
    """视觉增强 MJPEG 视频流。

    每帧直接读取摄像头 → 从缓存 VisionResult 实时绘制检测框/分割 mask → JPEG 编码。
    绘制仅需 ~2ms，视频保持摄像头原生帧率；Vision 推理由后台状态线程异步完成。
    """

    def _draw_on_frame(bgr: np.ndarray, vr) -> np.ndarray:
        """在 BGR 帧上绘制检测框 + 语义分割 mask，不重复推理。"""
        try:
            import cv2

            out = bgr.copy()
            h, w = out.shape[:2]

            # ── 语义分割 mask ──
            has_mask = False
            if vr.drivable_mask is not None and vr.drivable_mask.size > 0:
                try:
                    from src.vision.segmentation.mask_utils import resize_mask_to_image
                    mask_resized = resize_mask_to_image(vr.drivable_mask, (w, h))
                    mask_area = (mask_resized > 0).astype(np.float32)
                    color = np.array([0.0, 220.0, 180.0], dtype=np.float32)
                    alpha = 0.35
                    out_f = out.astype(np.float32)
                    out_f = out_f * (1.0 - alpha * mask_area[..., None]) \
                        + color.reshape(1, 1, 3) * (alpha * mask_area[..., None])
                    out = np.clip(out_f, 0, 255).astype(np.uint8)
                    dr_ratio = vr.segmentation.drivable_ratio if (
                        vr.segmentation is not None and vr.segmentation.drivable_ratio is not None
                    ) else float(np.count_nonzero(mask_resized)) / max(1, mask_resized.size)
                    cv2.putText(out, f"Drivable Area: {dr_ratio * 100:.1f}%",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 180), 2, cv2.LINE_AA)
                    has_mask = True
                except Exception:
                    pass

            if not has_mask and vr.segmentation is None:
                cv2.putText(out, "Segmentation: N/A", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)

            # ── 检测框 ──
            for det in vr.detections:
                x1, y1, x2, y2 = (int(det.bbox[0]), int(det.bbox[1]),
                                  int(det.bbox[2]), int(det.bbox[3]))
                risk = det.visual_risk if det.visual_risk is not None else 0.0
                if risk >= 0.7:
                    color = (0, 0, 255)
                elif risk >= 0.3:
                    color = (0, 220, 255)
                else:
                    color = (0, 255, 0)
                cls_display = det.risk_class if det.risk_class else det.class_name
                label = f"{cls_display} {det.confidence:.2f} R:{risk:.2f}"
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, y1), color, -1)
                cv2.putText(out, label, (x1 + 2, max(lh, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return out
        except Exception:
            return bgr

    def _generate():
        while True:
            frame_bytes = b""
            if _camera is not None:
                try:
                    # 读取摄像头原始帧
                    raw_jpeg = _camera.get_jpeg_frame()
                    # 尝试叠加标注
                    with _vr_lock:
                        vr = _vision_result_cache
                    if vr is not None and raw_jpeg:
                        import cv2
                        buf = np.frombuffer(raw_jpeg, np.uint8)
                        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            annotated = _draw_on_frame(bgr, vr)
                            ret, enc = cv2.imencode(".jpg", annotated,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                frame_bytes = enc.tobytes()
                    if not frame_bytes:
                        frame_bytes = raw_jpeg
                except Exception:
                    pass

            if not frame_bytes:
                frame_bytes = b""

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            time.sleep(0.03)  # ~30 FPS

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/state")
async def api_state() -> JSONResponse:
    """返回当前系统状态 JSON。

    数据来源：DashboardStateStore（由后台线程持续更新）。
    """
    if _state_store is not None:
        return JSONResponse(content=_state_store.get_state())
    # 回退：state_store 未注入时返回空状态
    return JSONResponse(
        content={
            "timestamp": time.time(),
            "risk_score": 0.0,
            "risk_level": 0,
            "risk_label": "未连接",
            "risk_items": {"obs": 0.0, "dist": 0.0, "pose": 0.0, "speed": 0.0},
            "weights": {"obs": 0.30, "dist": 0.35, "pose": 0.20, "speed": 0.15},
            "sensors": {"camera": False, "vision": "mock", "radar": "mock", "imu": "mock", "gps": "mock"},
            "mode": "mock",
            "message": "state_store not injected",
        }
    )


@app.get("/api/health")
async def api_health() -> JSONResponse:
    """服务健康检查。"""
    return JSONResponse(
        content={
            "status": "ok",
            "camera_available": _camera is not None and _camera.is_available if _camera else False,
            "state_store_attached": _state_store is not None,
            "version": "0.1.0",
        }
    )


# ============================================================
#  外部注入接口
# ============================================================


def inject_camera(camera) -> None:
    """由 run_dashboard.py 调用，将摄像头实例注入到服务中。"""
    global _camera
    _camera = camera


def inject_state_store(store) -> None:
    """由 run_dashboard.py 调用，将 DashboardStateStore 注入到服务中。"""
    global _state_store
    _state_store = store


def update_vision_result_cache(vr) -> None:
    """由后台状态线程调用，将最新 VisionResult 写入缓存。

    传入 None 表示视觉不可用。
    """
    global _vision_result_cache
    with _vr_lock:
        _vision_result_cache = vr
