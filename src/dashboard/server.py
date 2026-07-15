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

import asyncio
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

# ── 共享帧抓取器（单摄像头 → 多消费者，防止帧被瓜分） ──
_shared_jpeg: Optional[bytes] = None
_shared_bgr: Optional[np.ndarray] = None
_shared_frame_id = -1
_shared_capture_ns = 0
_shared_lock = threading.Lock()
_grabber_alive = False
_grabber_thread: Optional[threading.Thread] = None


def _ensure_grabber_started() -> None:
    """惰性启动共享帧抓取线程。"""
    global _grabber_thread, _grabber_alive
    if _grabber_alive:
        return
    _grabber_alive = True
    _grabber_thread = threading.Thread(target=_grab_loop, daemon=True, name="frame-grabber")
    _grabber_thread.start()


def _grab_loop() -> None:
    """后台持续抓帧，缓存最新 JPEG 和 BGR，供所有视频端点消费。"""
    global _shared_jpeg, _shared_bgr, _shared_frame_id, _shared_capture_ns
    import cv2
    while _grabber_alive:
        jpeg_bytes = b""
        bgr = None
        capture_ns = 0
        frame_id = -1
        if _camera is not None:
            try:
                bgr, capture_ns, frame_id = _camera.get_bgr_frame_with_timestamp()
                if bgr is not None and frame_id != _shared_frame_id:
                    ret, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        jpeg_bytes = enc.tobytes()
                elif bgr is None:
                    jpeg_bytes = _camera.get_jpeg_frame()
                    if jpeg_bytes:
                        buf = np.frombuffer(jpeg_bytes, np.uint8)
                        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                pass
        with _shared_lock:
            if jpeg_bytes:
                _shared_jpeg = jpeg_bytes
            if bgr is not None:
                _shared_bgr = bgr
                _shared_frame_id = frame_id
                _shared_capture_ns = capture_ns
        time.sleep(0.05)  # ~20 FPS，兼顾演示流畅度与端侧负载


def _get_shared_jpeg() -> bytes:
    with _shared_lock:
        return _shared_jpeg or b""


def _get_shared_bgr() -> Optional[np.ndarray]:
    with _shared_lock:
        return _shared_bgr


def _get_video_frame_age_ms() -> Optional[float]:
    with _shared_lock:
        if _shared_capture_ns <= 0:
            return None
        return max(0.0, (time.monotonic_ns() - _shared_capture_ns) / 1_000_000.0)


def _get_state_payload() -> dict:
    if _state_store is None:
        return {}
    state, version, age_ms = _state_store.get_snapshot()
    state["_dashboard"] = {
        "version": version,
        "state_age_ms": round(age_ms, 1),
    }
    return state

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
    帧来自共享抓取器，与 video_annotated_feed 共用同一路摄像头数据。
    """

    def _generate():
        _ensure_grabber_started()
        try:
            while True:
                frame_bytes = _get_shared_jpeg()
                if not frame_bytes:
                    time.sleep(0.05)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
                time.sleep(0.05)  # ~20 FPS
        except GeneratorExit:
            pass  # 客户端断开，正常退出

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video_annotated_feed")
async def video_annotated_feed():
    """视觉增强 MJPEG 视频流。

    每帧直接读取摄像头 → 从缓存 VisionResult 实时绘制检测框/分割 mask → JPEG 编码。
    视频以约 20 FPS 输出；Vision 推理由后台状态线程异步完成。
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
        _ensure_grabber_started()
        try:
            while True:
                frame_bytes = b""
                with _vr_lock:
                    vr = _vision_result_cache
                if vr is not None:
                    bgr = _get_shared_bgr()
                    if bgr is not None:
                        try:
                            import cv2
                            annotated = _draw_on_frame(bgr, vr)
                            ret, enc = cv2.imencode(".jpg", annotated,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                frame_bytes = enc.tobytes()
                        except Exception:
                            pass
                if not frame_bytes:
                    frame_bytes = _get_shared_jpeg()

                if not frame_bytes:
                    time.sleep(0.03)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
                time.sleep(0.05)  # ~20 FPS，避免视频编码挤占状态与推理线程
        except GeneratorExit:
            pass  # 客户端断开，正常退出

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket) -> None:
    """WebSocket 端点，后台线程状态变化时主动推送到前端。

    替代前端 HTTP 轮询 /api/state，减少冗余请求。
    连接建立后立即推送当前状态，之后每检测到版本号变化即推送。
    """
    await websocket.accept()
    last_version = -1
    try:
        while True:
            # 检查版本号是否变化
            new_version = -1
            current_state = None
            if _state_store is not None:
                new_version = _state_store.get_version()
                if new_version != last_version:
                    current_state = _get_state_payload()

            if current_state is not None:
                await websocket.send_json(current_state)
                last_version = new_version

            # 等 200ms 或收到客户端消息（支持 ping/pong）
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.2)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass


@app.get("/api/state")
def api_state() -> JSONResponse:
    """返回当前系统状态 JSON。

    数据来源：DashboardStateStore（由后台线程持续更新）。
    """
    if _state_store is not None:
        return JSONResponse(content=_get_state_payload())
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
    state_age_ms = None
    state_version = None
    if _state_store is not None:
        _, state_version, state_age_ms = _state_store.get_snapshot()
    return JSONResponse(
        content={
            "status": "ok",
            "camera_available": _camera is not None and _camera.is_available if _camera else False,
            "state_store_attached": _state_store is not None,
            "state_version": state_version,
            "state_age_ms": round(state_age_ms, 1) if state_age_ms is not None else None,
            "video_frame_age_ms": (
                round(frame_age, 1) if (frame_age := _get_video_frame_age_ms()) is not None else None
            ),
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
