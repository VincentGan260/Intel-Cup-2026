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

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── 静态文件目录 ──
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── 外部注入实例（由 run_dashboard.py 注入） ──
_camera = None       # type: Optional[object]  # CameraFrameProducer
_state_store = None  # type: Optional[object]  # DashboardStateStore

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
