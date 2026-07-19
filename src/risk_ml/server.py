"""Independent FastAPI surface for the standalone XGBoost experiment."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


app = FastAPI(title="Standalone XGBoost Risk Monitor", version="0.1.0")
_state_store = None
_camera = None


def inject_runtime(state_store, camera=None) -> None:
    global _state_store, _camera
    _state_store = state_store
    _camera = camera


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XGBoost 独立风险测试</title>
  <style>
    :root{color-scheme:dark;font-family:Inter,"PingFang SC",sans-serif}
    body{margin:0;background:#08111f;color:#e8eef8}
    main{max-width:1180px;margin:auto;padding:24px}
    h1{font-size:24px;margin:0 0 6px}.sub{color:#94a3b8;margin-bottom:20px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
    .card{background:#101d31;border:1px solid #223451;border-radius:14px;padding:18px}
    .label{color:#94a3b8;font-size:13px}.value{font-size:30px;font-weight:750;margin-top:7px}
    .low{color:#44d17a}.mid{color:#ffbf47}.high{color:#ff5f67}.muted{color:#94a3b8}
    .prob{height:8px;background:#23324a;border-radius:9px;margin-top:7px;overflow:hidden}
    .prob>i{display:block;height:100%;background:#59a5ff}
    img{width:100%;border-radius:12px;background:#000;min-height:240px;object-fit:contain}
    table{width:100%;border-collapse:collapse;font-size:13px}
    td{padding:7px;border-bottom:1px solid #223451}td:last-child{text-align:right}
    .warn{color:#ffbf47}.ok{color:#44d17a}.error{color:#ff5f67}
    @media(max-width:700px){main{padding:14px}.value{font-size:24px}}
  </style>
</head>
<body><main>
  <h1>XGBoost 独立风险测试</h1>
  <div class="sub">不使用旧规则 · 不连接电机 · 独立服务端口</div>
  <div class="grid">
    <section class="card"><div class="label">风险等级</div><div id="level" class="value muted">启动中</div></section>
    <section class="card"><div class="label">综合风险评分</div><div id="score" class="value">--</div></section>
    <section class="card"><div class="label">模型置信度</div><div id="confidence" class="value">--</div></section>
    <section class="card"><div class="label">运行状态</div><div id="status" class="value muted">等待数据</div></section>
  </div>
  <div class="grid" style="margin-top:14px">
    <section class="card">
      <div class="label">实时画面</div>
      <img src="/video_feed" alt="camera">
    </section>
    <section class="card">
      <div class="label">三分类概率</div>
      <div id="probabilities"></div>
      <div class="label" style="margin-top:18px">传感器状态</div>
      <table id="sensors"></table>
    </section>
  </div>
  <section class="card" style="margin-top:14px">
    <div class="label">31 项模型输入</div>
    <table id="features"></table>
  </section>
</main>
<script>
const fmt=(v,n=3)=>v===null||v===undefined?"缺失":Number(v).toFixed(n);
async function refresh(){
  try{
    const s=await (await fetch("/api/state",{cache:"no-store"})).json();
    const p=s.prediction||{}, level=document.getElementById("level");
    level.textContent=p.label||s.status||"等待";
    level.className="value "+(p.level===2?"high":p.level===1?"mid":p.level===0?"low":"muted");
    document.getElementById("score").textContent=p.risk_score_100==null?"--":fmt(p.risk_score_100,1);
    document.getElementById("confidence").textContent=p.confidence==null?"--":fmt(p.confidence*100,1)+"%";
    const status=document.getElementById("status");
    status.textContent=s.status||"unknown";
    status.className="value "+(s.status==="active"?"ok":s.status==="warming_up"?"warn":"error");
    const probs=p.probabilities||{};
    document.getElementById("probabilities").innerHTML=Object.entries(probs).map(([k,v])=>
      `<div style="margin-top:14px">${k}<span style="float:right">${fmt(v*100,1)}%</span><div class="prob"><i style="width:${v*100}%"></i></div></div>`
    ).join("");
    document.getElementById("sensors").innerHTML=Object.entries(s.sensors||{}).map(([k,v])=>
      `<tr><td>${k}</td><td>${v.status||v}</td></tr>`).join("");
    document.getElementById("features").innerHTML=Object.entries(s.features||{}).map(([k,v])=>
      `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("");
  }catch(e){document.getElementById("status").textContent="连接失败"}
}
refresh();setInterval(refresh,500);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/state")
def api_state() -> JSONResponse:
    if _state_store is None:
        return JSONResponse({"status": "not_initialized"}, status_code=503)
    return JSONResponse(_state_store.get_state())


@app.get("/api/health")
def api_health() -> JSONResponse:
    if _state_store is None:
        return JSONResponse({"status": "not_initialized"}, status_code=503)
    _, version, age_ms = _state_store.get_snapshot()
    healthy = age_ms <= 2000.0
    return JSONResponse({
        "status": "ok" if healthy else "stale",
        "service": "rider-xgb",
        "decision_engine": "xgboost-only",
        "motor_control": False,
        "state_version": version,
        "state_age_ms": round(age_ms, 1),
        "camera_available": bool(_camera and _camera.is_available),
    }, status_code=200 if healthy else 503)


def _frames():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame = _camera.get_jpeg_frame() if _camera is not None else b""
        yield boundary + frame + b"\r\n"
        time.sleep(0.08)


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
