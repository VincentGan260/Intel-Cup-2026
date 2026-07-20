"""Independent FastAPI surface for the standalone XGBoost experiment."""

from __future__ import annotations

import threading
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


app = FastAPI(title="Standalone XGBoost Risk Monitor", version="0.1.0")
_state_store = None
_camera = None
_vision_adapter = None
_overlay_lock = threading.Lock()
_overlay_cache_key = None
_overlay_cache_jpeg = b""


def inject_runtime(state_store, camera=None, vision_adapter=None) -> None:
    global _state_store, _camera, _vision_adapter
    _state_store = state_store
    _camera = camera
    _vision_adapter = vision_adapter


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XGBoost 骑行风险</title>
  <style>
    :root{color-scheme:dark;font-family:Inter,"PingFang SC",sans-serif;
      --bg:#07111d;--panel:#0e1c2c;--line:#20344c;--text:#edf5ff;
      --muted:#8da2ba;--blue:#5aa7ff;--green:#42d383;--yellow:#ffc857;--red:#ff6470}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#132d49,var(--bg) 38%);color:var(--text)}
    main{max-width:1280px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 5px}
    .sub,.hint{color:var(--muted);font-size:13px}.hint{margin-top:7px}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:18px}
    .two{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(320px,.75fr);gap:13px;margin-top:13px}
    .modules{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:13px}
    .card{background:linear-gradient(145deg,#11243a,var(--panel));border:1px solid var(--line);border-radius:15px;padding:17px;box-shadow:0 15px 35px #0004}
    .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
    .value{font-size:30px;font-weight:780;margin-top:7px}.low,.raises{color:var(--green)}
    .mid{color:var(--yellow)}.high,.error{color:var(--red)}.muted{color:var(--muted)}.ok{color:var(--green)}
    .prob,.meter{height:7px;background:#1c3047;border-radius:9px;margin-top:7px;overflow:hidden}
    .prob>i,.meter>i{display:block;height:100%;background:var(--blue);transition:width .35s}
    img{width:100%;border-radius:11px;background:#000;min-height:300px;object-fit:contain;margin-top:10px}
    table{width:100%;border-collapse:collapse;font-size:12px}td{padding:6px;border-bottom:1px solid #1d3045}
    td:last-child{text-align:right}.module-head{display:flex;align-items:center;justify-content:space-between}
    .module-name{font-size:18px;font-weight:700}.module-pct{font-size:24px;font-weight:750}
    .direction{font-size:12px;margin-top:9px}.lowers{color:var(--blue)}.neutral{color:var(--muted)}
    .feature{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:11px;margin-top:6px}
    .pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 8px;font-size:11px}
    .motor-line{font-size:13px;margin-top:7px}.warning{margin-top:13px;padding:11px 14px;border-radius:10px;background:#4a351933;color:#ffd890;font-size:12px}
    details{margin-top:13px}summary{cursor:pointer;color:var(--muted)}
    @media(max-width:900px){.grid,.modules{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}}
    @media(max-width:560px){main{padding:13px}.grid,.modules{grid-template-columns:1fr}.value{font-size:25px}img{min-height:220px}}
  </style>
</head>
<body><main>
  <h1>XGBoost 骑行风险</h1>
  <div class="sub">XGBoost 主模型 · 确定性传感器降级 · 类别无关输入 · DRV2605 安全门禁</div>
  <div class="grid">
    <section class="card"><div class="label">总体风险</div><div id="level" class="value muted">启动中</div><div id="score" class="hint">--</div></section>
    <section class="card"><div class="label">模型置信度</div><div id="confidence" class="value">--</div><div id="probHint" class="hint">等待概率</div></section>
    <section class="card"><div class="label">运行状态</div><div id="status" class="value muted">等待数据</div><div id="stateAge" class="hint">schema xgb-risk-v2</div></section>
    <section class="card"><div class="label">真实电机</div><div id="motor" class="value muted">检查中</div><div id="motorDetail" class="motor-line">--</div></section>
  </div>

  <div class="two">
    <section class="card">
      <div class="label">YOLO26n v2 检测框 + Road-ADAS 语义分割</div>
      <img src="/video_feed?annotated=1" alt="annotated camera">
    </section>
    <section class="card">
      <div class="label">三分类概率</div><div id="probabilities"></div>
      <div class="label" style="margin-top:18px">传感器新鲜度</div><table id="sensors"></table>
    </section>
  </div>

  <div id="modules" class="modules"></div>
  <div class="warning">模块百分比是 TreeSHAP 绝对贡献占比；“推高/压低”表示对高风险类别 margin 的方向，不是四个独立风险概率。</div>
  <details class="card"><summary>查看模型输入</summary><table id="features"></table></details>
</main>
<script>
const $=id=>document.getElementById(id);
const fmt=(v,n=3)=>v===null||v===undefined||!Number.isFinite(Number(v))?"缺失":Number(v).toFixed(n);
const moduleLabels={gps:"GPS",imu:"IMU",radar:"Radar",vision:"Vision"};
const directionLabels={raises:"推高高风险",lowers:"压低高风险",neutral:"中性"};
const levelNames=["静默","轻振","强振"];
function renderModules(s,p){
  const modules=p.module_contributions||{}, outputs=s.modules||{}, sensors=s.sensors||{};
  $("modules").innerHTML=["gps","imu","radar","vision"].map(name=>{
    const output=outputs[name]||{}, sensor=output.sensor||sensors[name]||{};
    if((sensor.status||"unknown")!=="active"){
      return `<section class="card"><div class="module-head"><div><div class="module-name">${moduleLabels[name]}</div><span class="pill">${sensor.status||"unknown"}</span></div><div class="module-pct">--</div></div><div class="meter"><i style="width:0"></i></div><div class="direction neutral">传感器不可用 · 不展示残留贡献</div></section>`;
    }
    const m=output.contribution||modules[name]||{};
    if((s.decision_source||"xgboost")!=="xgboost"){
      const rawScore=s[`${name}_score`], score=Number(rawScore);
      const pct=rawScore!==null&&rawScore!==undefined&&Number.isFinite(score)?Math.max(0,Math.min(100,score*100)):null;
      return `<section class="card"><div class="module-head"><div><div class="module-name">${moduleLabels[name]}</div><span class="pill">${sensor.status||"unknown"}</span></div><div class="module-pct">${pct===null?"--":fmt(pct,1)+"%"}</div></div><div class="meter"><i style="width:${pct||0}%"></i></div><div class="direction neutral">确定性降级规则评分</div></section>`;
    }
    const targetMargin=Number((m.class_margin_contributions||{})[p.label]||0), direction=targetMargin>0.000001?"raises":targetMargin<-0.000001?"lowers":"neutral";
    const top=(m.top_features||[]).map(f=>`<div class="feature"><span>${f.name}</span><span>${fmt(f.high_risk_margin,3)}</span></div>`).join("");
    const pct=Math.max(0,Math.min(100,Number(m.importance_pct)||0));
    return `<section class="card"><div class="module-head"><div><div class="module-name">${moduleLabels[name]}</div><span class="pill">${sensor.status||"unknown"}</span></div><div class="module-pct">${fmt(pct,1)}%</div></div><div class="meter"><i style="width:${pct}%"></i></div><div class="direction ${direction}">${directionLabels[direction]} · 对${p.label||"当前类别"} ${fmt(targetMargin,3)}</div><div class="hint">高风险 margin ${fmt(m.high_risk_margin,3)}</div>${top}</section>`;
  }).join("");
}
async function refresh(){
  try{
    const response=await fetch("/api/state",{cache:"no-store"});const s=await response.json();const p=s.prediction||{};
    const finalLevel=s.risk_level??p.level, finalLabel=s.risk_label||p.label||s.status||"等待", finalScore=s.risk_score==null?p.risk_score_100:Number(s.risk_score)*100;
    $("level").textContent=finalLabel;$("level").className="value "+(finalLevel===2?"high":finalLevel===1?"mid":finalLevel===0?"low":"muted");
    $("score").textContent=finalScore==null?"--":`风险评分 ${fmt(finalScore,1)} / 100`;
    $("confidence").textContent=p.confidence==null?"--":fmt(p.confidence*100,1)+"%";
    $("probHint").textContent=`推理 ${fmt(p.inference_ms,2)} ms`;
    $("status").textContent=s.status||"unknown";$("status").className="value "+(s.status==="active"?"ok":s.status==="warming_up"?"mid":"error");
    $("stateAge").textContent=`${s.schema_version||"unknown"} · old rules ${s.old_rules_loaded?"ON":"OFF"}`;
    const motor=s.motor||{}, motorHealthy=motor.connected&&!motor.faulted;
    $("motor").textContent=motorHealthy?(motor.gate_open?"已武装":"已连接"):(motor.enabled?"故障":"关闭");
    $("motor").className="value "+(motorHealthy?(motor.commanded_level>0?"mid":"ok"):motor.enabled?"error":"muted");
    $("motorDetail").textContent=`${levelNames[motor.commanded_level||0]} · ${motor.gate_reason||"--"} · ${motor.last_command?.pattern||"silent"}`;
    const probs=p.probabilities||{};$("probabilities").innerHTML=Object.entries(probs).map(([k,v])=>`<div style="margin-top:13px">${k}<span style="float:right">${fmt(v*100,2)}%</span><div class="prob"><i style="width:${v*100}%"></i></div></div>`).join("");
    $("sensors").innerHTML=Object.entries(s.sensors||{}).map(([k,v])=>`<tr><td>${k}</td><td>${v.status||v} · ${v.age_ms==null?"--":fmt(v.age_ms,0)+"ms"}</td></tr>`).join("");
    $("features").innerHTML=Object.entries(s.features||{}).map(([k,v])=>`<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("");
    renderModules(s,p);
  }catch(e){$("status").textContent="连接失败";$("status").className="value error"}
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
    state, version, age_ms = _state_store.get_snapshot()
    motor = dict(state.get("motor") or {})
    motor_control = bool(state.get("motor_control"))
    motor_healthy = (
        not motor_control
        or (
            bool(motor.get("connected"))
            and not bool(motor.get("faulted"))
        )
    )
    state_fresh = age_ms <= 2000.0
    healthy = state_fresh and motor_healthy
    health_status = (
        "ok" if healthy
        else "motor_fault" if not motor_healthy
        else "stale"
    )
    return JSONResponse({
        "status": health_status,
        "schema_version": state.get("schema_version", "xgb-risk-v2"),
        "service": "rider-xgb",
        "decision_engine": state.get(
            "decision_engine",
            "xgboost-with-deterministic-degradation",
        ),
        "old_rules_loaded": bool(state.get("old_rules_loaded", False)),
        "motor_control": motor_control,
        "motor_connected": bool(motor.get("connected")),
        "motor_faulted": bool(motor.get("faulted")),
        "motor_gate_open": bool(motor.get("gate_open")),
        "state_version": version,
        "state_age_ms": round(age_ms, 1),
        "camera_available": bool(_camera and _camera.is_available),
    }, status_code=200 if healthy else 503)


def _draw_vision_overlay(bgr, vision_result):
    """Draw display-only YOLO boxes and the Road-ADAS drivable mask."""
    import cv2
    import numpy as np

    out = bgr.copy()
    height, width = out.shape[:2]
    detections = list(getattr(vision_result, "detections", []) or [])
    road_ratio = None
    mask = getattr(vision_result, "drivable_mask", None)
    if mask is not None and getattr(mask, "size", 0) > 0:
        mask_resized = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
        road = mask_resized > 0
        road_ratio = float(np.count_nonzero(road)) / max(1, road.size)
        if np.any(road):
            road_color = np.array([40, 190, 40], dtype=np.float32)
            out_float = out.astype(np.float32)
            out_float[road] = out_float[road] * 0.62 + road_color * 0.38
            out = np.clip(out_float, 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                road.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, (0, 255, 255), 1)

    road_text = (
        f"{road_ratio * 100:.1f}%" if road_ratio is not None else "N/A"
    )
    status_text = (
        f"Road-ADAS mask: {road_text} | YOLO26n boxes: {len(detections)}"
    )
    cv2.rectangle(out, (0, 0), (width - 1, 38), (16, 24, 32), -1)
    cv2.putText(
        out,
        status_text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for detection in detections:
        x1, y1, x2, y2 = (
            int(round(float(value))) for value in detection.bbox
        )
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        on_road = getattr(detection, "in_drivable_area", None)
        risk = float(getattr(detection, "visual_risk", 0.0) or 0.0)
        if risk >= 0.7 or on_road is True:
            color = (0, 0, 255)
        elif risk >= 0.3 or on_road is None:
            color = (0, 200, 255)
        else:
            color = (0, 255, 0)

        class_name = str(getattr(detection, "class_name", "object"))
        confidence = float(getattr(detection, "confidence", 0.0) or 0.0)
        road_label = (
            "ROAD" if on_road is True
            else "OFF-ROAD" if on_road is False
            else "ROAD?"
        )
        label = f"{class_name} {confidence:.2f} {road_label}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_top = max(0, y1 - text_height - 7)
        cv2.rectangle(
            out,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 5), y1),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 2, max(text_height, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _get_video_jpeg() -> bytes:
    global _overlay_cache_key, _overlay_cache_jpeg

    if _camera is None:
        return b""
    try:
        bgr, _, frame_id = _camera.get_bgr_frame_with_timestamp()
        vision_result = (
            _vision_adapter.get_latest_vision_result()
            if _vision_adapter is not None else None
        )
        if bgr is None or vision_result is None:
            return _camera.get_jpeg_frame()

        cache_key = (frame_id, id(vision_result))
        with _overlay_lock:
            if cache_key == _overlay_cache_key and _overlay_cache_jpeg:
                return _overlay_cache_jpeg

            import cv2

            annotated = _draw_vision_overlay(bgr, vision_result)
            encoded, buffer = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if not encoded:
                return _camera.get_jpeg_frame()
            _overlay_cache_key = cache_key
            _overlay_cache_jpeg = buffer.tobytes()
            return _overlay_cache_jpeg
    except Exception:
        return _camera.get_jpeg_frame()


def _frames():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame = _get_video_jpeg()
        yield boundary + frame + b"\r\n"
        time.sleep(0.08)


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
