#!/usr/bin/env python3
"""Standalone DK-2500 XGBoost risk monitor.

This entry point owns the sensors while it runs.  It intentionally does not
import the deterministic risk rules, warning arbitration, or any motor driver.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class SensorSnapshot:
    value: Any
    updated_monotonic: float = 0.0
    error: str = ""


class SnapshotStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, SensorSnapshot] = {}

    def publish(self, name: str, value: Any, error: str = "") -> None:
        with self._lock:
            self._values[name] = SensorSnapshot(
                value=copy.deepcopy(value),
                updated_monotonic=time.monotonic(),
                error=error,
            )

    def get(self, name: str, default: Any) -> SensorSnapshot:
        with self._lock:
            snapshot = self._values.get(name)
            return copy.deepcopy(snapshot) if snapshot else SensorSnapshot(default)


def _sensor_worker(
    name: str,
    reader,
    store: SnapshotStore,
    stop_event: threading.Event,
    interval_s: float,
) -> None:
    while not stop_event.is_set():
        try:
            store.publish(name, reader.read_once())
        except Exception as exc:
            store.publish(name, reader.get_latest(), error=str(exc))
        stop_event.wait(interval_s)


def _vision_worker(
    camera,
    adapter,
    store: SnapshotStore,
    stop_event: threading.Event,
) -> None:
    from src.fusion.data_types import VisionData

    last_frame_id = -1
    while not stop_event.is_set():
        frame, _, frame_id = camera.get_bgr_frame_with_timestamp()
        if frame is None or frame_id <= last_frame_id:
            stop_event.wait(0.03)
            continue
        last_frame_id = frame_id
        try:
            store.publish("vision", adapter.process(frame))
        except Exception as exc:
            store.publish(
                "vision", VisionData(timestamp=time.time(), valid=False), str(exc)
            )
        stop_event.wait(0.03)


def _serializable_features(values: dict) -> dict:
    result = {}
    for name, value in values.items():
        if value is None:
            result[name] = None
            continue
        numeric = float(value)
        result[name] = round(numeric, 6) if math.isfinite(numeric) else None
    return result


def _with_freshness(snapshot: SensorSnapshot, max_age_ms: float, default):
    age_ms = (
        max(0.0, (time.monotonic() - snapshot.updated_monotonic) * 1000.0)
        if snapshot.updated_monotonic else None
    )
    value = snapshot.value if snapshot.value is not None else default
    fresh = age_ms is not None and age_ms <= max_age_ms
    if not fresh:
        value = copy.copy(value)
        value.valid = False
    status = (
        "active" if fresh and bool(getattr(value, "valid", False))
        else "invalid" if fresh else "stale" if age_ms is not None else "waiting"
    )
    return value, {
        "status": status,
        "valid": bool(getattr(value, "valid", False)),
        "age_ms": round(age_ms, 1) if age_ms is not None else None,
        "error": snapshot.error,
    }


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, state: dict) -> None:
        record = {
            "timestamp": state["timestamp"],
            "status": state["status"],
            "prediction": state.get("prediction"),
            "features": state.get("features"),
            "sensors": state.get("sensors"),
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.close()


class DisabledCamera:
    """Camera-compatible diagnostic stub used only with --disable-camera."""

    width = 640
    height = 480
    is_available = False

    def get_bgr_frame_with_timestamp(self):
        return None, 0, -1

    def get_jpeg_frame(self) -> bytes:
        return b""

    def release(self) -> None:
        return None


def _build_feature_config(config: dict):
    from src.risk_ml.feature_window import FeatureWindowConfig

    return FeatureWindowConfig.from_mapping({
        "window_s": config["window"]["window_s"],
        "warmup_s": config["window"]["warmup_s"],
        "path_gate_half_width_m": config["radar_features"]["path_gate_half_width_m"],
        "radar_max_abs_angle_deg": config["radar_features"]["max_abs_angle_deg"],
        "vision_center_x_min_ratio": config["vision_features"]["fallback_center_x_min_ratio"],
        "vision_center_x_max_ratio": config["vision_features"]["fallback_center_x_max_ratio"],
        "vision_path_bottom_min_ratio": config["vision_features"]["fallback_path_bottom_min_ratio"],
        "vision_growth_epsilon_per_s": config["vision_features"]["growth_epsilon_per_s"],
        "imu_turn_sign": config["imu_features"]["turn_sign"],
        "gravity_mps2": config["imu_features"]["gravity_mps2"],
        "min_turn_compensation_speed_kmh": config["imu_features"]["min_turn_compensation_speed_kmh"],
        "imu_attention_error_deg": config["imu_features"]["attention_error_deg"],
        "imu_critical_error_deg": config["imu_features"]["critical_error_deg"],
        "imu_attention_outward_rate_deg_s": config["imu_features"]["attention_outward_rate_deg_s"],
        "imu_urgent_min_error_deg": config["imu_features"]["urgent_min_error_deg"],
        "imu_urgent_outward_rate_deg_s": config["imu_features"]["urgent_outward_rate_deg_s"],
        "imu_prediction_horizon_s": config["imu_features"]["prediction_horizon_s"],
        "imu_max_sample_gap_ms": config["imu_features"]["max_sample_gap_ms"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone XGBoost risk monitor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--profile", default="dk2500", choices=["dk2500", "windows"])
    parser.add_argument("--sensor-mode", default="real", choices=["real", "mock"])
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument(
        "--disable-camera", action="store_true",
        help="diagnostic mode only; DK-2500 service leaves the real camera enabled",
    )
    parser.add_argument("--enable-vision", action="store_true")
    parser.add_argument("--vision-config", default="configs/vision/vision_pipeline.yaml")
    parser.add_argument("--runtime-config", default="configs/xgb_runtime.yaml")
    parser.add_argument("--model", default="models/xgboost_risk/risk_classifier.json")
    parser.add_argument("--metadata", default="models/xgboost_risk/metadata.json")
    parser.add_argument("--log", default="data/xgb_live/risk_predictions.jsonl")
    parser.add_argument("--state-hz", type=float, default=None)
    parser.add_argument("--inference-hz", type=float, default=None)
    args = parser.parse_args()

    import yaml

    def rooted(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else ROOT / candidate

    runtime_cfg = yaml.safe_load(rooted(args.runtime_config).read_text(encoding="utf-8"))
    ports_cfg = yaml.safe_load(
        (ROOT / "configs/sensor_ports.yaml").read_text(encoding="utf-8")
    )
    profile_cfg = ports_cfg[args.profile]
    runtime = runtime_cfg["runtime"]
    state_hz = float(args.state_hz or runtime["state_hz"])
    inference_hz = float(args.inference_hz or runtime["inference_hz"])
    if state_hz <= 0 or inference_hz <= 0 or inference_hz > state_hz:
        parser.error("require 0 < inference_hz <= state_hz")

    from src.dashboard.frame_producer import CameraFrameProducer
    from src.dashboard.state_store import DashboardStateStore
    from src.fusion.data_types import GPSData, IMUData, RadarData, VisionData
    from src.risk_ml.feature_window import FEATURE_NAMES, XGBoostFeatureWindow
    from src.risk_ml.predictor import XGBoostRiskPredictor
    from src.sensors.gps_reader import GPSReader
    from src.sensors.imu_reader import IMUReader
    from src.sensors.radar_reader import RadarReader

    predictor = XGBoostRiskPredictor(rooted(args.model), rooted(args.metadata))
    if tuple(predictor.feature_names) != FEATURE_NAMES:
        raise RuntimeError("runtime feature order does not match trained model metadata")
    extractor = XGBoostFeatureWindow(_build_feature_config(runtime_cfg))

    camera_cfg = ports_cfg.get("camera", {})
    if args.disable_camera:
        camera = DisabledCamera()
        camera.width = int(camera_cfg.get("width", 640))
        camera.height = int(camera_cfg.get("height", 480))
    else:
        camera = CameraFrameProducer(
            camera_id=args.camera_id,
            width=int(camera_cfg.get("width", 640)),
            height=int(camera_cfg.get("height", 480)),
        )
    radar_reader = RadarReader(mode=args.sensor_mode, config=profile_cfg["radar"])
    gps_reader = GPSReader(mode=args.sensor_mode, config=profile_cfg["gps"])
    imu_reader = IMUReader(mode=args.sensor_mode, config=profile_cfg["imu"])
    readers = (radar_reader, gps_reader, imu_reader)
    for reader in readers:
        reader.start()

    vision_adapter = None
    if args.enable_vision:
        from src.fusion.vision_adapter import VisionAdapter

        vision_adapter = VisionAdapter(
            pipeline_config_path=str(rooted(args.vision_config)),
            vision_enabled=True,
            use_camera=False,
        )
        vision_adapter.start()
        if not vision_adapter.vision_enabled:
            vision_adapter = None

    snapshots = SnapshotStore()
    state_store = DashboardStateStore()
    state_store.set_state({
        "timestamp": time.time(),
        "status": "starting",
        "decision_engine": "xgboost-only",
        "motor_control": False,
        "prediction": None,
        "features": {},
        "sensors": {},
    })
    logger = JsonlLogger(rooted(args.log))
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_sensor_worker,
            args=("radar", radar_reader, snapshots, stop_event, 0.02),
            daemon=True,
            name="xgb-radar-reader",
        ),
        threading.Thread(
            target=_sensor_worker,
            args=("gps", gps_reader, snapshots, stop_event, 0.10),
            daemon=True,
            name="xgb-gps-reader",
        ),
        threading.Thread(
            target=_sensor_worker,
            args=("imu", imu_reader, snapshots, stop_event, 0.02),
            daemon=True,
            name="xgb-imu-reader",
        ),
    ]
    if vision_adapter is not None:
        threads.append(threading.Thread(
            target=_vision_worker,
            args=(camera, vision_adapter, snapshots, stop_event),
            daemon=True,
            name="xgb-vision-inference",
        ))
    for thread in threads:
        thread.start()

    def inference_loop() -> None:
        update_interval = 1.0 / state_hz
        inference_interval = 1.0 / inference_hz
        next_inference = 0.0
        last_prediction = None
        while not stop_event.is_set():
            started = time.monotonic()
            radar, radar_status = _with_freshness(
                snapshots.get("radar", RadarData()), runtime["radar_stale_ms"], RadarData()
            )
            gps, gps_status = _with_freshness(
                snapshots.get("gps", GPSData()), runtime["gps_stale_ms"], GPSData()
            )
            imu, imu_status = _with_freshness(
                snapshots.get("imu", IMUData()), runtime["imu_stale_ms"], IMUData()
            )
            vision, vision_status = _with_freshness(
                snapshots.get("vision", VisionData()), runtime["vision_stale_ms"], VisionData()
            )
            frame = extractor.update(
                now_monotonic=started,
                gps=gps,
                imu=imu,
                radar=radar,
                vision=vision,
                frame_width=camera.width,
                frame_height=camera.height,
            )
            prediction_error = ""
            predicted_now = False
            if frame.warm and started >= next_inference:
                try:
                    last_prediction = predictor.predict(frame.values)
                    predicted_now = True
                except Exception as exc:
                    prediction_error = str(exc)
                    last_prediction = None
                next_inference = started + inference_interval

            if not frame.warm:
                status = "warming_up"
            elif last_prediction is None:
                status = "model_error"
            elif last_prediction.confidence < float(runtime["confidence_warning_below"]):
                status = "low_confidence"
            else:
                status = "active"
            state = {
                "timestamp": time.time(),
                "status": status,
                "decision_engine": "xgboost-only",
                "old_rules_loaded": False,
                "motor_control": False,
                "prediction": last_prediction.as_dict() if last_prediction else None,
                "prediction_error": prediction_error,
                "features": _serializable_features(frame.values),
                "feature_window": {
                    "warm": frame.warm,
                    "age_s": round(frame.window_age_s, 3),
                    **frame.diagnostics,
                },
                "sensors": {
                    "camera": {
                        "status": "active" if camera.is_available else "waiting",
                        "valid": camera.is_available,
                    },
                    "radar": radar_status,
                    "gps": gps_status,
                    "imu": imu_status,
                    "vision": vision_status if vision_adapter else {
                        "status": "disabled", "valid": False, "age_ms": None, "error": ""
                    },
                },
                "runtime": {
                    **predictor.runtime_info(),
                    "profile": args.profile,
                    "sensor_mode": args.sensor_mode,
                    "state_hz": state_hz,
                    "inference_hz": inference_hz,
                    "confidence_warning_below": runtime["confidence_warning_below"],
                    "vision": (
                        vision_adapter.get_runtime_info() if vision_adapter else None
                    ),
                },
            }
            state_store.set_state(state)
            if predicted_now:
                logger.write(state)
            stop_event.wait(max(0.0, update_interval - (time.monotonic() - started)))

    inference_thread = threading.Thread(
        target=inference_loop, daemon=True, name="xgb-risk-inference"
    )
    inference_thread.start()

    from src.risk_ml import server

    server.inject_runtime(state_store, camera)
    print("=" * 68)
    print("Standalone XGBoost Risk Monitor")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Decision engine: XGBoost only ({len(FEATURE_NAMES)} features)")
    print("Old deterministic rules: NOT LOADED")
    print("Motor control: OFF / NOT IMPORTED")
    print("=" * 68)

    received_signal = threading.Event()

    def request_shutdown(signum, _frame):
        received_signal.set()
        print(f"[XGB] received signal {signum}; shutting down")

    previous_handlers = {}
    for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[shutdown_signal] = signal.getsignal(shutdown_signal)
        signal.signal(shutdown_signal, request_shutdown)

    import uvicorn

    try:
        uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")
    finally:
        stop_event.set()
        camera.release()
        inference_thread.join(timeout=3.0)
        for thread in threads:
            thread.join(timeout=3.0)
        if vision_adapter is not None:
            vision_adapter.stop()
        for reader in readers:
            reader.stop()
        logger.close()
        for shutdown_signal, previous_handler in previous_handlers.items():
            signal.signal(shutdown_signal, previous_handler)


if __name__ == "__main__":
    main()
