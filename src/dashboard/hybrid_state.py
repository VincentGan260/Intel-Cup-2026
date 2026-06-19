"""Hybrid 模式状态生成器。

调用真实 RiskModel + RiskLevelClassifier，但 GPS/IMU/Radar 使用默认无效数据（不打开串口）。
Vision 可选真实接入，摄像头由 Dashboard 的 CameraFrameProducer 统一管理。

异常时自动 fallback 到 mock 模式，保证页面不崩溃。
"""

from __future__ import annotations

import time
from typing import Optional


def build_hybrid_state(
    *,
    risk_model,
    classifier,
    synchronizer,
    vision_adapter: Optional[object] = None,
    bgr_frame=None,
    camera_available: bool = False,
) -> dict:
    """使用真实 RiskModel 构建一帧系统状态。

    GPS / IMU / Radar 全部使用默认无效 dataclass（不打开串口）。
    Vision 仅在 vision_adapter 和 bgr_frame 都存在时执行推理。

    Args:
        risk_model: 已初始化的 RiskModel 实例
        classifier: 已初始化的 RiskLevelClassifier 实例
        synchronizer: 已初始化的 Synchronizer 实例
        vision_adapter: VisionAdapter 或 None。为 None 时 vision=off
        bgr_frame: BGR 图像 (H, W, 3) 或 None
        camera_available: 摄像头当前是否可用

    Returns:
        dict，兼容 DashboardStateStore.set_state() 格式。
        mode="hybrid"。异常时 fallback 到 mode="mock"。
    """
    try:
        # ── 1. 构造默认无效传感器数据（不打开串口） ──
        from src.fusion.data_types import GPSData, IMUData, RadarData

        ts = time.time()
        gps = GPSData(timestamp=ts, valid=False)
        imu = IMUData(timestamp=ts, valid=False)
        radar = RadarData(timestamp=ts, valid=False)

        # ── 2. 视觉（可选真实） ──
        vision_data = None
        vision_mode = "off"
        vision_infer_ms = 0.0
        if vision_adapter is not None and bgr_frame is not None:
            try:
                _t0 = time.time()
                vision_data = vision_adapter.process(bgr_frame)
                vision_infer_ms = round((time.time() - _t0) * 1000.0, 2)
                vision_mode = "real" if vision_data.valid else "invalid"
            except Exception as e:
                print(f"[HybridState] 视觉处理异常: {e}")
                vision_mode = "invalid"

        # ── 3. 同步 → 融合帧 ──
        synchronizer.update_gps(gps)
        synchronizer.update_imu(imu)
        synchronizer.update_radar(radar)
        if vision_data is not None:
            synchronizer.update_vision(vision_data)
        fusion = synchronizer.build_frame()

        # ── 4. 风险计算 ──
        risk_items_dict, weights = risk_model.compute(fusion)
        level, label = classifier.classify(risk_items_dict["risk_score"])

        # ── 5. 视觉详情 ──
        FRAME_W, FRAME_H = 640, 480
        if vision_data is not None and vision_data.valid:
            objects_list = []
            for obj in vision_data.objects[:20]:
                try:
                    obj_dict = {
                        "class_name": str(getattr(obj, "class_name", "")),
                        "confidence": round(float(getattr(obj, "confidence", 0.0)), 4),
                        "bbox": [
                            round(float(getattr(obj, "bbox", (0, 0, 0, 0))[0]), 4),
                            round(float(getattr(obj, "bbox", (0, 0, 0, 0))[1]), 4),
                            round(float(getattr(obj, "bbox", (0, 0, 0, 0))[2]), 4),
                            round(float(getattr(obj, "bbox", (0, 0, 0, 0))[3]), 4),
                        ],
                        "risk": round(float(getattr(obj, "visual_risk", 0.0)), 4),
                    }
                except Exception:
                    continue
                objects_list.append(obj_dict)

            vision_details = {
                "valid": True,
                "object_count": len(vision_data.objects),
                "person_count": vision_data.person_count,
                "vehicle_count": vision_data.vehicle_count,
                "max_confidence": round(vision_data.max_confidence, 4),
                "drivable_area_ratio": round(vision_data.drivable_area_ratio, 4),
                "max_visual_risk": round(vision_data.max_visual_risk, 4),
                "objects": objects_list,
                "frame_size": {"width": FRAME_W, "height": FRAME_H},
            }
        else:
            vision_details = {
                "valid": False,
                "object_count": 0,
                "person_count": 0,
                "vehicle_count": 0,
                "max_confidence": 0.0,
                "drivable_area_ratio": 0.0,
                "max_visual_risk": 0.0,
                "objects": [],
                "frame_size": {"width": FRAME_W, "height": FRAME_H},
            }

        # ── 6. 组装 Dashboard 状态 ──
        return {
            "performance": {
                "vision_infer_ms": vision_infer_ms,
            },
            # ▼ 以下字段继续保持不变

            "timestamp": time.time(),
            "risk_score": round(max(0.0, min(1.0, risk_items_dict["risk_score"])), 4),
            "risk_level": level,
            "risk_label": label,
            "risk_items": {
                "obs": round(max(0.0, min(1.0, risk_items_dict["R_obs"])), 4),
                "dist": round(max(0.0, min(1.0, risk_items_dict["R_dist"])), 4),
                "pose": round(max(0.0, min(1.0, risk_items_dict["R_pose"])), 4),
                "speed": round(max(0.0, min(1.0, risk_items_dict["R_speed"])), 4),
            },
            "weights": {
                "obs": round(weights.get("obs", 0.0), 4),
                "dist": round(weights.get("dist", 0.0), 4),
                "pose": round(weights.get("pose", 0.0), 4),
                "speed": round(weights.get("speed", 0.0), 4),
            },
            "sensors": {
                "camera": camera_available,
                "vision": vision_mode,
                "radar": "mock",
                "imu": "mock",
                "gps": "mock",
            },
            "mode": "hybrid",
            "message": f"hybrid active | vision={vision_mode}",
            "vision_details": vision_details,
        }

    except Exception as e:
        # ── fallback 到 mock ──
        print(f"[HybridState] 异常，fallback 到 mock: {e}")
        from src.dashboard.mock_state import build_mock_state

        state = build_mock_state(camera_available=camera_available)
        state["message"] = f"fallback to mock: {e}"
        return state
