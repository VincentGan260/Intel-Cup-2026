"""Mock 风险状态生成器。

生成缓慢变化的 mock 风险评分，供 Dashboard 后台状态更新线程调用。
不依赖 FastAPI、不依赖全局 server 对象。

权重与 configs/risk_params.yaml 保持一致：
  obs=0.30, dist=0.35, pose=0.20, speed=0.15
"""

from __future__ import annotations

import math
import time

_MOCK_RISK_THRESHOLDS = {"low": 0.30, "high": 0.70}


def build_mock_state(camera_available: bool = False) -> dict:
    """生成一帧 mock 系统状态 JSON。

    risk_score 在 [0, 1] 之间缓慢振荡（sin 波，周期约 15 秒）。
    四项风险分解使用不同相位同步振荡。
    风险等级/标签根据配置阈值自动判定。

    Args:
        camera_available: 摄像头当前是否可用

    Returns:
        dict，结构兼容前端 /api/state 期望格式
    """
    t = time.time()

    # 主风险分数：周期 ~15 秒，范围 [0, 1]
    risk_score = 0.5 + 0.5 * math.sin(t * 0.42)
    imu_risk_score = 0.12 + 0.05 * abs(math.sin(t * 0.7))

    # 四项分解：不同相位
    risk_items = {
        "obs":   0.5 + 0.5 * math.sin(t * 0.42 + 0.0),
        "dist":  0.5 + 0.5 * math.sin(t * 0.42 + 1.2),
        "pose":  0.5 + 0.5 * math.sin(t * 0.42 + 2.4),
        "speed": 0.5 + 0.5 * math.sin(t * 0.42 + 3.6),
    }

    # 钳制到 [0, 1]
    risk_score = max(0.0, min(1.0, risk_score))
    for k in risk_items:
        risk_items[k] = round(max(0.0, min(1.0, risk_items[k])), 4)

    # 风险等级
    if risk_score < _MOCK_RISK_THRESHOLDS["low"]:
        level = 0
        label = "低风险"
    elif risk_score < _MOCK_RISK_THRESHOLDS["high"]:
        level = 1
        label = "中风险"
    else:
        level = 2
        label = "高风险"

    return {
        "timestamp": time.time(),
        "risk_score": round(risk_score, 4),
        "radar_score": risk_items["dist"],
        "vision_score": risk_items["obs"],
        "imu_score": round(imu_risk_score, 2),
        "risk_level": level,
        "risk_label": label,
        "risk_items": risk_items,
        "weights": {
            "obs": 0.30,
            "dist": 0.35,
            "pose": 0.20,
            "speed": 0.15,
        },
        "sensors": {
            "camera": "real" if camera_available else "off",
            "vision": "mock",
            "radar": "mock",
            "imu": "mock",
            "gps": "mock",
            "motor": "mock",
        },
        "hardware_status": {
            "camera": {
                "status": "active" if camera_available else "unavailable",
                "reason": "camera is open" if camera_available else "camera is not available in mock state",
            },
            "vision": {"status": "mock", "reason": "vision is simulated in mock state"},
            "radar": {"status": "mock", "reason": "radar is simulated in mock state"},
            "gps": {"status": "mock", "reason": "gps is simulated in mock state"},
            "imu": {"status": "mock", "reason": "imu is simulated in mock state"},
            "motor": {"status": "mock", "reason": "motor is simulated in mock state"},
        },
        "mode": "mock",
        "message": "mock state active",
        "imu_data": {
            "connected": False,
            "valid": True,
            "roll": round(4.0 * math.sin(t * 0.8), 2),
            "pitch": round(2.5 * math.sin(t * 0.65), 2),
            "yaw": round(12.0 * math.sin(t * 0.2), 2),
            "acc_x": round(0.4 * math.sin(t), 2),
            "acc_y": round(0.3 * math.sin(t * 0.9), 2),
            "acc_z": round(9.8 + 0.2 * math.sin(t * 1.2), 2),
            "gyro_x": round(1.5 * math.sin(t * 0.7), 2),
            "gyro_y": round(1.2 * math.sin(t * 0.6), 2),
            "gyro_z": round(2.0 * math.sin(t * 0.4), 2),
            "brake_score": 0.08,
            "bump_score": 0.05,
            "tilt_score": 0.12,
            "risk_level": 0,
            "risk_score": round(imu_risk_score, 2),
            "risk_status": "usable",
            "turn_compensation_status": "usable",
            "roll_error_deg": round(2.0 * math.sin(t * 0.8), 1),
            "outward_rate_deg_s": round(abs(1.5 * math.sin(t * 0.7)), 1),
            "time_to_critical_s": None,
        },
        "performance": {
            "vision_infer_ms": 0.0,
        },
        "vision_details": {
            "valid": False,
            "object_count": 0,
            "person_count": 0,
            "vehicle_count": 0,
            "max_confidence": 0.0,
            "drivable_area_ratio": 0.0,
            "max_visual_risk": 0.0,
            "objects": [],
            "frame_size": {"width": 640, "height": 480},
        },
    }
