"""COCO / YOLO 类别到项目风险类别映射（单一事实来源，勿在业务里写死字符串）。"""

TARGET_CLASS_MAPPING = {
    "person": "pedestrian",
    "bicycle": "non_motor_vehicle",
    "motorcycle": "non_motor_vehicle",
    "car": "motor_vehicle",
    "bus": "motor_vehicle",
    "truck": "motor_vehicle",
}

TARGET_CLASSES = set(TARGET_CLASS_MAPPING.keys())

CLASS_BASE_RISK = {
    "pedestrian": 0.9,
    "motor_vehicle": 0.85,
    "non_motor_vehicle": 0.75,
}
