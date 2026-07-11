"""COCO / YOLO 类别 → 项目风险类别映射（单一事实来源，勿在业务里写死字符串）。

设计：最终输出「类别无关」。所有可能阻挡骑行路径的目标统一归为单一 obstacle 类，
按同一（较高）基础危险度处理——障碍物是否危险主要取决于其位置、接近度和尺度，
而非类别本身（见设计报告 §4.1.2）。原始 COCO 类名仍保留在
`DetectionResult.class_name` 中，仅供可视化 / 统计，不参与风险分级。
"""

# 视为「障碍物」的 COCO 类集合（可按需扩充；全部映射到单一 obstacle 类）。
OBSTACLE_COCO_CLASSES = (
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
)

# 类别无关输出的唯一风险类别标签。
OBSTACLE_CLASS = "obstacle"

# 所有目标类别 → 单一 obstacle（类别无关）。
TARGET_CLASS_MAPPING = {name: OBSTACLE_CLASS for name in OBSTACLE_COCO_CLASSES}

TARGET_CLASSES = set(TARGET_CLASS_MAPPING.keys())

# 类别无关：单一 obstacle 基础危险度。统一按高危处理——安全预警中漏报代价高于误报。
CLASS_BASE_RISK = {
    OBSTACLE_CLASS: 0.9,
}
