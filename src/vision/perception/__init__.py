"""感知层：检测与分割的组合流程。"""

from src.vision.perception.target_on_road import attach_drivable_area_flag
from src.vision.perception.vision_pipeline import VisionPipeline

__all__ = ["VisionPipeline", "attach_drivable_area_flag"]
