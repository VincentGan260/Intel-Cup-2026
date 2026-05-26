from dataclasses import dataclass
from typing import Protocol, List
import numpy as np


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[float]   # [x1, y1, x2, y2] in pixel coordinates
    risk_class: str     # "low", "medium", "high" 等


class BaseDetector(Protocol):
    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        ...