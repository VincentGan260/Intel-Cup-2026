"""公共 VisionPipeline 并行行为回归测试（不依赖 OpenVINO 硬件）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.common.interfaces import BaseDetector, BaseSegmenter
from src.vision.common.types import DetectionResult, SegmentationResult
from src.vision.perception.vision_pipeline import VisionPipeline


class SlowDetector(BaseDetector):
    def infer(self, frame: np.ndarray) -> list[DetectionResult]:
        time.sleep(0.08)
        return [DetectionResult("obstacle", "obstacle", 0.9, (1.0, 1.0, 4.0, 4.0))]


class SlowSegmenter(BaseSegmenter):
    def infer(self, frame: np.ndarray) -> SegmentationResult:
        time.sleep(0.08)
        mask = np.ones(frame.shape[:2], dtype=np.uint8)
        return SegmentationResult(drivable_mask=mask, drivable_ratio=1.0)


def run(parallel: bool):
    pipeline = VisionPipeline(
        SlowDetector(),
        SlowSegmenter(),
        parallel_inference=parallel,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        start = time.perf_counter()
        result = pipeline.process(frame)
        elapsed = time.perf_counter() - start
        return result, elapsed
    finally:
        pipeline.close()


def main() -> None:
    sequential_result, sequential_s = run(False)
    parallel_result, parallel_s = run(True)

    assert parallel_s < sequential_s * 0.75, (sequential_s, parallel_s)
    assert len(parallel_result.detections) == len(sequential_result.detections)
    assert np.array_equal(parallel_result.drivable_mask, sequential_result.drivable_mask)
    assert parallel_result.detections[0].in_drivable_area is True
    print(
        f"PASS: sequential={sequential_s * 1000:.1f}ms, "
        f"parallel={parallel_s * 1000:.1f}ms, outputs consistent"
    )


if __name__ == "__main__":
    main()
