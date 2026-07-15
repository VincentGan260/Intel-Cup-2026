"""A/B test OpenVINO default compilation against explicit LATENCY mode.

This test deliberately does not read or modify the production performance
configuration. Both variants use the production FP16@GPU segmentation model,
INT8@NPU detection model, identical images, and a persistent two-worker pool
matching VisionPipeline's GPU/NPU overlap.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vision.test_gpu_npu_parallel import (  # noqa: E402
    RoadAdasSegmenter,
    YOLO26nInt8Detector,
    find_test_images,
)


def p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values), 95))


def properties(compiled) -> dict[str, str]:
    result = {}
    for key in (
        "PERFORMANCE_HINT",
        "EXECUTION_DEVICES",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "NUM_STREAMS",
    ):
        try:
            result[key] = str(compiled.get_property(key))
        except Exception as exc:
            result[key] = f"unsupported:{type(exc).__name__}"
    return result


class Variant:
    def __init__(self, name: str, compile_config: dict[str, str]) -> None:
        self.name = name
        self.compile_config = compile_config
        self.segmenter = RoadAdasSegmenter(
            PROJECT_ROOT / "models/openvino/road-adas-fp16/road-segmentation-adas-0001.xml",
            "GPU",
            compile_config,
        )
        self.detector = YOLO26nInt8Detector(
            PROJECT_ROOT / "models/yolo26n_v2_int8_openvino_model/best.xml",
            "NPU",
            compile_config,
        )
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"latency-ab-{name}")

    def process(self, frame: np.ndarray):
        seg_future = self.executor.submit(self.segmenter.infer_sync, frame)
        det_future = self.executor.submit(self.detector.infer_sync, frame)
        return seg_future.result(), det_future.result()

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def runtime_properties(self) -> dict:
        return {
            "GPU": properties(self.segmenter.compiled),
            "NPU": properties(self.detector.compiled),
        }


def read_frames(paths: list[Path], count: int) -> list[np.ndarray]:
    selected = paths[: min(len(paths), count)]
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in selected]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError("failed to read A/B test images")
    return frames


def detections_equal(left, right) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a.class_name != b.class_name:
            return False
        if not np.isclose(a.confidence, b.confidence, rtol=1e-5, atol=1e-5):
            return False
        if not np.allclose(a.bbox, b.bbox, rtol=1e-5, atol=1e-4):
            return False
    return True


def consistency(default: Variant, latency: Variant, frames: list[np.ndarray], count: int) -> dict:
    matches = []
    for frame in frames[:count]:
        default_seg, default_det = default.process(frame)
        latency_seg, latency_det = latency.process(frame)
        matches.append(
            np.array_equal(default_seg.road_mask, latency_seg.road_mask)
            and detections_equal(default_det, latency_det)
        )
    return {"samples": len(matches), "matches": sum(matches), "passed": all(matches)}


def warmup(variant: Variant, frames: list[np.ndarray], iterations: int) -> None:
    for index in range(iterations):
        variant.process(frames[index % len(frames)])


def measure(variant: Variant, frames: list[np.ndarray], iterations: int) -> dict:
    samples = []
    errors = []
    for index in range(iterations):
        start = time.perf_counter()
        try:
            variant.process(frames[index % len(frames)])
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            break
        samples.append((time.perf_counter() - start) * 1000.0)
    if not samples:
        raise RuntimeError(f"variant {variant.name} produced no samples: {errors}")
    return {
        "samples": len(samples),
        "mean_ms": round(statistics.mean(samples), 3),
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(p95(samples), 3),
        "max_ms": round(max(samples), 3),
        "visual_hz": round(1000.0 / statistics.mean(samples), 3),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="datasets/det/dawn/images")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--consistency-samples", type=int, default=20)
    args = parser.parse_args()

    # The legacy finder combines a direct and recursive glob, which can return
    # top-level files twice. A/B coverage must count unique source images.
    paths = sorted(set(find_test_images(PROJECT_ROOT / args.test_dir)))
    if not paths:
        raise SystemExit("no A/B test images")
    frames = read_frames(paths, max(args.iters, args.consistency_samples))

    default = Variant("default", {})
    latency = Variant("latency", {"PERFORMANCE_HINT": "LATENCY"})
    try:
        output_consistency = consistency(
            default, latency, frames, args.consistency_samples
        )
        warmup(default, frames, args.warmup)
        warmup(latency, frames, args.warmup)

        results = {"default": [], "latency": []}
        # Alternate order to reduce thermal/order bias.
        orders = [(default, latency), (latency, default), (default, latency)]
        for round_index in range(args.rounds):
            order = orders[round_index % len(orders)]
            for variant in order:
                row = measure(variant, frames, args.iters)
                row["round"] = round_index + 1
                results[variant.name].append(row)

        default_p95 = statistics.median(row["p95_ms"] for row in results["default"])
        latency_p95 = statistics.median(row["p95_ms"] for row in results["latency"])
        improvement = (default_p95 - latency_p95) / default_p95 * 100.0
        runtime_properties = {
            "default": default.runtime_properties(),
            "latency": latency.runtime_properties(),
        }
        properties_identical = runtime_properties["default"] == runtime_properties["latency"]
        errors = sum(len(row["errors"]) for rows in results.values() for row in rows)
        decision = (
            "enable"
            if improvement > 5.0 and output_consistency["passed"] and errors == 0
            else "keep_default"
        )
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deployment": {"segmentation": "FP16@GPU", "detection": "INT8@NPU"},
            "test": {
                "images": len(frames),
                "warmup": args.warmup,
                "iterations_per_round": args.iters,
                "rounds": args.rounds,
                "parallel": True,
            },
            "runtime_properties": runtime_properties,
            "runtime_properties_identical": properties_identical,
            "output_consistency": output_consistency,
            "results": results,
            "median_p95_ms": {
                "default": round(default_p95, 3),
                "latency": round(latency_p95, 3),
            },
            "p95_improvement_percent": round(improvement, 3),
            "device_errors": errors,
            "threshold_percent": 5.0,
            "decision": decision,
        }
        output = PROJECT_ROOT / "runs/latency_ab/openvino_latency_ab.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"saved: {output}")
    finally:
        default.close()
        latency.close()


if __name__ == "__main__":
    main()
