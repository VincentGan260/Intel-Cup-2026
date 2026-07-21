"""Validate the real VisionPipeline serial/parallel deployment contract.

Uses the same factories and YAML files as every VisionAdapter consumer. It
checks output consistency, compares P95 across repeated rounds, and can soak
the GPU/NPU path for the required 30 minutes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.detection.detector import build_detector_from_config
from src.vision.perception.vision_pipeline import VisionPipeline
from src.vision.segmentation.segmenter import build_segmenter_from_config
from src.vision.profile_selector import select_segmentation_config


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def build_pipeline(parallel: bool) -> VisionPipeline:
    pipeline_cfg = load_yaml(PROJECT_ROOT / "configs/vision/vision_pipeline.yaml")
    det_cfg = load_yaml(resolve(pipeline_cfg["detection_config"]))
    seg_cfg = load_yaml(resolve(select_segmentation_config(pipeline_cfg)))

    contract = pipeline_cfg["deployment_contract"]
    actual_det = det_cfg["detector"]
    actual_seg = seg_cfg["openvino"]
    assert str(actual_det["device"]).upper() == str(contract["detection"]["device"]).upper()
    assert str(actual_det["precision"]).upper() == str(contract["detection"]["precision"]).upper()
    assert str(actual_seg["device"]).upper() == str(contract["segmentation"]["device"]).upper()
    assert str(actual_seg["inference_precision"]).upper() == str(contract["segmentation"]["precision"]).upper()

    return VisionPipeline(
        build_detector_from_config(det_cfg, project_root=PROJECT_ROOT),
        build_segmenter_from_config(seg_cfg, project_root=PROJECT_ROOT),
        enable_segmentation=True,
        parallel_inference=parallel,
    )


def find_images(path: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def outputs_equal(left, right) -> bool:
    if len(left.detections) != len(right.detections):
        return False
    if not np.array_equal(left.drivable_mask, right.drivable_mask):
        return False
    for a, b in zip(left.detections, right.detections):
        if (a.class_name, a.risk_class, a.in_drivable_area) != (
            b.class_name, b.risk_class, b.in_drivable_area
        ):
            return False
        if not np.isclose(a.confidence, b.confidence, rtol=1e-5, atol=1e-5):
            return False
        if not np.allclose(a.bbox, b.bbox, rtol=1e-5, atol=1e-4):
            return False
    return True


def consistency_check(images: list[Path], count: int) -> dict:
    selected = images[: min(count, len(images))]
    sequential = build_pipeline(False)
    try:
        references = [sequential.process(read_image(path)) for path in selected]
    finally:
        sequential.close()

    parallel = build_pipeline(True)
    try:
        matches = [
            outputs_equal(reference, parallel.process(read_image(path)))
            for path, reference in zip(selected, references)
        ]
    finally:
        parallel.close()
    return {"samples": len(matches), "matches": sum(matches), "passed": all(matches)}


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def benchmark(parallel: bool, images: list[Path], warmup: int, iters: int, rounds: int) -> dict:
    pipeline = build_pipeline(parallel)
    frames = [read_image(path) for path in images[: max(1, min(len(images), iters))]]
    try:
        for index in range(warmup):
            pipeline.process(frames[index % len(frames)])
        round_results = []
        for round_index in range(rounds):
            samples = []
            for index in range(iters):
                start = time.perf_counter()
                pipeline.process(frames[index % len(frames)])
                samples.append((time.perf_counter() - start) * 1000.0)
            round_results.append({
                "round": round_index + 1,
                "mean_ms": round(statistics.mean(samples), 3),
                "p50_ms": round(statistics.median(samples), 3),
                "p95_ms": round(percentile95(samples), 3),
                "max_ms": round(max(samples), 3),
            })
        return {
            "mode": "parallel" if parallel else "sequential",
            "rounds": round_results,
            "median_p95_ms": round(statistics.median(r["p95_ms"] for r in round_results), 3),
        }
    finally:
        pipeline.close()


def stability_check(images: list[Path], minutes: float) -> dict:
    if minutes <= 0:
        return {"minutes": 0.0, "frames": 0, "errors": 0, "passed": True}
    pipeline = build_pipeline(True)
    deadline = time.monotonic() + minutes * 60.0
    frames = [read_image(path) for path in images[: min(20, len(images))]]
    count = 0
    errors: list[str] = []
    try:
        while time.monotonic() < deadline:
            try:
                pipeline.process(frames[count % len(frames)])
                count += 1
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                break
    finally:
        pipeline.close()
    return {
        "minutes": minutes,
        "frames": count,
        "errors": len(errors),
        "error_messages": errors,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="datasets/det/dawn/images")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--consistency-samples", type=int, default=20)
    parser.add_argument("--stability-minutes", type=float, default=30.0)
    parser.add_argument(
        "--stability-only",
        action="store_true",
        help="reuse the saved consistency/latency result and only run the soak test",
    )
    args = parser.parse_args()

    images = find_images(resolve(args.test_dir))
    if not images:
        raise SystemExit(f"no test images: {resolve(args.test_dir)}")

    output = PROJECT_ROOT / "runs/parallel_test/production_pipeline_validation.json"
    if args.stability_only:
        if not output.is_file():
            raise SystemExit(f"missing previous validation result: {output}")
        result = json.loads(output.read_text(encoding="utf-8"))
        result["stability"] = stability_check(images, args.stability_minutes)
        result["passed"] = (
            result["p95_improvement_percent"] > 0
            and result["consistency"]["passed"]
            and result["stability"]["passed"]
        )
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"saved: {output}")
        if not result["passed"]:
            raise SystemExit(1)
        return

    consistency = consistency_check(images, args.consistency_samples)
    sequential = benchmark(False, images, args.warmup, args.iters, args.rounds)
    parallel = benchmark(True, images, args.warmup, args.iters, args.rounds)
    improvement = (
        (sequential["median_p95_ms"] - parallel["median_p95_ms"])
        / sequential["median_p95_ms"]
        * 100.0
    )
    stability = stability_check(images, args.stability_minutes)
    passed = improvement > 0 and consistency["passed"] and stability["passed"]
    result = {
        "deployment": {"detection": "INT8@NPU", "segmentation": "FP16@GPU"},
        "consistency": consistency,
        "sequential": sequential,
        "parallel": parallel,
        "p95_improvement_percent": round(improvement, 3),
        "stability": stability,
        "passed": passed,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
