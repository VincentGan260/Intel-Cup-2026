"""Evaluate the deployed OpenVINO white-tile model on the fixed validation split."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.segmentation.segmenter import build_segmenter_from_config


DATASET = PROJECT_ROOT / "white_tile_road" / "dataset"
CONFIG = PROJECT_ROOT / "configs" / "vision" / "segmentation_white_tile_fp16.yaml"
OUTPUT = PROJECT_ROOT / "runs" / "vision" / "white_tile_deeplab" / "openvino_predictions"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="Override OpenVINO device, e.g. CPU")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.device:
        config.setdefault("openvino", {})["device"] = args.device
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    names = [
        line.strip()
        for line in (DATASET / "splits" / "val.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ious: list[float] = []
    recalls: list[float] = []
    latencies: list[float] = []
    for name in names:
        image = cv2.imread(str(DATASET / "images" / name), cv2.IMREAD_COLOR)
        gt = cv2.imread(
            str(DATASET / "masks" / f"{Path(name).stem}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        started = time.perf_counter()
        result = segmenter.infer(image)
        latencies.append((time.perf_counter() - started) * 1000.0)
        pred = result.drivable_mask > 0
        valid = gt != 255
        road = (gt == 1) & valid
        pred = pred & valid
        intersection = int(np.sum(pred & road))
        union = int(np.sum(pred | road))
        iou = intersection / max(1, union)
        recall = intersection / max(1, int(np.sum(road)))
        ious.append(iou)
        recalls.append(recall)
        cv2.imwrite(str(OUTPUT / f"{Path(name).stem}.png"), pred.astype(np.uint8) * 255)
        print(f"{name}: IoU={iou:.4f}, recall={recall:.4f}")

    print(f"mean IoU={np.mean(ious):.4f}")
    print(f"mean recall={np.mean(recalls):.4f}")
    print(f"mean latency={np.mean(latencies):.2f} ms (includes first compile warm-up)")


if __name__ == "__main__":
    main()
