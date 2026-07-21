"""Run the white-tile OpenVINO segmenter on a video and save an overlay."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.segmentation.segmenter import build_segmenter_from_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vision" / "segmentation_white_tile_cpu.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "vision" / "white_tile_video" / "overlay.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="White-tile road segmentation video demo")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None, help="OpenVINO device override, e.g. CPU or GPU")
    parser.add_argument("--alpha", type=float, default=0.45, help="Green overlay opacity")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the whole video")
    parser.add_argument("--show", action="store_true", help="Show a live preview; press Q to stop")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.device:
        config.setdefault("openvino", {})["device"] = args.device
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_count = 0
    inference_seconds = 0.0
    started_all = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            started = time.perf_counter()
            result = segmenter.infer(frame)
            inference_seconds += time.perf_counter() - started
            mask = result.drivable_mask.astype(bool)

            color = np.zeros_like(frame)
            color[:, :] = (0, 210, 0)
            blended = cv2.addWeighted(frame, 1.0 - args.alpha, color, args.alpha, 0.0)
            overlay = frame.copy()
            overlay[mask] = blended[mask]
            cv2.putText(
                overlay,
                f"road={100.0 * float(mask.mean()):.1f}%",
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(overlay)
            frame_count += 1

            if args.show:
                cv2.imshow("white-tile road segmentation", overlay)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            if frame_count % 100 == 0:
                print(f"processed {frame_count} frames", flush=True)
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_all
    model_fps = frame_count / max(inference_seconds, 1e-9)
    print(f"done: frames={frame_count}, elapsed={elapsed:.1f}s, inference_fps={model_fps:.1f}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
