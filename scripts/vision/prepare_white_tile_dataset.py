"""Convert Labelme road polygons into a binary semantic-segmentation dataset.

Mask values:
    0   background
    1   road
    255 ignore (excluded from the training loss)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "white_tile_road" / "raw_images"
DEFAULT_OUTPUT = PROJECT_ROOT / "white_tile_road" / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _fill_shapes(mask: np.ndarray, shapes: list[dict], label: str, value: int) -> None:
    for shape in shapes:
        if str(shape.get("label", "")).strip().lower() != label:
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Only polygon shapes are supported, got {shape.get('shape_type')}")
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.shape[0] < 3:
            raise ValueError(f"{label} polygon has fewer than 3 points")
        points[:, 0] = np.clip(points[:, 0], 0, mask.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, mask.shape[0] - 1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], int(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    image_dir = output / "images"
    mask_dir = output / "masks"
    split_dir = output / "splits"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {source}")

    records: list[tuple[str, float, float]] = []
    for image_path in images:
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {json_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        annotation = json.loads(json_path.read_text(encoding="utf-8"))
        if int(annotation.get("imageWidth", width)) != width:
            raise ValueError(f"Width mismatch: {image_path.name}")
        if int(annotation.get("imageHeight", height)) != height:
            raise ValueError(f"Height mismatch: {image_path.name}")

        shapes = list(annotation.get("shapes", []))
        labels = {str(shape.get("label", "")).strip().lower() for shape in shapes}
        unknown = labels - {"road", "ignore"}
        if unknown:
            raise ValueError(f"Unknown labels in {json_path.name}: {sorted(unknown)}")
        if "road" not in labels:
            raise ValueError(f"No road polygon in {json_path.name}")

        mask = np.zeros((height, width), dtype=np.uint8)
        _fill_shapes(mask, shapes, "road", 1)
        # Ignore must be rendered last so it overrides an underlying road polygon.
        _fill_shapes(mask, shapes, "ignore", 255)
        if not np.any(mask == 1):
            raise ValueError(f"Road mask is empty: {json_path.name}")

        resized_image = cv2.resize(
            image, (args.width, args.height), interpolation=cv2.INTER_AREA,
        )
        output_image_path = image_dir / image_path.name
        image_write_options = (
            [cv2.IMWRITE_JPEG_QUALITY, 95]
            if image_path.suffix.lower() in {".jpg", ".jpeg"}
            else []
        )
        if not cv2.imwrite(str(output_image_path), resized_image, image_write_options):
            raise RuntimeError(f"Failed to write {output_image_path}")

        mask = cv2.resize(
            mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST,
        )
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to write {mask_path}")

        road_ratio = float(np.mean(mask == 1))
        ignore_ratio = float(np.mean(mask == 255))
        records.append((image_path.name, road_ratio, ignore_ratio))

    # Fixed, inspectable split spanning the original set and the newly collected
    # formal-camera frames. Keep nearby frames in training where possible.
    preferred_val = {
        "0005", "0012", "0017", "0022",
        "0028", "0038", "0050", "0065",
    }
    val = [name for name, _, _ in records if Path(name).stem in preferred_val]
    if len(val) != 8:
        val_indices = np.linspace(0, len(records) - 1, min(8, len(records)), dtype=int)
        val = [name for index, (name, _, _) in enumerate(records) if index in set(val_indices)]
    train = [name for name, _, _ in records if name not in set(val)]

    (split_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")

    print(f"Prepared {len(records)} samples: train={len(train)}, val={len(val)}")
    print(f"Output: {output}")
    print("Validation:", ", ".join(val))
    for name, road_ratio, ignore_ratio in records:
        print(f"  {name}: road={road_ratio:.3f}, ignore={ignore_ratio:.3f}")


if __name__ == "__main__":
    main()
