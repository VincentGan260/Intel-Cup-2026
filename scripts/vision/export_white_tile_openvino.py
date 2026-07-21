"""Export the fine-tuned white-tile DeepLabV3 checkpoint to OpenVINO FP16 IR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openvino as ov
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vision.train_white_tile_deeplab import build_model


class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["out"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "vision" / "white_tile_deeplab" / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "openvino" / "white-tile-road-fp16",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    height, width = map(int, checkpoint.get("input_size", [512, 896]))
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    wrapper = LogitsOnly(model.eval())
    example = torch.zeros(1, 3, height, width, dtype=torch.float32)

    with torch.no_grad():
        expected_shape = tuple(wrapper(example).shape)
    if expected_shape[0] != 1 or expected_shape[1] != 2:
        raise RuntimeError(f"Unexpected output shape: {expected_shape}")

    ov_model = ov.convert_model(wrapper, example_input=example)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = args.output_dir / "white-tile-road.xml"
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    actual_shape = tuple(compiled(example.numpy())[compiled.output(0)].shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"PyTorch/OpenVINO shape mismatch: {expected_shape} vs {actual_shape}"
        )
    print(f"OpenVINO FP16 model: {xml_path}")
    print(f"Input: {(1, 3, height, width)}, output: {actual_shape}")


if __name__ == "__main__":
    main()
