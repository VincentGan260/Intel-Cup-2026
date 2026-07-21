"""Show or switch the production segmentation model with one number.

Profiles:
    1 = competition white-tile model
    2 = original road-segmentation-adas model
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.profile_selector import select_segmentation_config


DEFAULT_PIPELINE = PROJECT_ROOT / "configs" / "vision" / "vision_pipeline.yaml"
PROFILE_NAMES = {1: "white-tile", 2: "road-adas"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch the production segmentation profile")
    parser.add_argument("model", nargs="?", type=int, choices=sorted(PROFILE_NAMES))
    parser.add_argument("--config", type=Path, default=DEFAULT_PIPELINE)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) or {}

    if args.model is not None:
        updated, count = re.subn(
            r"(?m)^(segmentation_model:\s*)\d+\s*$",
            rf"\g<1>{args.model}",
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"segmentation_model not found in {config_path}")
        config = yaml.safe_load(updated) or {}
        selected_path = select_segmentation_config(config)
        resolved = Path(selected_path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        if not resolved.is_file():
            raise FileNotFoundError(f"Selected config not found: {resolved}")
        config_path.write_text(updated, encoding="utf-8")
        print(f"switched to {args.model}: {PROFILE_NAMES[args.model]}")

    selected = int(config.get("segmentation_model", 1))
    selected_path = select_segmentation_config(config)
    print(f"active profile: {selected} ({PROFILE_NAMES.get(selected, 'unknown')})")
    print(f"config: {selected_path}")


if __name__ == "__main__":
    main()
