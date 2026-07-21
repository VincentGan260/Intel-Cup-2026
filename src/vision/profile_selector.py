"""Select a segmentation config from the numeric profile in the pipeline YAML."""

from __future__ import annotations

from collections.abc import Mapping


DEFAULT_SEGMENTATION_CONFIG = "configs/vision/segmentation_openvino.yaml"


def select_segmentation_config(
    pipeline_config: Mapping,
    default: str = DEFAULT_SEGMENTATION_CONFIG,
) -> str:
    """Return the selected child config while preserving legacy YAML support."""
    profiles = pipeline_config.get("segmentation_profiles")
    if not profiles:
        return str(pipeline_config.get("segmentation_config", default))

    normalized = {str(key): str(value) for key, value in dict(profiles).items()}
    selected = str(pipeline_config.get("segmentation_model", "1"))
    if selected not in normalized:
        choices = ", ".join(sorted(normalized))
        raise ValueError(
            f"Unknown segmentation_model={selected}; available values: {choices}"
        )
    return normalized[selected]
