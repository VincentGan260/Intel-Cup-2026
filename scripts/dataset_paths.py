"""测试脚本共用的路径配置加载器。

读取 configs/vision/datasets.yaml，若存在 datasets.local.yaml 则按顶层键覆盖，
从而让每个人在不动 git 的前提下用自己的路径。相对路径以项目根目录为基准。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "vision"
BASE_CONFIG = CONFIG_DIR / "datasets.yaml"
LOCAL_CONFIG = CONFIG_DIR / "datasets.local.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_path_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """加载路径配置：base ← local ← 显式 config_path（后者覆盖前者，按顶层键）。"""
    cfg = _read_yaml(BASE_CONFIG)
    cfg.update(_read_yaml(LOCAL_CONFIG))
    if config_path is not None:
        cfg.update(_read_yaml(Path(config_path)))
    return cfg


def resolve(path: str | Path) -> Path:
    """相对路径按项目根目录展开；绝对路径原样返回。"""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
