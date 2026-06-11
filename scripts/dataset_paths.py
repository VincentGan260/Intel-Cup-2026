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
EVAL_CONFIG = CONFIG_DIR / "eval.yaml"
EVAL_LOCAL = CONFIG_DIR / "eval.local.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并：override 里的嵌套 dict 只覆盖对应子键，其余保留 base。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_path_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """加载 datasets.yaml：base ← local ← 显式 config_path（按顶层键覆盖）。"""
    cfg = _read_yaml(BASE_CONFIG)
    cfg.update(_read_yaml(LOCAL_CONFIG))
    if config_path is not None:
        cfg.update(_read_yaml(Path(config_path)))
    return cfg


def load_eval_config() -> dict[str, Any]:
    """加载 eval.yaml：base ← eval.local.yaml（嵌套深度覆盖）。"""
    return _deep_merge(_read_yaml(EVAL_CONFIG), _read_yaml(EVAL_LOCAL))


def resolve(path: str | Path) -> Path:
    """相对路径按项目根目录展开；绝对路径原样返回。"""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
