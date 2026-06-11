"""把现有的 FP32 OpenVINO IR 转成 FP16 IR（纯模型转换，不需要任何数据集）。

FP16 精度 ≈ FP32，主要在 iGPU 上提速,用于端侧延迟轮。转换源/目标路径都读自
configs/vision/eval.yaml 的 models 段。

运行：
    conda run --no-capture-output -n intel python scripts/vision/convert_fp16.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import openvino as ov

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.dataset_paths import load_eval_config, resolve


def convert_one(src_xml: Path, dst_xml: Path) -> str:
    if not src_xml.is_file():
        return f"[跳过] 源不存在：{src_xml}"
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    model = ov.Core().read_model(str(src_xml))
    ov.save_model(model, str(dst_xml), compress_to_fp16=True)
    size_mb = dst_xml.with_suffix(".bin").stat().st_size / 1e6
    return f"[OK]  {dst_xml}  ({size_mb:.1f} MB)"


def main() -> None:
    cfg = load_eval_config()
    m = cfg.get("models", {})
    det = m.get("detection", {})
    seg = m.get("segmentation", {})

    # (源 FP32, 目标 FP16) 对，缺哪个跳哪个
    pairs = [
        (det.get("fp32"), det.get("fp16")),
        (seg.get("pidnet_fp32"), seg.get("pidnet_fp16")),
        (seg.get("road_adas"), seg.get("road_adas_fp16")),
    ]

    print("=" * 60)
    print("FP32 → FP16 转换")
    print("=" * 60)
    for src, dst in pairs:
        if not src or not dst:
            continue
        print(convert_one(resolve(src), resolve(dst)))
    print("=" * 60)
    print("完成。可再跑 check_eval_paths.py 确认 FP16 已就位。")


if __name__ == "__main__":
    main()
