"""把微调 v2 检测模型导成可部署的 OpenVINO IR（FP32 + FP16）到 models/。

为什么单独写：检测模型用 ultralytics 导出会带上 metadata.yaml（类别/NMS 配置），
管线能与基础模型一样加载；FP16 用 ultralytics `half=True`，精度≈FP32、iGPU 上更快（端侧延迟轮用）。
分割模型的 FP16 走 convert_fp16.py（IR→IR）。

产出：
    models/yolo26n_v2_openvino_model/        ← FP32（管线默认用）
    models/yolo26n_v2_fp16_openvino_model/   ← FP16（端侧延迟轮）

运行：
    D:/Anaconda_envs/envs/intel/python.exe scripts/vision/export_finetuned.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def export_one(weights: Path, half: bool, dst_dir: Path, imgsz: int) -> None:
    """ultralytics 导出 OpenVINO 到 dst_dir（半精度由 half 决定）。"""
    from ultralytics import YOLO

    out = Path(YOLO(str(weights)).export(format="openvino", imgsz=imgsz, half=half, dynamic=False))
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.move(str(out), str(dst_dir))
    bins = list(dst_dir.glob("*.bin"))
    mb = bins[0].stat().st_size / 1e6 if bins else 0.0
    print(f"  [{'FP16' if half else 'FP32'}] → {dst_dir}  ({mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="导出微调 v2 检测模型为 OpenVINO FP32+FP16")
    ap.add_argument("--weights", default="runs/finetune/yolo26n_obstacle_v2/weights/best.pt",
                    help="权重路径（相对项目根）。基础 COCO 基线用 yolo26n.pt")
    ap.add_argument("--tag", default="v2",
                    help="输出目录命名 models/yolo26n_{tag}_openvino_model（基线用 base）")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    weights = PROJECT_ROOT / args.weights
    if not weights.is_file():
        print(f"错误：找不到权重 {weights}"); return

    models = PROJECT_ROOT / "models"
    print("=" * 60)
    print(f"导出检测模型：{weights.name}  tag={args.tag}（FP32+FP16，同一 recipe）")
    print("=" * 60)
    export_one(weights, False, models / f"yolo26n_{args.tag}_openvino_model", args.imgsz)
    export_one(weights, True, models / f"yolo26n_{args.tag}_fp16_openvino_model", args.imgsz)
    print("=" * 60)
    print("完成。detection.yaml 的 model_path 指向 models/yolo26n_v2_openvino_model 即用 FP32；")
    print("FP16 模型在 models/yolo26n_v2_fp16_openvino_model（端侧延迟轮 iGPU 用）。")


if __name__ == "__main__":
    main()
