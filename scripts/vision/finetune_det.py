"""微调 yolo26n 为单类「障碍物」检测器（BDD 留出图，4060 GPU）。

数据由 build_yolo_finetune.py 生成（runs/finetune/data/obstacle.yaml）。
评测口径一致：imgsz=640、单类。导出后用 eval_det.py 重测对比。

需用带 CUDA 的环境（本机 pytorch env）：
    D:/Anaconda_envs/envs/pytorch/python.exe scripts/vision/finetune_det.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    import argparse
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description="微调 yolo26n 单类障碍物")
    ap.add_argument("--data", default="runs/finetune/data/obstacle.yaml",
                    help="data.yaml 路径（相对项目根）")
    ap.add_argument("--name", default="yolo26n_obstacle", help="训练 run 名")
    args = ap.parse_args()

    data = PROJECT_ROOT / args.data
    run_dir = PROJECT_ROOT / "runs" / "finetune" / args.name
    last = run_dir / "weights" / "last.pt"

    # 断点续训：若该 run 已有 last.pt（上次中途停了），从断点接着跑；否则从预训练权重起步
    if last.is_file():
        print(f"检测到断点，续训：{last}")
        model = YOLO(str(last))
        model.train(resume=True)
        print("\n续训完成。")
        return

    model = YOLO(str(PROJECT_ROOT / "yolo26n.pt"))
    model.train(
        data=str(data),
        epochs=60,
        patience=15,            # 15 个 epoch 无提升早停
        imgsz=640,              # 与 eval_det.py 一致
        batch=16,               # 4060 8GB，nano 足够
        device=0,
        workers=6,
        cache=False,
        project=str(PROJECT_ROOT / "runs" / "finetune"),
        name=args.name,
        exist_ok=True,
        seed=0,
        # 单类检测，关掉无意义的分类相关增强、保留几何/mosaic（利于小目标）
        plots=True,
    )
    print("\n训练完成。best.pt:", run_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
