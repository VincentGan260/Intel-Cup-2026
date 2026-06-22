"""构建 yolo26n 微调数据集（BDD 留出图 → YOLO 单类 obstacle）。

为什么只用 BDD：本地 det 数据里 IDD(500)/DAWN(300) 全部在测试 manifest 内，
拿来训练即数据泄漏；只有 BDD 10000 张里除去 1500 张测试图，剩 8500 张可干净训练。
IDD/DAWN 因此保留为纯泛化测试（模型训练时完全没见过）。

类别无关：所有障碍物（人+各类车）映射为单类 0=obstacle，
排除 traffic sign/light/train 与 lane/area（与 eval_det.py 口径一致）。

输出自包含目录（可整体删除）：
    runs/finetune/data/{images,labels}/{train,val}/  + obstacle.yaml

运行：
    D:/Anaconda_envs/envs/intel/python.exe scripts/vision/build_yolo_finetune.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.dataset_paths import load_eval_config, resolve

# 障碍物保留类（其余 box2d 一律丢弃）
KEEP = {"car", "person", "truck", "bus", "bike", "rider", "motor"}
TWOWHEEL = {"bike", "rider", "motor"}   # 两轮车类（用于过采样判定）
VAL_FRACTION = 0.1          # 从留出图里再切一小份做训练监控 val（非测试集）
SEED = 1234                 # 固定划分，可复现


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="构建 yolo 微调数据集（BDD 留出图，单类）")
    ap.add_argument("--manifest", default="det_manifest.csv",
                    help="测试 manifest 文件名（test_plan 下）；其中 BDD 图从训练排除")
    ap.add_argument("--out", default="runs/finetune/data", help="输出数据集目录")
    ap.add_argument("--tw-oversample", type=int, default=0,
                    help="训练集中含两轮车的图额外复制几份（0=不过采样）")
    args = ap.parse_args()

    cfg = load_eval_config()
    ds_root = Path(cfg["dataset_root"])
    manifest = ds_root / "test_plan" / args.manifest

    # 1) 收集测试集 BDD 图片名（须从训练中排除）
    test_names = set()
    for r in csv.DictReader(manifest.open(encoding="utf-8")):
        if r["dataset"] == "bdd":
            test_names.add(Path(r["image"]).name)
    print(f"manifest={args.manifest}  测试集 BDD 图片数（排除）：{len(test_names)}")

    img_dir = ds_root / "det" / "bdd100k" / "images" / "100k" / "val"
    lbl_dir = ds_root / "det" / "bdd100k" / "labels" / "100k" / "val"
    all_imgs = sorted(img_dir.glob("*.jpg"))
    print(f"BDD val 总图片：{len(all_imgs)}")

    # 2) 留出训练池（确定性划分 train/val）
    pool = [p for p in all_imgs if p.name not in test_names]
    print(f"可训练留出图：{len(pool)}")

    out = resolve(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_train = n_val = n_box = n_empty = n_tw_extra = 0
    for idx, ip in enumerate(pool):
        # 用 hash 而非随机库，保证完全可复现的稳定划分
        split = "val" if (hash((SEED, ip.name)) % 1000) < VAL_FRACTION * 1000 else "train"
        jp = lbl_dir / (ip.stem + ".json")
        if not jp.is_file():
            continue
        d = json.load(jp.open(encoding="utf-8"))
        objs = (d.get("frames") or [{}])[0].get("objects", [])

        try:
            W, H = Image.open(ip).size
        except Exception:
            continue

        lines = []
        has_tw = False
        for o in objs:
            if o.get("category") not in KEEP or "box2d" not in o:
                continue
            if o["category"] in TWOWHEEL:
                has_tw = True
            b = o["box2d"]
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # 即使无框也复制图 + 空 txt（负样本，对压误报有益）
        txt = "\n".join(lines)
        shutil.copy2(ip, out / f"images/{split}" / ip.name)
        (out / f"labels/{split}" / (ip.stem + ".txt")).write_text(txt, encoding="utf-8")
        n_box += len(lines)
        if not lines:
            n_empty += 1
        if split == "train":
            n_train += 1
            # 两轮车过采样：仅训练集，含两轮车的图额外复制 N 份（带后缀新名）
            if has_tw and args.tw_oversample > 0:
                for k in range(1, args.tw_oversample + 1):
                    nm = f"{ip.stem}_tw{k}"
                    shutil.copy2(ip, out / "images/train" / f"{nm}{ip.suffix}")
                    (out / "labels/train" / f"{nm}.txt").write_text(txt, encoding="utf-8")
                    n_tw_extra += 1
        else:
            n_val += 1
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(pool)}")

    yaml = out / "obstacle.yaml"
    yaml.write_text(
        f"# 自动生成：BDD 留出图，单类障碍物\n"
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: obstacle\n",
        encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"train={n_train}(+两轮车过采样 {n_tw_extra})  val={n_val}  "
          f"共标注框={n_box}  无框(负样本)图={n_empty}")
    print(f"data.yaml: {yaml}")
    print("=" * 60)


if __name__ == "__main__":
    main()
