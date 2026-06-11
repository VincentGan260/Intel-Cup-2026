"""测试前自检：读 configs/vision/eval.yaml，逐项检查路径/标注/模型是否就位。

只读不改，告诉你「哪些已准备好、哪些还缺」，方便对照填路径。
运行：
    conda run -n intel python scripts/vision/check_eval_paths.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码/报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_paths import load_eval_config, resolve

OK = "[OK]"
MISS = "[缺]"


def check(label: str, rel_path, kind: str = "dir", count_glob: str | None = None) -> bool:
    """检查一个路径是否存在；kind=dir/file。可选统计目录下文件数。"""
    if rel_path is None:
        print(f"  {MISS} {label}: (未配置)")
        return False
    p = resolve(rel_path)
    exists = p.is_dir() if kind == "dir" else p.is_file()
    tag = OK if exists else MISS
    extra = ""
    if exists and count_glob:
        n = len(list(p.rglob(count_glob)))
        extra = f"  ({n} 个 {count_glob})"
    print(f"  {tag} {label}: {p}{extra}")
    return exists


def main() -> None:
    cfg = load_eval_config()
    if not cfg:
        print("错误：未找到/读不到 configs/vision/eval.yaml")
        return

    missing = 0

    print("=" * 70)
    print("视觉测试 · 路径自检")
    print("=" * 70)

    # --- 检测 ---
    det = cfg.get("detection", {})
    print("\n[检测精度] 需要：BDD 图片 + 逐图 JSON 标注")
    missing += not check("图片目录 images_dir", det.get("images_dir"), "dir", "*.jpg")
    missing += not check("标注目录 labels_dir", det.get("labels_dir"), "dir", "*.json")

    # --- 分割 ---
    print("\n[分割精度] 需要：各数据集 图片 + road 语义标注")
    for ds in cfg.get("segmentation", {}).get("datasets", []):
        name = ds.get("name", "?")
        print(f"  · 数据集 {name}（road_value={ds.get('road_value')}, suffix={ds.get('label_suffix')}）")
        missing += not check(f"  {name} images", ds.get("images_dir"), "dir")
        missing += not check(f"  {name} labels", ds.get("labels_dir"), "dir")

    # --- 模型 ---
    print("\n[模型] FP32 现在要有；FP16/INT8 由后续转换/量化产出（暂缺正常）")
    mdl = cfg.get("models", {})
    det_m = mdl.get("detection", {})
    check("检测 PyTorch", det_m.get("pytorch"), "file")
    check("检测 OpenVINO FP32", det_m.get("fp32"), "file")
    check("检测 INT8（待量化）", det_m.get("int8"), "file")
    seg_m = mdl.get("segmentation", {})
    check("分割 PIDNet FP32", seg_m.get("pidnet_fp32"), "file")
    check("分割 PIDNet FP16（待转换）", seg_m.get("pidnet_fp16"), "file")
    check("分割 PIDNet INT8（待量化）", seg_m.get("pidnet_int8"), "file")
    check("分割 road-adas", seg_m.get("road_adas"), "file")

    # --- 量化校准 ---
    print("\n[量化] 校准图（无需标注）")
    check("校准图目录", cfg.get("quantization", {}).get("calib_images_dir"), "dir", "*.jpg")

    print("\n" + "=" * 70)
    if missing == 0:
        print("精度评测所需的图片/标注都就位，可以开跑。")
    else:
        print(f"还有 {missing} 项数据/标注缺失（上面标 {MISS} 的）。")
        print("→ 去 configs/vision/eval.yaml 改成你本地的真实路径，或建 eval.local.yaml 覆盖。")
    print("（FP16/INT8 模型暂缺是正常的，那是后续步骤产出。）")
    print("=" * 70)


if __name__ == "__main__":
    main()
