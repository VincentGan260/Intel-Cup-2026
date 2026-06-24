"""YOLO26n INT8 量化（固定尺寸，解决延迟抖动）。

默认量化**部署用的微调 v2 检测模型**（runs/finetune/yolo26n_obstacle_v2/weights/best.pt）；
若要量化原始 COCO 基线，把 MODEL_PATH 改回 yolo26n.pt 即可。

校准数据从 eval.local.yaml 的 dataset_root 读真实驾驶图（BDD/IDD/DAWN det，覆盖含雾天），
递归查找 + 采样数下限断言，避免「目录写错→采 0 张→校准退化」。

运行（需 nncf）：
    D:/Anaconda_envs/envs/intel/python.exe quantize_yolo26n_v2.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import nncf
from openvino import Core, serialize
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.dataset_paths import load_eval_config, resolve  # noqa: E402

# ========== 配置（可被命令行覆盖）==========
# 默认量化部署模型（微调 v2）；量化 COCO 基线用 --weights yolo26n.pt --out yolo26n_base_int8
_ap = argparse.ArgumentParser(description="YOLO26n INT8 量化（nncf MIXED，真实图校准）")
_ap.add_argument("--weights", default="runs/finetune/yolo26n_obstacle_v2/weights/best.pt",
                 help="权重路径（相对项目根）。基础 COCO 基线用 yolo26n.pt")
_ap.add_argument("--out", default="yolo26n_obstacle_v2_int8", help="输出 xml/bin 名（不含扩展名）")
_args, _ = _ap.parse_known_args()

MODEL_PATH = PROJECT_ROOT / _args.weights
OUTPUT_NAME = _args.out     # 输出 xml/bin 名
CALIB_SAMPLES = 1000
IMG_SIZE = 640
MIN_CALIB = 50            # 校准图下限，低于此直接报错（防静默退化）
# ========== 配置结束 ==========


def calib_dirs_from_config() -> list[Path]:
    """从 dataset_root 推导真实校准目录（det 驾驶图）。"""
    cfg = load_eval_config()
    root = cfg.get("dataset_root")
    if not root:
        raise RuntimeError("eval.local.yaml 缺 dataset_root，无法定位校准数据")
    root = Path(root)
    return [
        root / "det" / "bdd100k" / "images" / "100k" / "val",
        root / "det" / "idd" / "images" / "val",
        root / "det" / "dawn" / "images",
    ]


def get_calibration_images(num_samples: int) -> list[Path]:
    import random
    all_images = []
    for d in calib_dirs_from_config():
        if d.exists():
            imgs = list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
            all_images.extend(imgs)
            print(f"  {d}: {len(imgs)} 张")
        else:
            print(f"  {d}: 目录不存在，跳过")
    if len(all_images) < MIN_CALIB:
        raise FileNotFoundError(
            f"校准图不足 {MIN_CALIB} 张（当前 {len(all_images)}）——检查 dataset_root 与 det 目录")
    selected = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"\n总计 {len(all_images)} 张，采样 {len(selected)} 张")
    return selected


def make_dataset(image_paths, input_size, xml_path):
    core = Core()
    input_name = core.read_model(str(xml_path)).input(0).get_any_name()

    def preprocess_fn(p):
        img = cv2.imread(str(p))
        if img is None:
            return None
        img = cv2.resize(img, (input_size, input_size))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return {input_name: np.expand_dims(img, 0)}

    return nncf.Dataset(image_paths, preprocess_fn)


def main():
    print("=" * 60)
    print(f"YOLO26n INT8 量化  模型={MODEL_PATH.name}")
    print("=" * 60)

    if not MODEL_PATH.is_file():
        print(f"错误：找不到权重 {MODEL_PATH}"); return

    print("\n[1/4] 导出 FP32 OpenVINO（固定尺寸）...")
    export_dir = Path(YOLO(str(MODEL_PATH)).export(format="openvino", imgsz=IMG_SIZE, half=False))
    xml_candidates = list(export_dir.glob("*.xml"))
    if not xml_candidates:
        print(f"错误：导出目录无 xml: {export_dir}"); return
    xml_path = xml_candidates[0]
    print(f"      FP32: {xml_path}")

    print("\n[2/4] 收集校准数据...")
    calib_images = get_calibration_images(CALIB_SAMPLES)

    print("\n[3/4] 创建校准数据集...")
    calib_dataset = make_dataset(calib_images, IMG_SIZE, xml_path)

    print(f"\n[4/4] INT8 量化（MIXED 预设，{len(calib_images)} 张校准，{IMG_SIZE}x{IMG_SIZE}）...")
    core = Core()
    ov_model = core.read_model(str(xml_path))
    quantized = nncf.quantize(
        ov_model, calib_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calib_images),
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            overflow_fix=nncf.OverflowFix.FIRST_LAYER),
    )

    output_xml = PROJECT_ROOT / f"{OUTPUT_NAME}.xml"
    serialize(quantized, str(output_xml))
    fp32_mb = xml_path.with_suffix(".bin").stat().st_size / 1024 / 1024
    int8_mb = output_xml.with_suffix(".bin").stat().st_size / 1024 / 1024

    print("\n" + "=" * 60)
    print(f"量化完成: {output_xml}")
    print(f"FP32={fp32_mb:.2f}MB  INT8={int8_mb:.2f}MB  压缩 {fp32_mb / int8_mb:.2f}x")
    print("下一步：用 eval_det.py --weights <int8目录> 出 FP32 vs INT8 精度对比")
    print("=" * 60)


if __name__ == "__main__":
    main()
