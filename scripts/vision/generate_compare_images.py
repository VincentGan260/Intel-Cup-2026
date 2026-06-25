"""生成模型对比图片脚本。

生成对比图片：
1. v2 模型 FP32 vs FP16 vs INT8 三种精度对比（10张）
2. 自动挑选 v1 和 v2 检测结果差异明显的图片

每张图片包含：原图 + FP32 + FP16 + INT8 检测结果
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_paths import load_eval_config, resolve

CONF_THRES = 0.25
IMGSZ = 640
COCO_OBSTACLE = {0, 1, 2, 3, 5, 7}

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

MODEL_CONFIG = {
    "base": "models/yolo26n_base_openvino_model/yolo26n.xml",
    "v2": "models/yolo26n_v2_openvino_model/best.xml",
    "v2_fp16": "models/yolo26n_v2_fp16_openvino_model/best.xml",
    "v2_int8": "models/yolo26n_v2_int8_openvino_model/best.xml",
}

COLORS = {
    "fp32": (255, 0, 0),
    "fp16": (0, 255, 0),
    "int8": (0, 0, 255),
    "base": (255, 0, 0),
    "v2": (0, 255, 0),
}


class YOLO26Detector:
    def __init__(self, model_path: str, device: str = "GPU"):
        self.core = ov.Core()
        self.model = self.core.compile_model(str(model_path), device)
        self.infer_request = self.model.create_infer_request()
        self.input_tensor = self.model.input(0)
        self.input_shape = self.input_tensor.shape
        self.input_h, self.input_w = self.input_shape[2], self.input_shape[3]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        scale = min(self.input_w / img_w, self.input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        input_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor, scale, pad_x, pad_y, img_w, img_h

    def _postprocess(self, output, scale, pad_x, pad_y, img_w, img_h, conf_thres=CONF_THRES):
        predictions = output[0]
        detections = []
        for det in predictions:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf >= conf_thres:
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(conf),
                    "cls": int(cls),
                    "name": CLASS_NAMES.get(int(cls), str(int(cls))),
                })
        return detections

    def predict(self, image_path: str):
        image = cv2.imread(str(image_path))
        if image is None:
            return [], None
        input_tensor, scale, pad_x, pad_y, img_w, img_h = self._preprocess(image)
        self.infer_request.infer([input_tensor])
        output = self.infer_request.get_output_tensor(0).data
        detections = self._postprocess(output, scale, pad_x, pad_y, img_w, img_h)
        return detections, image


def draw_detections(image: np.ndarray, detections: list, color: tuple, label: str):
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        name = det["name"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{name}: {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def create_quad_comparison(image: np.ndarray, dets_list: list, labels: list, colors: list):
    h, w = image.shape[:2]
    padding = 10
    n_cols = len(dets_list) + 1
    total_w = w * n_cols + padding * (n_cols - 1)
    result = np.full((h, total_w, 3), 255, dtype=np.uint8)

    result[:, :w] = image
    cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    for i, (dets, label, color) in enumerate(zip(dets_list, labels, colors)):
        img = draw_detections(image, dets, color, label)
        start = w * (i + 1) + padding * i
        end = start + w
        result[:, start:end] = img

    return result


def load_all_images(ds_root: Path):
    manifest = ds_root / "test_plan" / "det_manifest.csv"
    rows = []
    with manifest.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def select_diverse_images(ds_root: Path, count: int = 10):
    all_rows = load_all_images(ds_root)
    base_detector = YOLO26Detector(resolve(MODEL_CONFIG["base"]), "GPU")
    v2_detector = YOLO26Detector(resolve(MODEL_CONFIG["v2"]), "GPU")

    scored = []
    for row in all_rows:
        img_path = ds_root / row["image"]
        dets_base, _ = base_detector.predict(str(img_path))
        dets_v2, _ = v2_detector.predict(str(img_path))

        diff_count = abs(len(dets_base) - len(dets_v2))
        has_twowheel = row.get("has_twowheel") == "1"
        has_small = row.get("has_small") == "1"
        is_night = row.get("timeofday") == "night"

        score = diff_count * 10
        if has_twowheel:
            score += 5
        if has_small:
            score += 3
        if is_night:
            score += 3

        scored.append((score, diff_count, row))

    scored.sort(key=lambda x: -x[0])

    selected = []
    seen_scenes = set()
    for score, diff, row in scored:
        scene = row.get("scene", "")
        timeofday = row.get("timeofday", "")

        if len(selected) >= count:
            break

        if diff >= 3:
            selected.append(row)
            seen_scenes.add((scene, timeofday))
        elif len(selected) < count and (scene, timeofday) not in seen_scenes:
            selected.append(row)
            seen_scenes.add((scene, timeofday))

    return selected


def select_twowheel_diverse_images(ds_root: Path, count: int = 10):
    """专门选择两轮车检测差异大的图片"""
    all_rows = load_all_images(ds_root)
    base_detector = YOLO26Detector(resolve(MODEL_CONFIG["base"]), "GPU")
    v2_detector = YOLO26Detector(resolve(MODEL_CONFIG["v2"]), "GPU")

    scored = []
    for row in all_rows:
        img_path = ds_root / row["image"]
        dets_base, _ = base_detector.predict(str(img_path))
        dets_v2, _ = v2_detector.predict(str(img_path))

        # 计算两轮车检测数量
        twowheel_classes = {1, 3}  # bicycle, motorcycle
        base_tw = sum(1 for d in dets_base if d["cls"] in twowheel_classes)
        v2_tw = sum(1 for d in dets_v2 if d["cls"] in twowheel_classes)

        diff_tw = abs(base_tw - v2_tw)
        diff_total = abs(len(dets_base) - len(dets_v2))
        has_twowheel_gt = row.get("has_twowheel") == "1"
        is_night = row.get("timeofday") == "night"

        # 评分：两轮车差异权重最高
        score = diff_tw * 20  # 两轮车差异权重最高
        score += diff_total * 5  # 总体差异次之
        if has_twowheel_gt:
            score += 10  # GT有两轮车的额外加分
        if is_night:
            score += 5  # 夜间场景加分

        scored.append((score, diff_tw, base_tw, v2_tw, row))

    scored.sort(key=lambda x: -x[0])

    selected = []
    seen_scenes = set()
    for score, diff_tw, base_tw, v2_tw, row in scored:
        scene = row.get("scene", "")
        timeofday = row.get("timeofday", "")

        if len(selected) >= count:
            break

        # 优先选择两轮车差异 >= 1 的图片
        if diff_tw >= 1:
            selected.append(row)
            seen_scenes.add((scene, timeofday))
        elif len(selected) < count and (scene, timeofday) not in seen_scenes:
            selected.append(row)
            seen_scenes.add((scene, timeofday))

    return selected


def main():
    ap = argparse.ArgumentParser(description="生成模型对比图片")
    ap.add_argument("--device", default="GPU", help="推理设备")
    ap.add_argument("--count", type=int, default=10, help="对比图片数量")
    ap.add_argument("--output", default="runs/compare_images", help="输出目录")
    args = ap.parse_args()

    cfg = load_eval_config()
    ds_root = Path(cfg.get("dataset_root", "/home/intelcup/Intel-Cup-2026/datasets"))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对比1: v1 vs v2 (两轮车差异大)
    print(f"正在筛选两轮车检测差异大的图片...")
    twowheel_images = select_twowheel_diverse_images(ds_root, args.count)
    print(f"筛选出 {len(twowheel_images)} 张两轮车差异大的图片")

    base_detector = YOLO26Detector(resolve(MODEL_CONFIG["base"]), args.device)
    v2_detector = YOLO26Detector(resolve(MODEL_CONFIG["v2"]), args.device)

    print("\n=== 对比1: V1 (Base) vs V2 (两轮车差异大) ===")
    for i, row in enumerate(twowheel_images, 1):
        img_path = ds_root / row["image"]
        dets_base, image = base_detector.predict(str(img_path))
        dets_v2, _ = v2_detector.predict(str(img_path))

        # 计算两轮车检测数量
        twowheel_classes = {1, 3}
        base_tw = sum(1 for d in dets_base if d["cls"] in twowheel_classes)
        v2_tw = sum(1 for d in dets_v2 if d["cls"] in twowheel_classes)

        print(f"\n  图片 {i}: {row['image']}")
        print(f"    场景: {row.get('scene', '')} | 时间: {row.get('timeofday', '')} | 天气: {row.get('weather', '')}")
        print(f"    V1 检测: {len(dets_base)} 个目标 (两轮车: {base_tw})")
        print(f"    V2 检测: {len(dets_v2)} 个目标 (两轮车: {v2_tw})")
        print(f"    两轮车差异: {v2_tw - base_tw:+d}")

        comp_img = create_quad_comparison(
            image,
            [dets_base, dets_v2],
            ["V1 (Base)", "V2"],
            [COLORS["base"], COLORS["v2"]]
        )

        out_path = output_dir / f"v1_vs_v2_twowheel_{i:02d}.jpg"
        cv2.imwrite(str(out_path), comp_img)
        print(f"    保存: {out_path}")

    # 对比2: v2 FP32 vs FP16 vs INT8
    print(f"\n正在筛选差异明显的图片...")
    sample_images = select_diverse_images(ds_root, args.count)
    print(f"筛选出 {len(sample_images)} 张示例图片")

    v2_fp32_detector = YOLO26Detector(resolve(MODEL_CONFIG["v2"]), args.device)
    v2_fp16_detector = YOLO26Detector(resolve(MODEL_CONFIG["v2_fp16"]), args.device)
    v2_int8_detector = YOLO26Detector(resolve(MODEL_CONFIG["v2_int8"]), args.device)

    print("\n=== 对比2: V2 FP32 vs FP16 vs INT8 ===")
    for i, row in enumerate(sample_images, 1):
        img_path = ds_root / row["image"]
        dets_fp32, image = v2_fp32_detector.predict(str(img_path))
        dets_fp16, _ = v2_fp16_detector.predict(str(img_path))
        dets_int8, _ = v2_int8_detector.predict(str(img_path))

        print(f"\n  图片 {i}: {row['image']}")
        print(f"    场景: {row.get('scene', '')} | 时间: {row.get('timeofday', '')} | 天气: {row.get('weather', '')}")
        print(f"    FP32 检测: {len(dets_fp32)} 个目标")
        print(f"    FP16 检测: {len(dets_fp16)} 个目标")
        print(f"    INT8 检测: {len(dets_int8)} 个目标")

        comp_img = create_quad_comparison(
            image,
            [dets_fp32, dets_fp16, dets_int8],
            ["V2 FP32", "V2 FP16", "V2 INT8"],
            [COLORS["fp32"], COLORS["fp16"], COLORS["int8"]]
        )

        out_path = output_dir / f"v2_fp32_vs_fp16_vs_int8_{i:02d}.jpg"
        cv2.imwrite(str(out_path), comp_img)
        print(f"    保存: {out_path}")

    print(f"\n所有对比图片已保存至: {output_dir}")


if __name__ == "__main__":
    main()