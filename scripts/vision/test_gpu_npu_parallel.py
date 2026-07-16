"""GPU + NPU OpenVINO 并行推理测试脚本

展示语义分割(FP16 road-adas)和目标检测(INT8 YOLO26n v2)的并行推理过程：
- road-adas-fp16 → GPU
- yolo26n_v2_int8 → NPU
- 异步并发执行，体现并行加速效果

测试集：
- 默认使用 datasets/det/dawn/images/ 目录（同时用于检测和分割测试）
- 支持 .jpg, .png, .jpeg 格式

运行：
    python scripts/vision/test_gpu_npu_parallel.py
    python scripts/vision/test_gpu_npu_parallel.py --test_dir datasets/det/dawn/images --warmup 10 --iters 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import openvino as ov

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_image_bgr(path: Path) -> np.ndarray:
    """读取BGR图像"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def logits_chw_to_label_map(logits: np.ndarray) -> np.ndarray:
    """将CHW格式的logits转换为标签图"""
    return np.argmax(logits, axis=0).astype(np.uint8)


class SegmentationResult:
    """分割结果"""
    def __init__(self, road_mask: np.ndarray):
        self.road_mask = road_mask


class DetectionResult:
    """检测结果"""
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[float, float, float, float]):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox


def build_segmentation_result_from_label_map(label_small: np.ndarray, wh: Tuple[int, int], road_class_index: int) -> SegmentationResult:
    """从标签图构建分割结果"""
    mask = (label_small == road_class_index).astype(np.uint8)
    mask = cv2.resize(mask, wh, interpolation=cv2.INTER_NEAREST)
    return SegmentationResult(road_mask=mask)


class RoadAdasSegmenter:
    """FP16 road-adas 语义分割模型"""

    def __init__(self, xml_path: Path, device: str, compile_config: Dict[str, Any] | None = None):
        self.xml_path = xml_path
        self.device = device
        self.input_height = 1024
        self.input_width = 1024
        self.road_class_index = 1

        core = ov.Core()
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, device, compile_config or {})
        self.input_name = self.compiled.input(0).get_any_name()
        self.infer_request = self.compiled.create_infer_request()

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image_bgr,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized.transpose(2, 0, 1)[None].astype(np.float32)

    def infer_sync(self, image_bgr: np.ndarray) -> SegmentationResult:
        tensor = self.preprocess(image_bgr)
        outputs = self.compiled({self.input_name: tensor})
        out_tensor = next(iter(outputs.values()))
        logits = np.array(out_tensor)[0]
        label_small = logits_chw_to_label_map(logits)
        wh = (image_bgr.shape[1], image_bgr.shape[0])
        return build_segmentation_result_from_label_map(
            label_small, wh, self.road_class_index
        )

    def infer_async_start(self, image_bgr: np.ndarray) -> None:
        tensor = self.preprocess(image_bgr)
        self.infer_request.set_tensor(self.input_name, ov.Tensor(tensor))
        self.infer_request.start_async()

    def infer_async_wait(self, image_bgr: np.ndarray) -> SegmentationResult:
        self.infer_request.wait()
        out_tensor = self.infer_request.get_output_tensor()
        logits = out_tensor.data
        logits = np.array(logits)
        if logits.ndim == 4:
            logits = logits[0]
        label_small = logits_chw_to_label_map(logits)
        wh = (image_bgr.shape[1], image_bgr.shape[0])
        return build_segmentation_result_from_label_map(
            label_small, wh, self.road_class_index
        )


class YOLO26nInt8Detector:
    """INT8 YOLO26n v2 目标检测模型"""

    def __init__(self, xml_path: Path, device: str, compile_config: Dict[str, Any] | None = None):
        self.xml_path = xml_path
        self.device = device
        self.image_size = 640
        self.confidence = 0.25

        core = ov.Core()
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, device, compile_config or {})
        self.input_name = self.compiled.input(0).get_any_name()
        self.infer_request = self.compiled.create_infer_request()

        self.names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "traffic sign", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
            49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
            59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
            69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
            74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
            79: "toothbrush"
        }

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        img_h, img_w = image.shape[:2]
        scale = min(self.image_size / img_w, self.image_size / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        pad_x = (self.image_size - new_w) // 2
        pad_y = (self.image_size - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        input_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, 0)

        return input_tensor, scale, scale, pad_x, pad_y

    def nms(self, boxes: np.ndarray, confidences: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = confidences.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def postprocess(self, output: np.ndarray, scale: float, pad_x: int, pad_y: int, img_w: int, img_h: int) -> List[DetectionResult]:
        detections = []
        output = output[0]

        if output.ndim == 2 and output.shape[-1] == 6:
            pass
        else:
            return detections

        x_center, y_center, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
        confidences = output[:, 4]
        class_ids = output[:, 5].astype(int)

        mask = confidences > self.confidence
        if not np.any(mask):
            return detections

        x_center = x_center[mask]
        y_center = y_center[mask]
        w = w[mask]
        h = h[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        x1 = (x_center - w / 2 - pad_x) / scale
        y1 = (y_center - h / 2 - pad_y) / scale
        x2 = (x_center + w / 2 - pad_x) / scale
        y2 = (y_center + h / 2 - pad_y) / scale

        x1 = np.clip(x1, 0, img_w - 1)
        y1 = np.clip(y1, 0, img_h - 1)
        x2 = np.clip(x2, 0, img_w - 1)
        y2 = np.clip(y2, 0, img_h - 1)

        for i in range(len(class_ids)):
            class_id = int(class_ids[i])
            confidence = float(confidences[i])
            if confidence < self.confidence:
                continue

            class_name = self.names.get(class_id, f"cls_{class_id}")

            detections.append(DetectionResult(
                class_name=class_name,
                confidence=confidence,
                bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            ))

        return detections

    def infer_sync(self, image_bgr: np.ndarray) -> List[DetectionResult]:
        input_tensor, scale, _, pad_x, pad_y = self.preprocess(image_bgr)
        outputs = self.compiled({self.input_name: input_tensor})
        out_tensor = next(iter(outputs.values()))
        output = np.array(out_tensor)
        img_h, img_w = image_bgr.shape[:2]
        return self.postprocess(output, scale, pad_x, pad_y, img_w, img_h)

    def infer_async_start(self, image_bgr: np.ndarray) -> None:
        input_tensor, _, _, _, _ = self.preprocess(image_bgr)
        self.infer_request.set_tensor(self.input_name, ov.Tensor(input_tensor))
        self.infer_request.start_async()

    def infer_async_wait(self, image_bgr: np.ndarray) -> List[DetectionResult]:
        self.infer_request.wait()
        out_tensor = self.infer_request.get_output_tensor()
        output = np.array(out_tensor.data)
        img_h, img_w = image_bgr.shape[:2]
        _, scale, _, pad_x, pad_y = self.preprocess(image_bgr)
        return self.postprocess(output, scale, pad_x, pad_y, img_w, img_h)


def find_test_images(test_dir: Path) -> List[Path]:
    """查找测试目录下的所有图片"""
    if not test_dir.exists():
        return []

    extensions = [".jpg", ".png", ".jpeg"]
    images = []
    for ext in extensions:
        images.extend(test_dir.glob(f"*{ext}"))
        images.extend(test_dir.glob(f"**/*{ext}"))

    return sorted(images)


def benchmark_sequential(
    segmenter: RoadAdasSegmenter,
    detector: YOLO26nInt8Detector,
    images: List[Path],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    """顺序执行（串行）基准测试"""
    print("\n[串行模式] 先跑分割，再跑检测")
    print("  时间线: [分割] → [检测]（等待分割完成后才开始检测）")

    for _ in range(warmup):
        img = read_image_bgr(images[0])
        segmenter.infer_sync(img)
        detector.infer_sync(img)

    times = []
    seg_times = []
    det_times = []

    for i in range(min(iters, len(images))):
        img = read_image_bgr(images[i])

        t0 = time.perf_counter()

        t_seg_start = time.perf_counter()
        segmenter.infer_sync(img)
        seg_times.append((time.perf_counter() - t_seg_start) * 1000)

        t_det_start = time.perf_counter()
        detector.infer_sync(img)
        det_times.append((time.perf_counter() - t_det_start) * 1000)

        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mode": "sequential",
        "total_mean_ms": round(np.mean(times), 2),
        "total_median_ms": round(np.median(times), 2),
        "total_p95_ms": round(np.percentile(times, 95), 2),
        "total_fps": round(1000.0 / np.mean(times), 2),
        "seg_mean_ms": round(np.mean(seg_times), 2),
        "det_mean_ms": round(np.mean(det_times), 2),
    }


def benchmark_parallel(
    segmenter: RoadAdasSegmenter,
    detector: YOLO26nInt8Detector,
    images: List[Path],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    """并行执行（异步并发）基准测试"""
    print("\n[并行模式] GPU(分割) + NPU(检测) 异步并发执行")
    print("  时间线: [启动分割] → [启动检测] → [等待分割完成] → [等待检测完成]")
    print("  并行度: 两个模型同时在不同设备上推理，重叠时间即为并行收益")

    for _ in range(warmup):
        img = read_image_bgr(images[0])
        segmenter.infer_async_start(img)
        detector.infer_async_start(img)
        segmenter.infer_async_wait(img)
        detector.infer_async_wait(img)

    times = []
    seg_times = []
    det_times = []
    overlap_times = []

    for i in range(min(iters, len(images))):
        img = read_image_bgr(images[i])

        t0 = time.perf_counter()

        t_seg_start = time.perf_counter()
        segmenter.infer_async_start(img)
        t_det_start = time.perf_counter()
        detector.infer_async_start(img)

        t_seg_wait_start = time.perf_counter()
        segmenter.infer_async_wait(img)
        t_seg_end = time.perf_counter()

        t_det_wait_start = time.perf_counter()
        detector.infer_async_wait(img)
        t_det_end = time.perf_counter()

        total_time = (t_det_end - t0) * 1000
        seg_time = (t_seg_end - t_seg_start) * 1000
        det_time = (t_det_end - t_det_start) * 1000
        overlap_time = max(0, (t_seg_end - t_det_start) * 1000)

        times.append(total_time)
        seg_times.append(seg_time)
        det_times.append(det_time)
        overlap_times.append(overlap_time)

    return {
        "mode": "parallel",
        "total_mean_ms": round(np.mean(times), 2),
        "total_median_ms": round(np.median(times), 2),
        "total_p95_ms": round(np.percentile(times, 95), 2),
        "total_fps": round(1000.0 / np.mean(times), 2),
        "seg_mean_ms": round(np.mean(seg_times), 2),
        "det_mean_ms": round(np.mean(det_times), 2),
        "overlap_mean_ms": round(np.mean(overlap_times), 2),
    }


def print_results(sequential: Dict, parallel: Dict, devices: List[str]) -> None:
    """打印测试结果对比"""
    print("\n" + "=" * 80)
    print("GPU + NPU OpenVINO 并行推理测试结果")
    print("=" * 80)

    print(f"\n模型配置:")
    print(f"  语义分割: road-adas-fp16 (FP16) @ {devices[0]}")
    print(f"  目标检测: yolo26n_v2_int8 (INT8) @ {devices[1]}")

    print("\n" + "-" * 90)
    print(f"| {'模式':<10} | {'总延迟(ms)':>12} | {'分割耗时(ms)':>14} | {'检测耗时(ms)':>14} | {'重叠时间(ms)':>14} | {'FPS':>8} |")
    print("-" * 90)
    print(f"| {'串行':<10} | {sequential['total_mean_ms']:>12.2f} | {sequential['seg_mean_ms']:>14.2f} | {sequential['det_mean_ms']:>14.2f} | {'-':>14} | {sequential['total_fps']:>8.2f} |")
    print(f"| {'并行':<10} | {parallel['total_mean_ms']:>12.2f} | {parallel['seg_mean_ms']:>14.2f} | {parallel['det_mean_ms']:>14.2f} | {parallel.get('overlap_mean_ms', '-'):>14} | {parallel['total_fps']:>8.2f} |")
    print("-" * 90)

    speedup = sequential['total_mean_ms'] / parallel['total_mean_ms'] if parallel['total_mean_ms'] > 0 else 0
    fps_improve = parallel['total_fps'] / sequential['total_fps'] if sequential['total_fps'] > 0 else 0
    overlap = parallel.get('overlap_mean_ms', 0)

    print(f"\n并行加速效果分析:")
    print(f"  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │ 串行总耗时 = 分割耗时 + 检测耗时                              │")
    print(f"  │  = {sequential['seg_mean_ms']:.1f}ms + {sequential['det_mean_ms']:.1f}ms = {sequential['total_mean_ms']:.1f}ms     │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │ 并行总耗时 = max(分割耗时, 检测耗时) ≈ 并行重叠后实际耗时        │")
    print(f"  │  = {parallel['total_mean_ms']:.1f}ms（两个模型同时运行）       │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │ 并行重叠时间: {overlap:.1f}ms（两个模型同时推理的时间）          │")
    print(f"  │ 延迟降低: {speedup:.2f}x                                        │")
    print(f"  │ FPS提升:  {fps_improve:.2f}x                                     │")
    print(f"  │ 理论最大加速: ~2.0x（两个模型完全并行）                           │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n时间线示意图:")
    print(f"  串行模式:")
    print(f"    GPU:  ████████████████████ (分割 {sequential['seg_mean_ms']:.1f}ms)")
    print(f"    NPU:  ────────────────────█████████████████████ (检测 {sequential['det_mean_ms']:.1f}ms)")
    print(f"    总耗时: {sequential['total_mean_ms']:.1f}ms")
    print(f"\n  并行模式:")
    print(f"    GPU:  ████████████████████ (分割 {parallel['seg_mean_ms']:.1f}ms)")
    print(f"    NPU:  ██████████████████████████████ (检测 {parallel['det_mean_ms']:.1f}ms)")
    print(f"           ↑同时启动                    ↑重叠运行")
    print(f"    总耗时: {parallel['total_mean_ms']:.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU + NPU OpenVINO 并行推理测试")
    parser.add_argument("--test_dir", type=str, default="datasets/det/dawn/images", help="测试图片目录")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--iters", type=int, default=50, help="测试次数")
    parser.add_argument("--seg_device", type=str, default="GPU", help="分割模型设备 (CPU/GPU/NPU/AUTO)")
    parser.add_argument("--det_device", type=str, default="NPU", help="检测模型设备 (CPU/GPU/NPU/AUTO)")
    parser.add_argument("--all_configs", action="store_true", help="自动测试所有设备配置组合")
    args = parser.parse_args()

    test_dir = PROJECT_ROOT / args.test_dir

    seg_model_path = PROJECT_ROOT / "models" / "openvino" / "road-adas-fp16" / "road-segmentation-adas-0001.xml"
    det_model_path = PROJECT_ROOT / "models" / "yolo26n_v2_int8_openvino_model" / "best.xml"

    print("=" * 80)
    print("GPU + NPU OpenVINO 并行推理测试")
    print("=" * 80)

    core = ov.Core()
    available_devices = core.available_devices
    print(f"\nOpenVINO 可用设备: {available_devices}")

    print(f"\n模型路径:")
    print(f"  分割模型 (FP16): {seg_model_path}")
    print(f"  检测模型 (INT8): {det_model_path}")

    if not seg_model_path.exists():
        print(f"\n错误: 分割模型文件不存在: {seg_model_path}")
        return

    if not det_model_path.exists():
        print(f"\n错误: 检测模型文件不存在: {det_model_path}")
        return

    images = find_test_images(test_dir)
    if not images:
        print(f"\n警告: 测试目录 {test_dir} 下未找到图片")
        print("请将测试图片放入该目录，支持 .jpg, .png, .jpeg 格式")
        return

    print(f"\n测试图片: {len(images)} 张")

    if args.all_configs:
        configs = [
            ("GPU", "GPU", "都用GPU"),
            ("NPU", "NPU", "都用NPU"),
            ("GPU", "NPU", "GPU+NPU"),
            ("NPU", "GPU", "NPU+GPU"),
        ]
        print(f"\n自动测试所有配置组合:")
        for i, (sd, dd, desc) in enumerate(configs):
            print(f"  {i+1}. {desc}: 分割@{sd}, 检测@{dd}")
    else:
        configs = [(args.seg_device, args.det_device, f"{args.seg_device}+{args.det_device}")]

    all_results = {}

    for seg_device, det_device, desc in configs:
        print(f"\n{'='*80}")
        print(f"配置: {desc}")
        print(f"  road-adas-fp16 (FP16) → {seg_device}")
        print(f"  yolo26n_v2_int8 (INT8) → {det_device}")
        print(f"{'='*80}")

        try:
            segmenter = RoadAdasSegmenter(seg_model_path, seg_device)
            detector = YOLO26nInt8Detector(det_model_path, det_device)
        except Exception as e:
            print(f"\n警告: 配置 {desc} 加载失败: {type(e).__name__}: {e}")
            continue

        print("\n开始测试...")

        sequential_results = benchmark_sequential(segmenter, detector, images, args.warmup, args.iters)
        parallel_results = benchmark_parallel(segmenter, detector, images, args.warmup, args.iters)

        print_results(sequential_results, parallel_results, [seg_device, det_device])

        all_results[desc] = {
            "seg_device": seg_device,
            "det_device": det_device,
            "sequential": sequential_results,
            "parallel": parallel_results,
        }

    if args.all_configs and len(all_results) > 1:
        print(f"\n{'='*100}")
        print("所有配置对比总结")
        print(f"{'='*100}")

        print(f"\n{'配置':<12} | {'模式':<10} | {'总延迟(ms)':>12} | {'分割耗时(ms)':>14} | {'检测耗时(ms)':>14} | {'重叠(ms)':>10} | {'FPS':>8}")
        print(f"{'-'*100}")

        for desc, data in all_results.items():
            seq = data["sequential"]
            par = data["parallel"]
            print(f"{desc:<12} | {'串行':<10} | {seq['total_mean_ms']:>12.2f} | {seq['seg_mean_ms']:>14.2f} | {seq['det_mean_ms']:>14.2f} | {'-':>10} | {seq['total_fps']:>8.2f}")
            print(f"{'':<12} | {'并行':<10} | {par['total_mean_ms']:>12.2f} | {par['seg_mean_ms']:>14.2f} | {par['det_mean_ms']:>14.2f} | {par.get('overlap_mean_ms', '-'):>10} | {par['total_fps']:>8.2f}")

        print(f"\n设备占用分析:")
        for desc, data in all_results.items():
            seq = data["sequential"]
            par = data["parallel"]
            speedup = seq['total_mean_ms'] / par['total_mean_ms'] if par['total_mean_ms'] > 0 else 0
            print(f"  {desc}:")
            print(f"    串行总耗时: {seq['total_mean_ms']:.1f}ms = 分割{seq['seg_mean_ms']:.1f}ms + 检测{seq['det_mean_ms']:.1f}ms")
            print(f"    并行总耗时: {par['total_mean_ms']:.1f}ms（重叠{par.get('overlap_mean_ms', 0):.1f}ms）")
            print(f"    加速比: {speedup:.2f}x")
            if "GPU+NPU" in desc or "NPU+GPU" in desc:
                print(f"    ✓ 双设备并行: 两个模型在不同硬件上同时运行")
            else:
                print(f"    ✗ 单设备竞争: 两个模型共享同一硬件资源，存在资源竞争")

    output_dir = PROJECT_ROOT / "runs" / "parallel_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "segmentation": {"name": "road-adas-fp16", "path": str(seg_model_path)},
            "detection": {"name": "yolo26n_v2_int8", "path": str(det_model_path)},
        },
        "test_config": {
            "test_dir": str(test_dir),
            "image_count": len(images),
            "warmup": args.warmup,
            "iters": args.iters,
            "all_configs": args.all_configs,
        },
        "configurations": all_results,
    }

    json_path = output_dir / "parallel_test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "parallel_test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["配置", "模式", "总延迟均值(ms)", "总延迟中位(ms)", "总延迟P95(ms)", "FPS", "分割耗时(ms)", "检测耗时(ms)", "重叠时间(ms)"])
        for desc, data in all_results.items():
            seq = data["sequential"]
            par = data["parallel"]
            writer.writerow([
                desc, "sequential",
                seq["total_mean_ms"], seq["total_median_ms"], seq["total_p95_ms"],
                seq["total_fps"], seq["seg_mean_ms"], seq["det_mean_ms"], "-",
            ])
            writer.writerow([
                desc, "parallel",
                par["total_mean_ms"], par["total_median_ms"], par["total_p95_ms"],
                par["total_fps"], par["seg_mean_ms"], par["det_mean_ms"], par.get("overlap_mean_ms", "-"),
            ])

    print(f"\n结果已保存至: {output_dir}")
    print(f"  - parallel_test_results.json")
    print(f"  - parallel_test_results.csv")


if __name__ == "__main__":
    main()
