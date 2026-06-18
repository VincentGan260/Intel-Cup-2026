"""road-adas FP32 → FP16 转换"""
from pathlib import Path
import openvino as ov
from openvino import convert_model, save_model

FP32_XML = "models/openvino/road-segmentation-adas-0001/road-segmentation-adas-0001.xml"
OUT_DIR = Path("models/openvino/road-adas-fp16")

core = ov.Core()
model = core.read_model(FP32_XML)

print(f"FP32 模型: {FP32_XML}")
print(f"  大小: {Path(FP32_XML).with_suffix('.bin').stat().st_size / 1024:.2f} KB")

# 转换为 FP16
fp16_model = convert_model(model, compress_to_fp16=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_xml = OUT_DIR / "road-segmentation-adas-0001.xml"
save_model(fp16_model, str(out_xml))

fp16_size = out_xml.with_suffix(".bin").stat().st_size
print(f"FP16 模型: {out_xml}")
print(f"  大小: {fp16_size / 1024:.2f} KB")
print(f"  压缩比: {Path(FP32_XML).with_suffix('.bin').stat().st_size / fp16_size:.2f}x")

# 快速验证
print(f"\n验证推理...")
import cv2
import numpy as np

img_path = Path("datasets/idd20k_lite/leftImg8bit/val/132/475092_image.jpg")
img = cv2.imread(str(img_path))
img = cv2.resize(img, (896, 512), interpolation=cv2.INTER_LINEAR)
tensor = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

compiled = core.compile_model(str(out_xml), "CPU")
result = compiled(tensor)
logits = next(iter(result.values())).squeeze(0)
label = np.argmax(logits, axis=0).astype(np.uint8)

print(f"  Logits per class mean: {logits.mean(axis=(1,2))}")
print(f"  Label unique: {np.unique(label)}")
print(f"  Road pixels (class 1): {np.sum(label == 1)}")

# 与 FP32 对比
compiled_fp32 = core.compile_model(FP32_XML, "CPU")
result_fp32 = compiled_fp32(tensor)
logits_fp32 = next(iter(result_fp32.values())).squeeze(0)
label_fp32 = np.argmax(logits_fp32, axis=0).astype(np.uint8)
print(f"\n对比 FP32 原版:")
print(f"  FP32  Road pixels: {np.sum(label_fp32 == 1)}")
print(f"  FP16  Road pixels: {np.sum(label == 1)}")
print(f"  一致性: {np.mean(label == label_fp32) * 100:.2f}%")
