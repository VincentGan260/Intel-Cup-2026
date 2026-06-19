"""延迟稳定性测试"""
import cv2, numpy as np, openvino as ov, time
from pathlib import Path

# 模型路径
fp32_xml = "yolo26n_openvino_model/yolo26n.xml"
int8_xml = "yolo26n_int8_v2.xml"

print("=" * 60)
print("延迟稳定性测试（100次推理）")
print("=" * 60)

core = ov.Core()

compiled_fp32 = core.compile_model(fp32_xml, "AUTO")
compiled_int8 = core.compile_model(int8_xml, "AUTO")

# 准备测试数据
test_img = list(Path("datasets/bdd100k/images/100k/val").glob("*.jpg"))[0]
img = cv2.imread(str(test_img))
img = cv2.resize(img, (640, 640))
test_tensor = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

# 预热
print("预热中...")
for _ in range(10):
    compiled_fp32(test_tensor)
    compiled_int8(test_tensor)

# 测试FP32
print("\n测试FP32...")
fp32_times = []
for _ in range(100):
    start = time.perf_counter()
    compiled_fp32(test_tensor)
    fp32_times.append((time.perf_counter() - start) * 1000)

# 测试INT8
print("测试INT8...")
int8_times = []
for _ in range(100):
    start = time.perf_counter()
    compiled_int8(test_tensor)
    int8_times.append((time.perf_counter() - start) * 1000)

fp32_times.sort()
int8_times.sort()

print(f"\nFP32延迟:")
print(f"  均值: {np.mean(fp32_times):.2f} ms")
print(f"  中位: {np.median(fp32_times):.2f} ms")
print(f"  P95:  {fp32_times[94]:.2f} ms")
print(f"  最大: {fp32_times[-1]:.2f} ms")

print(f"\nINT8延迟 (v2):")
print(f"  均值: {np.mean(int8_times):.2f} ms")
print(f"  中位: {np.median(int8_times):.2f} ms")
print(f"  P95:  {int8_times[94]:.2f} ms")
print(f"  最大: {int8_times[-1]:.2f} ms")

print(f"\n加速比: {np.mean(fp32_times)/np.mean(int8_times):.2f}x")
