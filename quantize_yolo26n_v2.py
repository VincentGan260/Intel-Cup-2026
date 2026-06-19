"""
YOLOv26n INT8量化脚本（优化版）
解决延迟抖动问题
"""
import nncf
import cv2
import numpy as np
from openvino import Core, serialize
from pathlib import Path
import random
from ultralytics import YOLO

# ========== 配置区域 ==========
MODEL_NAME = "yolo26n"
MODEL_PATH = Path("yolo26n.pt")

# 校准数据集目录
CALIB_DIRS = [
    Path("datasets/bdd100k/images/100k/val"),
    Path("datasets/idd20k_lite/leftImg8bit/val"),
]

# 优化参数
CALIB_SAMPLES = 1000  # 增加校准样本
IMG_SIZE = 640

# 量化预设: PERFORMANCE 最大化速度
# ========== 配置结束 ==========

def get_calibration_images(dirs, num_samples):
    """收集校准图片"""
    all_images = []
    for d in dirs:
        if d.exists():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            all_images.extend(imgs)
            print(f"  {d}: {len(imgs)} 张图片")
        else:
            print(f"  {d}: 目录不存在，跳过")

    if len(all_images) < 50:
        raise FileNotFoundError(f"校准图片不足50张，当前{len(all_images)}张")

    selected = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"\n总计: {len(all_images)} 张图片，采样 {len(selected)} 张")
    return selected

def get_input_name(model_path):
    """获取模型输入名称"""
    core = Core()
    model = core.read_model(str(model_path))
    return model.input(0).get_any_name()

def create_calibration_dataset(image_paths, input_size, xml_path):
    """创建NNCF校准数据集"""
    import nncf

    input_name = get_input_name(xml_path)

    def preprocess_fn(p):
        img = cv2.imread(str(p))
        if img is None:
            return None
        img = cv2.resize(img, (input_size, input_size))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return {input_name: np.expand_dims(img, 0)}

    dataset = nncf.Dataset(image_paths, preprocess_fn)
    return dataset

def main():
    print("=" * 60)
    print("YOLOv26n INT8 量化（优化版）")
    print("=" * 60)

    # 1. 导出FP32 OpenVINO模型（固定尺寸，解决dynamic延迟问题）
    print("\n[1/4] 导出 FP32 OpenVINO 模型（固定尺寸）...")
    model = YOLO(str(MODEL_PATH))
    # 使用固定尺寸，不使用dynamic
    export_path = model.export(
        format="openvino",
        imgsz=IMG_SIZE,
        half=False
    )
    xml_path = Path(export_path) / f"{MODEL_NAME}.xml"
    if not xml_path.exists():
        xml_path = Path(export_path).with_suffix(".xml")

    print(f"      FP32模型: {xml_path}")

    # 2. 收集校准数据
    print("\n[2/4] 收集校准数据...")
    print("      数据集列表:")
    calib_images = get_calibration_images(CALIB_DIRS, CALIB_SAMPLES)

    # 3. 创建校准数据集
    print("\n[3/4] 创建校准数据集...")
    calib_dataset = create_calibration_dataset(calib_images, IMG_SIZE, xml_path)
    print(f"      校准样本数: {len(calib_images)}")

    # 4. 执行INT8量化
    print("\n[4/4] 开始 INT8 量化...")
    print(f"      量化预设: PERFORMANCE（最大化速度）")
    print(f"      校准样本: {len(calib_images)}")
    print(f"      输入尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"      溢出修复: FIRST_LAYER")

    core = Core()
    ov_model = core.read_model(str(xml_path))

    # 使用MIXED预设，平衡精度和速度
    quantized_model = nncf.quantize(
        ov_model,
        calib_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calib_images),
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            overflow_fix=nncf.OverflowFix.FIRST_LAYER,
        )
    )

    # 保存量化模型
    output_xml = f"{MODEL_NAME}_int8_v2.xml"
    serialize(quantized_model, output_xml)
    output_bin = output_xml.replace(".xml", ".bin")

    fp32_size = Path(str(xml_path).replace(".xml", ".bin")).stat().st_size
    int8_size = Path(output_bin).stat().st_size

    print("\n" + "=" * 60)
    print("量化完成!")
    print("=" * 60)
    print(f"输出模型: {output_xml}")
    print(f"FP32大小: {fp32_size / 1024 / 1024:.2f} MB")
    print(f"INT8大小: {int8_size / 1024 / 1024:.2f} MB")
    print(f"压缩比:   {fp32_size / int8_size:.2f}x")
    print("=" * 60)

def test_latency():
    """测试延迟稳定性"""
    import time

    output_xml = f"{MODEL_NAME}_int8_v2.xml"
    fp32_xml = Path(export_path) / f"{MODEL_NAME}.xml"

    print("\n" + "=" * 60)
    print("延迟稳定性测试（100次推理）")
    print("=" * 60)

    core = Core()

    # 强制使用CPU，避免AUTO设备分配问题
    core.set_property({'CPU': {'ENABLE_MMAP': True}})

    compiled_fp32 = core.compile_model(str(fp32_xml), "CPU")
    compiled_int8 = core.compile_model(str(output_xml), "CPU")

    # 准备测试数据
    test_img = cv2.imread(str(calib_images[0]))
    test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
    test_tensor = test_img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

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

    print(f"\nINT8延迟:")
    print(f"  均值: {np.mean(int8_times):.2f} ms")
    print(f"  中位: {np.median(int8_times):.2f} ms")
    print(f"  P95:  {int8_times[94]:.2f} ms")
    print(f"  最大: {int8_times[-1]:.2f} ms")

    print(f"\n加速比: {np.mean(fp32_times)/np.mean(int8_times):.2f}x")

if __name__ == "__main__":
    main()
