"""
YOLOv26n INT8量化脚本
使用多数据集（BDD100K + IDD + Cityscapes）作为校准数据
"""
import nncf
import cv2
import numpy as np
from openvino import Core, serialize
from pathlib import Path
import random
from ultralytics import YOLO

# ========== 配置区域 ==========
MODEL_NAME = "yolo26n"  # 模型名称
MODEL_PATH = Path("yolo26n.pt")  # 模型路径

# 校准数据集目录（全部使用）
CALIB_DIRS = [
    Path("datasets/bdd100k/images/100k/val"),      # BDD100K验证集
    Path("datasets/idd20k_lite/leftImg8bit/val"),  # IDD验证集
    Path("datasets/cityscapes/leftImg8bit/val"),   # Cityscapes验证集
]

# 校准参数
CALIB_SAMPLES = 500  # 校准样本数量（越多越准确但越慢）
IMG_SIZE = 640        # 输入图片尺寸

# 量化预设: MIXED (平衡精度和速度)
# 可选: PERFORMANCE (更快), MIXED (推荐), VALIDATION (更准)
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

    # 随机采样
    selected = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"\n总计: {len(all_images)} 张图片，采样 {len(selected)} 张")
    return selected

def preprocess_image(p, target_size):
    """预处理单张图片"""
    img = cv2.imread(str(p))
    if img is None:
        return None
    # YOLO预处理: resize + 归一化 + HWC->CHW
    img = cv2.resize(img, (target_size, target_size))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

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
        """预处理函数"""
        img = cv2.imread(str(p))
        if img is None:
            return None
        img = cv2.resize(img, (input_size, input_size))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return {input_name: np.expand_dims(img, 0)}

    # 创建NNCF数据集
    dataset = nncf.Dataset(image_paths, preprocess_fn)
    return dataset

def main():
    print("=" * 60)
    print("YOLOv26n INT8 量化")
    print("=" * 60)

    # 1. 导出FP32 OpenVINO模型
    print("\n[1/4] 导出 FP32 OpenVINO 模型...")
    model = YOLO(str(MODEL_PATH))
    export_path = model.export(format="openvino", dynamic=True)
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
    print(f"      量化预设: MIXED")
    print(f"      校准样本: {len(calib_images)}")
    print(f"      输入尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"      溢出修复: FIRST_LAYER")

    core = Core()
    ov_model = core.read_model(str(xml_path))

    quantized_model = nncf.quantize(
        ov_model,
        calib_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calib_images),
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            overflow_fix=nncf.OverflowFix.FIRST_LAYER
        )
    )

    # 保存量化模型
    output_xml = f"{MODEL_NAME}_int8.xml"
    serialize(quantized_model, output_xml)
    output_bin = output_xml.replace(".xml", ".bin")

    # 计算压缩比
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
    print("\n建议使用 benchmark_app 测试性能:")
    print(f"  benchmark_app -m {output_xml} -d CPU")

def test_model():
    """测试量化模型"""
    import time

    output_xml = f"{MODEL_NAME}_int8.xml"
    fp32_xml = Path(export_path) / f"{MODEL_NAME}.xml"

    print("\n" + "=" * 60)
    print("模型测试")
    print("=" * 60)

    core = Core()
    compiled_fp32 = core.compile_model(str(fp32_xml), "CPU")
    compiled_int8 = core.compile_model(str(output_xml), "CPU")

    # 准备测试数据
    test_img = cv2.imread(str(calib_images[0]))
    test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
    test_tensor = test_img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

    # 性能测试
    num_runs = 100

    print("运行推理测试...")
    for _ in range(10):  # 预热
        compiled_fp32(test_tensor)
        compiled_int8(test_tensor)

    start = time.time()
    for _ in range(num_runs):
        compiled_fp32(test_tensor)
    fp32_time = (time.time() - start) / num_runs * 1000

    start = time.time()
    for _ in range(num_runs):
        compiled_int8(test_tensor)
    int8_time = (time.time() - start) / num_runs * 1000

    print(f"\n性能对比 ({num_runs}次平均):")
    print(f"  FP32延迟: {fp32_time:.2f} ms ({1000/fp32_time:.1f} FPS)")
    print(f"  INT8延迟: {int8_time:.2f} ms ({1000/int8_time:.1f} FPS)")
    print(f"  加速比:   {fp32_time/int8_time:.2f}x")

if __name__ == "__main__":
    main()
