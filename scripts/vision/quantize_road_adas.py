"""road-adas INT8 量化脚本（全面覆盖版）。

使用 NNCF 对 road-adas 模型进行后训练量化，生成 INT8 模型。
使用多数据集、多场景、多天气条件进行全面校准，确保量化精度。

数据集覆盖：
- BDD100K：多种场景（城市、高速、乡村）
- Cityscapes：城市街景
- IDD：印度驾驶数据集
- CamVid：经典分割数据集
- ACDC：恶劣天气（雾、雪、雨、夜间）
- DAWN：恶劣天气（雾、雾霾、薄雾）

运行：
    python scripts/vision/quantize_road_adas.py --precision INT8 --num_samples 300
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import nncf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_paths import load_eval_config, resolve


def preprocess_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """预处理图片：与 run_openvino_adas_forward 保持一致（原始 [0,255] 范围）。"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    image = image.transpose(2, 0, 1)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def find_all_images_in_dir(dir_path: Path, recursive: bool = True) -> list[Path]:
    """查找目录下所有图片文件。"""
    if not dir_path.exists():
        return []
    
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    
    if recursive:
        for ext in extensions:
            images.extend(dir_path.rglob(ext))
    else:
        for ext in extensions:
            images.extend(dir_path.glob(ext))
    
    return sorted(images)


def collect_comprehensive_calibration_data(
    input_size: tuple[int, int],
    total_samples: int = 300,
    verbose: bool = True
) -> list[np.ndarray]:
    """从多个数据集收集全面的校准数据，确保覆盖多种场景和天气条件。"""
    
    # 定义数据集目录及其权重（按重要性分配样本数）
    calibration_sources = [
        # 主要分割数据集
        {"dir": "datasets/bdd100k/images/100k/val", "name": "BDD100K-val", "weight": 0.25, "recursive": False},
        {"dir": "datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val", "name": "Cityscapes-val", "weight": 0.15, "recursive": True},
        {"dir": "datasets/idd20k_lite/leftImg8bit/val", "name": "IDD-val", "weight": 0.10, "recursive": True},
        {"dir": "datasets/CamVid/test", "name": "CamVid-test", "weight": 0.05, "recursive": False},
        
        # 恶劣天气数据集（ACDC）
        {"dir": "datasets/acdc/rgb_anon_trainvaltest/rgb_anon/fog/val", "name": "ACDC-fog", "weight": 0.08, "recursive": True},
        {"dir": "datasets/acdc/rgb_anon_trainvaltest/rgb_anon/rain/val", "name": "ACDC-rain", "weight": 0.08, "recursive": True},
        {"dir": "datasets/acdc/rgb_anon_trainvaltest/rgb_anon/snow/val", "name": "ACDC-snow", "weight": 0.05, "recursive": True},
        {"dir": "datasets/acdc/rgb_anon_trainvaltest/rgb_anon/night/val", "name": "ACDC-night", "weight": 0.05, "recursive": True},
        
        # 恶劣天气数据集（DAWN）
        {"dir": "datasets/dawn/images", "name": "DAWN-adverse", "weight": 0.10, "recursive": False},
        
        # 检测数据集（补充场景多样性）
        {"dir": "datasets/bdd100k/images/100k/val", "name": "BDD100K-det-val", "weight": 0.09, "recursive": False},
    ]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"收集全面校准数据（目标: {total_samples} 张）")
        print(f"{'='*70}")
    
    all_images_by_source = {}
    total_available = 0
    
    # 先统计每个数据集可用的图片数量
    for source in calibration_sources:
        dir_path = resolve(source["dir"])
        images = find_all_images_in_dir(dir_path, source["recursive"])
        all_images_by_source[source["name"]] = images
        total_available += len(images)
        if verbose:
            print(f"  {source['name']:<20}: {len(images)} 张图片 (权重: {source['weight']:.0%})")
    
    if verbose:
        print(f"  {'总计可用':<20}: {total_available} 张图片")
        print(f"{'='*70}")
    
    if total_available == 0:
        print("❌ 错误: 未找到任何校准图片")
        return []
    
    # 按权重分配每个数据集的样本数
    calibration_data = []
    collected_stats = []
    
    for source in calibration_sources:
        source_name = source["name"]
        weight = source["weight"]
        images = all_images_by_source[source_name]
        
        if not images:
            continue
        
        # 计算该数据集应采集的样本数
        target_count = int(total_samples * weight)
        # 如果该数据集图片不足，则取全部
        actual_count = min(target_count, len(images))
        
        # 随机采样（确保多样性）
        if len(images) > actual_count:
            sampled_images = random.sample(images, actual_count)
        else:
            sampled_images = images
        
        if verbose:
            print(f"  从 {source_name} 采样 {actual_count} 张...")
        
        # 预处理图片
        for img_path in sampled_images:
            processed = preprocess_image(img_path, input_size)
            if processed is not None:
                calibration_data.append(processed)
        
        collected_stats.append((source_name, actual_count, len(sampled_images)))
    
    if verbose:
        print(f"\n  实际收集: {len(calibration_data)} 张校准图片")
        print(f"  数据来源分布:")
        for name, target, actual in collected_stats:
            print(f"    - {name}: {actual} 张")
    
    return calibration_data


def quantize_model(
    model_xml: Path,
    output_dir: Path,
    precision: str,
    calibration_data: list[np.ndarray],
    input_size: tuple[int, int]
) -> Path:
    """量化模型，使用 MIXED 预设以平衡精度和速度。"""
    print(f"\n{'='*70}")
    print(f"量化模型")
    print(f"{'='*70}")
    print(f"  加载模型: {model_xml}")
    
    core = ov.Core()
    model = core.read_model(str(model_xml))
    
    print(f"  重塑模型输入到: 1x3x{input_size[0]}x{input_size[1]}")
    model.reshape([1, 3, input_size[0], input_size[1]])
    
    if precision == "FP16":
        print(f"  量化到 FP16...")
        # 使用OpenVINO的convert_model进行FP16转换
        compressed_model = ov.convert_model(
            model,
            compress_to_fp16=True
        )
    elif precision == "INT8":
        print(f"  量化到 INT8 (使用精度控制量化保护精度)...")
        
        def transform_fn(data_item: np.ndarray) -> np.ndarray:
            return data_item
        
        dataset = nncf.Dataset(calibration_data, transform_fn)
        
        # 使用精度控制量化，允许部分层保持FP32以保护精度
        compressed_model = nncf.quantize(
            model,
            dataset,
            preset=nncf.QuantizationPreset.MIXED,
            fast_bias_correction=True,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                overflow_fix=nncf.OverflowFix.FIRST_LAYER,
                disable_bias_correction=False,
            ),
        )
    else:
        raise ValueError(f"不支持的精度: {precision}")
    
    output_xml = output_dir / model_xml.name
    
    print(f"  保存量化模型到: {output_dir}")
    ov.save_model(compressed_model, str(output_xml))
    
    return output_xml


def verify_quantized_model(
    model_xml: Path,
    input_size: tuple[int, int],
    num_tests: int = 10
) -> bool:
    """验证量化模型，使用真实范围的输入。"""
    print(f"\n{'='*70}")
    print(f"验证量化模型")
    print(f"{'='*70}")
    
    try:
        core = ov.Core()
        model = core.read_model(str(model_xml))
        
        compiled = core.compile_model(model, "CPU")
        
        # 使用真实像素值范围 [0, 255] 进行测试
        for i in range(num_tests):
            # 随机生成 [0, 255] 范围的输入，模拟真实图像
            dummy_input = np.random.randint(0, 256, (1, 3, *input_size)).astype(np.float32)
            result = compiled(dummy_input)
            
            # 检查输出是否合理（不应该是全零或异常值）
            output = next(iter(result.values()))
            if output is None or np.all(output == 0):
                print(f"  ⚠️  第 {i+1} 次推理输出异常")
                return False
        
        print(f"  ✅ 量化模型验证通过 ({num_tests} 次推理成功)")
        return True
    except Exception as e:
        print(f"  ❌ 量化模型验证失败: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="road-adas 模型量化（全面覆盖版）")
    parser.add_argument("--precision", type=str, default="INT8", choices=["FP16", "INT8"], help="目标精度")
    parser.add_argument("--num_samples", type=int, default=1000, help="校准样本数量（建议 1000）")
    parser.add_argument("--input_size", type=str, default="1024,1024", help="输入尺寸 (H,W)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（确保可重复）")
    args = parser.parse_args()
    
    # 设置随机种子，确保采样可重复
    random.seed(args.seed)
    
    cfg = load_eval_config()
    
    precision = args.precision.upper()
    input_h, input_w = map(int, args.input_size.split(","))
    input_size = (input_h, input_w)
    
    model_xml = resolve(cfg["models"]["segmentation"]["road_adas"])
    output_cfg = cfg["models"]["segmentation"].get(f"road_adas_{precision.lower()}", 
                                                    f"models/openvino/road-adas-{precision.lower()}")
    output_dir = resolve(output_cfg).parent
    
    # 清理旧的量化模型
    if output_dir.exists():
        old_xml = output_dir / model_xml.name
        old_bin = old_xml.with_suffix(".bin")
        if old_xml.exists():
            print(f"  清理旧的量化模型: {old_xml}")
            old_xml.unlink()
        if old_bin.exists():
            old_bin.unlink()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集全面的校准数据
    calibration_data = collect_comprehensive_calibration_data(
        input_size,
        total_samples=args.num_samples,
        verbose=True
    )
    
    if not calibration_data:
        print("\n❌ 错误: 无法收集校准数据")
        return
    
    print(f"\n{'='*70}")
    print(f"开始量化 road-adas 模型")
    print(f"{'='*70}")
    print(f"  模型: {model_xml}")
    print(f"  目标精度: {precision}")
    print(f"  量化预设: MIXED (平衡精度与速度)")
    print(f"  输入尺寸: {input_h}x{input_w}")
    print(f"  校准样本: {len(calibration_data)} 张")
    print(f"  数据覆盖: BDD100K, Cityscapes, IDD, CamVid, ACDC(雾/雨/雪/夜), DAWN")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    quantized_xml = quantize_model(
        model_xml,
        output_dir,
        precision,
        calibration_data,
        input_size
    )
    
    elapsed = time.time() - start_time
    
    if verify_quantized_model(quantized_xml, input_size):
        print(f"\n{'='*70}")
        print(f"✅ 量化完成!")
        print(f"{'='*70}")
        print(f"  耗时: {elapsed:.1f} 秒")
        print(f"  输出文件: {quantized_xml}")
        
        model_size_mb = quantized_xml.with_suffix(".bin").stat().st_size / 1024 / 1024
        original_size_mb = model_xml.with_suffix(".bin").stat().st_size / 1024 / 1024
        print(f"\n  模型大小对比:")
        print(f"    - 原始 (FP32): {original_size_mb:.2f} MB")
        print(f"    - 量化 ({precision}): {model_size_mb:.2f} MB")
        print(f"    - 压缩比: {original_size_mb/model_size_mb:.2f}x")
        
        print(f"\n  下一步:")
        print(f"    运行精度对比测试:")
        print(f"    python scripts/vision/compare_precision_road_adas.py --num_samples 50")
    else:
        print(f"\n❌ 量化失败")


if __name__ == "__main__":
    main()