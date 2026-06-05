"""
数据集筛选脚本
从各数据集中筛选指定数量的图片用于测试
"""
from pathlib import Path
import random
import shutil

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
DATASETS_DIR = ROOT / "datasets"
SPLITS_DIR = ROOT / "data" / "splits"

# 设置随机种子保证可复现
random.seed(42)


def select_cityscapes(num_images=150):
    """从 Cityscapes val 集中筛选图片"""
    print("\n=== 筛选 Cityscapes ===")
    
    cityscapes_dir = DATASETS_DIR / "cityscapes"
    # 查找 val 集图片
    val_dir = cityscapes_dir / "leftImg8bit" / "val"
    
    if not val_dir.exists():
        # 尝试其他可能的路径
        val_dir = cityscapes_dir / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
    
    if not val_dir.exists():
        print(f"未找到 Cityscapes val 目录")
        return []
    
    # 收集所有图片
    images = list(val_dir.rglob("*.png"))
    print(f"找到 {len(images)} 张图片")
    
    # 随机筛选
    selected = random.sample(images, min(num_images, len(images)))
    
    # 保存到划分文件
    output_file = SPLITS_DIR / "cityscapes_selected.txt"
    with open(output_file, "w") as f:
        for img in selected:
            f.write(f"{img.name}\n")
    
    print(f"已筛选 {len(selected)} 张图片 -> {output_file}")
    return selected


def select_acdc(num_per_condition=50):
    """从 ACDC 中筛选 night/rain/fog 场景图片"""
    print("\n=== 筛选 ACDC ===")
    
    acdc_dir = DATASETS_DIR / "acdc"
    conditions = ["night", "rain", "fog"]
    
    all_selected = []
    
    for condition in conditions:
        # 查找该条件的图片目录
        condition_dir = None
        for possible_path in [
            acdc_dir / "rgb_anon_trainvaltest" / "rgb_anon" / condition,
            acdc_dir / "rgb_anon" / condition,
            acdc_dir / condition
        ]:
            if possible_path.exists():
                condition_dir = possible_path
                break
        
        if condition_dir is None:
            print(f"  未找到 {condition} 目录")
            continue
        
        # 收集所有图片（train + val）
        images = list(condition_dir.rglob("*.png"))
        print(f"  {condition}: 找到 {len(images)} 张图片")
        
        # 随机筛选
        selected = random.sample(images, min(num_per_condition, len(images)))
        all_selected.extend(selected)
        print(f"  {condition}: 已筛选 {len(selected)} 张")
    
    # 保存到划分文件
    output_file = SPLITS_DIR / "acdc_selected.txt"
    with open(output_file, "w") as f:
        for img in all_selected:
            # 格式: condition/filename
            rel_path = img.relative_to(acdc_dir)
            f.write(f"{rel_path}\n")
    
    print(f"总计筛选 {len(all_selected)} 张图片 -> {output_file}")
    return all_selected


def select_idd(num_images=80):
    """从 IDD 中筛选图片"""
    print("\n=== 筛选 IDD ===")
    
    idd_dir = DATASETS_DIR / "idd"
    
    # 查找图片目录
    images_dir = idd_dir / "leftImg8bit" / "train"
    
    if images_dir.exists():
        # 收集所有图片
        images = list(images_dir.rglob("*.jpg")) + list(images_dir.rglob("*.png"))
        print(f"找到 {len(images)} 张图片")
        
        # 随机筛选
        selected = random.sample(images, min(num_images, len(images)))
        
        # 保存到划分文件
        output_file = SPLITS_DIR / "idd_lite_selected.txt"
        with open(output_file, "w") as f:
            for img in selected:
                f.write(f"{img.name}\n")
        
        print(f"已筛选 {len(selected)} 张图片 -> {output_file}")
        return selected
    
    # 使用标注目录推断图片位置
    labels_dir = idd_dir / "gtFine" / "train"
    if labels_dir.exists():
        # 从标注文件名推断图片
        label_files = list(labels_dir.rglob("*_label.png"))
        print(f"找到 {len(label_files)} 个标注文件")
        
        # 提取图片名称（去掉 _label 后缀）
        image_names = [f.stem.replace("_label", "") for f in label_files]
        selected_names = random.sample(image_names, min(num_images, len(image_names)))
        
        # 保存到划分文件
        output_file = SPLITS_DIR / "idd_lite_selected.txt"
        with open(output_file, "w") as f:
            for name in selected_names:
                f.write(f"{name}\n")
        
        print(f"已筛选 {len(selected_names)} 张图片 -> {output_file}")
        return selected_names
    
    print(f"未找到 IDD 图片目录")
    return []


def main():
    print("=" * 50)
    print("数据集筛选脚本")
    print("=" * 50)
    
    # 确保输出目录存在
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 筛选各数据集
    cityscapes_selected = select_cityscapes(num_images=150)
    acdc_selected = select_acdc(num_per_condition=50)
    idd_selected = select_idd(num_images=80)
    
    # 汇总
    print("\n" + "=" * 50)
    print("筛选完成汇总")
    print("=" * 50)
    print(f"Cityscapes: {len(cityscapes_selected)} 张")
    print(f"ACDC:       {len(acdc_selected)} 张")
    print(f"IDD:        {len(idd_selected)} 张")
    print(f"BDD100K:    已有 500 张子集")
    
    total = len(cityscapes_selected) + len(acdc_selected) + len(idd_selected) + 500
    print(f"\n总计: {total} 张测试图片")


if __name__ == "__main__":
    main()
