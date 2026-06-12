"""
创建测试数据集打包脚本
从各数据集中选取指定数量的图片和标注，打包成统一的测试集
支持目标识别和语义分割任务
"""
import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, '.')
from scripts.dataset_paths import load_eval_config, resolve


def copy_files(image_paths, labels_dir, output_images_dir, output_labels_dir,
              image_suffix, label_suffix, det_labels_dir=None, output_det_labels_dir=None, count=100):
    """复制图片和标注到输出目录"""
    copied = 0

    for img_path, img_name in image_paths[:count]:
        # 复制图片
        output_img_path = output_images_dir / img_name
        shutil.copy2(img_path, output_img_path)

        # 复制语义分割标注
        label_name = img_name.replace(image_suffix, label_suffix)
        candidates = list(labels_dir.rglob(label_name))

        if candidates:
            label_path = candidates[0]
            output_label_path = output_labels_dir / label_name
            shutil.copy2(label_path, output_label_path)

            # 复制目标识别标注（如果有）
            if det_labels_dir and output_det_labels_dir:
                det_name = img_name.replace(image_suffix, '.json')
                det_candidates = list(det_labels_dir.rglob(det_name))
                if det_candidates:
                    det_path = det_candidates[0]
                    output_det_path = output_det_labels_dir / det_name
                    shutil.copy2(det_path, output_det_path)

            copied += 1
            print(f"\r复制进度: {copied}/{count}", end="")

    print()
    return copied


def main():
    # 加载配置
    cfg = load_eval_config()
    datasets = cfg.get('segmentation', {}).get('datasets', [])

    # 输出目录
    output_root = Path('datasets/test_package')
    output_root.mkdir(parents=True, exist_ok=True)

    # 每种数据集选取的数量
    sample_count = 100

    # 遍历所有数据集
    for ds in datasets:
        ds_name = ds['name']
        print(f"\n处理数据集: {ds_name}")

        # 输入目录
        images_dir = resolve(ds['images_dir'])
        labels_dir = resolve(ds['labels_dir'])

        # 目标识别标注目录（仅BDD100K有）
        det_labels_dir = None
        if ds_name == 'bdd100k':
            det_labels_dir = resolve('datasets/bdd100k/labels/100k/val')

        # 输出目录
        ds_output_root = output_root / ds_name
        output_images_dir = ds_output_root / 'images'
        output_seg_labels_dir = ds_output_root / 'labels_seg'
        output_det_labels_dir = ds_output_root / 'labels_det'

        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_seg_labels_dir.mkdir(parents=True, exist_ok=True)
        if det_labels_dir:
            output_det_labels_dir.mkdir(parents=True, exist_ok=True)

        # 查找图片
        image_suffix = ds['image_suffix']
        if ds.get('recursive', False):
            images = list(images_dir.rglob(f'*{image_suffix}'))
        else:
            images = list(images_dir.glob(f'*{image_suffix}'))

        images = sorted(images)
        print(f"找到 {len(images)} 张图片")

        # 复制文件
        label_suffix = ds['label_suffix']
        image_paths = [(img, img.name) for img in images]
        copied = copy_files(image_paths, labels_dir, output_images_dir, output_seg_labels_dir,
                          image_suffix, label_suffix, det_labels_dir, output_det_labels_dir, sample_count)

        print(f"成功复制 {copied} 对图片和标注")

        # 保存配置信息
        with open(ds_output_root / 'config.txt', 'w') as f:
            f.write(f"数据集: {ds_name}\n")
            f.write(f"图片数量: {copied}\n")
            f.write(f"图片后缀: {image_suffix}\n")
            f.write(f"语义分割标注后缀: {label_suffix}\n")
            f.write(f"道路类别值: {ds.get('road_value', 0)}\n")
            f.write(f"忽略值: {ds.get('ignore_value', 255)}\n")
            if det_labels_dir:
                f.write(f"支持目标识别: 是（JSON格式）\n")
            else:
                f.write(f"支持目标识别: 否\n")

    # 创建总配置文件
    with open(output_root / 'README.txt', 'w') as f:
        f.write("测试数据集包\n")
        f.write("=" * 50 + "\n\n")
        f.write("包含以下数据集（各100张图片+标注）：\n")
        for ds in datasets:
            f.write(f"- {ds['name']}\n")
        f.write("\n目录结构:\n")
        f.write("datasets/test_package/\n")
        f.write("  ├── dataset_name/\n")
        f.write("  │   ├── images/          # 图片文件\n")
        f.write("  │   ├── labels_seg/      # 语义分割标注（图片格式）\n")
        f.write("  │   ├── labels_det/      # 目标识别标注（JSON格式，仅BDD100K）\n")
        f.write("  │   └── config.txt       # 配置信息\n")
        f.write("  └── README.txt           # 说明文件\n")
        f.write("\n支持的任务:\n")
        f.write("- 语义分割：所有数据集均支持\n")
        f.write("- 目标识别：仅BDD100K支持（包含bounding box标注）\n")

    print(f"\n测试数据集包已创建完成！")
    print(f"输出目录: {output_root.resolve()}")


if __name__ == '__main__':
    main()
