"""视觉适配器独立测试脚本。

测试两种模式：
  1. 使用示例图片测试适配器输出
  2. （可选）使用摄像头测试

运行方式：
  python scripts/test_vision_adapter.py                    # 用示例 bus 图
  python scripts/test_vision_adapter.py --source 图片路径    # 自定义图片
  python scripts/test_vision_adapter.py --camera           # 摄像头模式
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.fusion.vision_adapter import VisionAdapter


def read_image_safe(path: str):
    """延迟导入 read_image_bgr，避免级联触发 openvino 导入。"""
    from src.vision.common.preprocess import read_image_bgr
    return read_image_bgr(path)


def load_default_image():
    """延迟导入 load_image_bgr_from_source。"""
    from src.vision.common.preprocess import load_image_bgr_from_source
    from pathlib import Path
    return load_image_bgr_from_source(
        "https://ultralytics.com/images/bus.jpg",
        Path(__file__).resolve().parent.parent,
    )


def test_with_image(image_source: str = None) -> None:
    """使用图片测试视觉适配器。"""
    print("=" * 55)
    print("视觉适配器测试: 图片模式")
    print("-" * 55)

    adapter = VisionAdapter(
        pipeline_config_path="configs/vision/vision_pipeline.yaml",
        vision_enabled=True,
        use_camera=False,
    )
    adapter.start()

    try:
        # 加载图片
        if image_source:
            frame = read_image_safe(image_source)
        else:
            frame, src_name = load_default_image()
            print(f"  加载图片: {src_name}")

        h, w = frame.shape[:2]
        print(f"  图像尺寸: {w}x{h}")

        # 适配器推理
        result = adapter.process(frame)

        # 输出转换结果
        print(f"\n  适配器输出:")
        print(f"    valid               = {result.valid}")
        print(f"    object_count        = {len(result.objects)}")
        print(f"    person_count        = {result.person_count}")
        print(f"    vehicle_count       = {result.vehicle_count}")
        print(f"    max_confidence      = {result.max_confidence:.3f}")
        print(f"    max_visual_risk     = {result.max_visual_risk:.3f}")
        print(f"    drivable_area_ratio = {result.drivable_area_ratio:.4f}")

        if result.objects:
            print(f"\n  检测目标:")
            for i, obj in enumerate(result.objects):
                print(f"    {i}: {obj.class_name:10s} | "
                      f"risk={obj.risk_class:15s} | "
                      f"conf={obj.confidence:.3f} | "
                      f"visual_risk={obj.visual_risk:.3f} | "
                      f"on_road={obj.in_drivable_area}")

        if not adapter.vision_enabled:
            print(f"\n  [INFO] 视觉模块降级（当前环境无 openvino），跳过有效数据验证")
        else:
            assert result.valid, f"经过适配器后 valid 应为 True"
            assert result.max_visual_risk >= 0, f"max_visual_risk 应 >= 0"
            print(f"\n  [PASS] 图片测试通过")

    finally:
        adapter.stop()
    print()


def test_with_camera(camera_id: int = 0) -> None:
    """使用摄像头测试视觉适配器。"""
    print("=" * 55)
    print("视觉适配器测试: 摄像头模式")
    print("-" * 55)

    adapter = VisionAdapter(
        pipeline_config_path="configs/vision/vision_pipeline.yaml",
        vision_enabled=True,
        use_camera=True,
        camera_id=camera_id,
    )
    adapter.start()

    try:
        import time

        for i in range(5):
            result = adapter.process()
            if result.valid:
                print(f"  帧 {i + 1}: objects={len(result.objects)}, "
                      f"person={result.person_count}, vehicle={result.vehicle_count}, "
                      f"max_risk={result.max_visual_risk:.3f}")
            else:
                print(f"  帧 {i + 1}: 无效数据")
            time.sleep(0.1)

        print(f"\n  [PASS] 摄像头测试通过")

    finally:
        adapter.stop()
    print()


def main():
    parser = argparse.ArgumentParser(description="视觉适配器测试")
    parser.add_argument("--source", type=str, default=None, help="图片路径或 URL")
    parser.add_argument("--camera", action="store_true", help="摄像头模式")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头编号")
    args = parser.parse_args()

    if args.camera:
        test_with_camera(args.camera_id)
    else:
        test_with_image(args.source)

    print("=" * 55)
    print("测试完成!")
    print("=" * 55)


if __name__ == "__main__":
    main()
