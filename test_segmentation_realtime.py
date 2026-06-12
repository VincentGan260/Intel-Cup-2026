"""实时摄像头语义分割测试（OpenVINO 加速）。"""
import time
import cv2
import yaml
import numpy as np
from pathlib import Path
from src.vision.segmentation.segmenter import build_segmenter_from_config
from src.vision.common.visualize import blend_binary_mask

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    # 加载配置
    config_path = PROJECT_ROOT / "configs/vision/segmentation_openvino.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 构建分割器
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    
    # 获取可视化参数
    alpha = config.get("visualization", {}).get("alpha", 0.45)
    colors_bgr = config.get("visualization", {}).get("colors_bgr", [[0, 0, 0], [0, 200, 0], [0, 128, 255], [200, 200, 200]])
    road_color = tuple(colors_bgr[config.get("model", {}).get("road_class_index", 1)])

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查权限")
        return

    print(f"📷 摄像头已开启，实时语义分割模式，按 q 退出...")
    print(f"🎨 叠加透明度: {alpha}")
    print(f"🚗 可行驶区域颜色: {road_color}")
    
    last_time = time.time()
    fps_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧")
            break

        # 执行分割推理
        result = segmenter.infer(frame)
        
        # 计算 FPS
        fps_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = fps_count / (current_time - last_time)
            fps_count = 0
            last_time = current_time

        # 将分割结果叠加到原图
        if result.drivable_mask is not None:
            overlay = blend_binary_mask(frame, result.drivable_mask, road_color, alpha)
        else:
            overlay = frame

        # 在画面上显示 FPS
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示可行驶区域占比
        if result.drivable_ratio is not None:
            cv2.putText(overlay, f"Drivable: {result.drivable_ratio:.1%}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 显示带分割结果的画面
        cv2.imshow("Camera - Semantic Segmentation (Press Q to quit)", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
