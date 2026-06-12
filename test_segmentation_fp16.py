"""ADAS 语义分割 FP16 优化版（更高帧率）。

优化策略：
1. 使用 FP16 量化模型，减少内存带宽占用
2. 降低输入分辨率提高帧率
3. 关闭不必要的后处理操作
4. 使用更高效的可视化方式
"""
import time
import cv2
import yaml
import numpy as np
from pathlib import Path
from src.vision.segmentation.segmenter import build_segmenter_from_config
from src.vision.common.visualize import blend_binary_mask

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    # 加载 FP16 优化配置
    config_path = PROJECT_ROOT / "configs/vision/segmentation_adas_fp16.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("📦 正在构建 FP16 语义分割器...")
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    
    # 获取可视化参数
    alpha = config.get("visualization", {}).get("alpha", 0.45)
    road_class_index = config.get("model", {}).get("road_class_index", 1)
    colors_bgr = config.get("visualization", {}).get("colors_bgr", [[0, 0, 0], [0, 200, 0], [0, 128, 255], [200, 200, 200]])
    road_color = tuple(colors_bgr[road_class_index])

    print(f"🎨 配置: device={config['openvino']['device']}, alpha={alpha}")
    print(f"📐 模型输入: {config['model']['input_width']}x{config['model']['input_height']}")
    
    # 打开摄像头（优化配置）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    # 摄像头优化设置
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 摄像头已开启: {width}x{height}, 按 q 退出")

    # FPS 计算（滑动窗口平均）
    fps_window = 10
    fps_times = []
    frame_count = 0
    last_display_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧")
            break
        
        # 执行分割推理
        start_time = time.perf_counter()
        result = segmenter.infer(frame)
        infer_time = time.perf_counter() - start_time
        
        # FPS 计算
        fps_times.append(infer_time)
        if len(fps_times) > fps_window:
            fps_times.pop(0)
        avg_infer_time = sum(fps_times) / len(fps_times) if fps_times else 0
        current_fps = 1.0 / avg_infer_time if avg_infer_time > 0 else 0
        
        # 叠加分割结果
        if result.drivable_mask is not None:
            overlay = blend_binary_mask(frame, result.drivable_mask, road_color, alpha)
        else:
            overlay = frame
        
        # 显示信息（每帧更新）
        cv2.putText(overlay, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(overlay, f"Infer: {infer_time*1000:.0f}ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # 显示画面
        cv2.imshow("ADAS FP16 - High Performance", cv2.flip(overlay, 1))
        
        # 退出检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
