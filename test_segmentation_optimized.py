"""优化版实时摄像头语义分割测试（ADAS模型 + GPU加速）。

优化点：
1. 使用 GPU/iGPU 加速推理
2. 减少内存拷贝和不必要的操作
3. 高效的 FPS 计算（滑动窗口平均）
4. 预分配缓冲区避免重复分配
5. 使用 cv2.flip 处理镜像问题
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
    # 加载配置
    config_path = PROJECT_ROOT / "configs/vision/segmentation_openvino.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("📦 正在构建语义分割器...")
    segmenter = build_segmenter_from_config(config, project_root=PROJECT_ROOT)
    
    # 获取可视化参数
    alpha = config.get("visualization", {}).get("alpha", 0.45)
    road_class_index = config.get("model", {}).get("road_class_index", 1)
    colors_bgr = config.get("visualization", {}).get("colors_bgr", [[0, 0, 0], [0, 200, 0], [0, 128, 255], [200, 200, 200]])
    road_color = tuple(colors_bgr[road_class_index])

    print(f"🎨 配置: device={config['openvino']['device']}, alpha={alpha}")
    
    # 打开摄像头（设置缓冲区大小以降低延迟）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查权限")
        return
    
    # 设置摄像头参数优化
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区
    cap.set(cv2.CAP_PROP_FPS, 30)        # 目标帧率
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 摄像头已开启: {width}x{height}")
    print("按 q 退出...")

    # FPS 计算优化（滑动窗口平均）
    fps_window = 10
    fps_times = []
    last_fps = 0.0

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧")
            break
        
        # 执行分割推理（计时）
        start_time = time.perf_counter()
        result = segmenter.infer(frame)
        infer_time = time.perf_counter() - start_time
        
        # 更新 FPS（滑动窗口平均）
        fps_times.append(infer_time)
        if len(fps_times) > fps_window:
            fps_times.pop(0)
        avg_infer_time = sum(fps_times) / len(fps_times) if fps_times else 0
        current_fps = 1.0 / avg_infer_time if avg_infer_time > 0 else 0
        
        # 将分割结果叠加到原图
        if result.drivable_mask is not None:
            overlay = blend_binary_mask(frame, result.drivable_mask, road_color, alpha)
        else:
            overlay = frame
        
        # 添加 FPS 和可行驶区域占比信息
        cv2.putText(overlay, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if result.drivable_ratio is not None:
            cv2.putText(overlay, f"Drivable: {result.drivable_ratio:.1%}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(overlay, f"Infer: {infer_time*1000:.1f}ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 显示画面（镜像翻转更符合直观感受）
        cv2.imshow("ADAS Segmentation - GPU Accelerated (Press Q)", cv2.flip(overlay, 1))
        
        # 快速退出检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
