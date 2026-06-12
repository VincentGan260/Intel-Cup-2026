"""ADAS 语义分割最终优化版。

修复问题：
1. 移除镜像翻转
2. 优化 CPU 性能
3. 提高帧率
"""
import time
import cv2
import numpy as np
from pathlib import Path
import openvino as ov
from src.vision.common.types import SegmentationResult

PROJECT_ROOT = Path(__file__).resolve().parent


class FastAdasSegmenter:
    """优化版 ADAS 分割器"""
    
    def __init__(self, xml_path, road_class=1):
        self.core = ov.Core()
        self.model = self.core.read_model(str(xml_path))
        
        # 使用默认配置
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # 获取模型输入尺寸
        input_shape = self.input_layer.get_shape()
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.road_class = road_class
    
    def infer(self, frame):
        # 预处理：缩放并转换为模型输入格式
        img_resized = cv2.resize(frame, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), 0).astype(np.float32)
        
        # 推理
        result = self.compiled_model(input_data)[self.output_layer]
        
        # 后处理：获取可行驶区域 mask
        logits = result[0]
        label_map = np.argmax(logits, axis=0)
        drivable_mask = (label_map == self.road_class).astype(np.uint8)
        
        # 缩放到原始尺寸
        drivable_mask = cv2.resize(drivable_mask, (frame.shape[1], frame.shape[0]))
        
        # 计算可行驶区域占比
        drivable_ratio = np.sum(drivable_mask) / drivable_mask.size
        
        return SegmentationResult(
            drivable_mask=drivable_mask,
            raw_mask=label_map,
            drivable_ratio=drivable_ratio
        )


def main():
    # 使用 FP16 模型（如果可用）
    model_path = PROJECT_ROOT / "models/openvino/road-adas-fp16/road-segmentation-adas-0001.xml"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models/openvino/road-segmentation-adas-0001/road-segmentation-adas-0001.xml"
    
    print(f"📦 加载模型: {model_path}")
    
    # 创建分割器
    segmenter = FastAdasSegmenter(model_path, road_class=1)
    
    print(f"✅ 分割器初始化完成")
    print(f"📐 模型输入: {segmenter.input_width}x{segmenter.input_height}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    # 优化设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 摄像头: {width}x{height}, 按 q 退出")
    
    # 预热
    ret, frame = cap.read()
    if ret:
        for _ in range(3):
            segmenter.infer(frame)
        print("🔄 预热完成")
    
    # FPS 计算
    fps_window = 10
    fps_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧")
            break
        
        # 推理
        start_time = time.perf_counter()
        result = segmenter.infer(frame)
        infer_time = time.perf_counter() - start_time
        
        # FPS
        fps_times.append(infer_time)
        if len(fps_times) > fps_window:
            fps_times.pop(0)
        avg_time = sum(fps_times) / len(fps_times) if fps_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # 可视化（不镜像）
        overlay = frame.copy()
        if result.drivable_mask is not None:
            overlay[result.drivable_mask > 0, 1] = np.minimum(overlay[result.drivable_mask > 0, 1] + 150, 255)
        
        # 显示信息
        cv2.putText(overlay, f"FPS: {fps:.1f}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"Drivable: {result.drivable_ratio:.0%}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 不进行镜像翻转
        cv2.imshow("ADAS Segmentation", overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
