
"""定时捕获摄像头图像并使用 OpenVINO 检测器进行推理。"""
import time
import cv2
import yaml
from pathlib import Path
from src.vision.detection.detector import build_detector_from_config

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    # 加载配置（复用你已有的 detection.yaml）
    with open(PROJECT_ROOT / "configs/vision/detection.yaml", "r") as f:
        config = yaml.safe_load(f)

    detector = build_detector_from_config(config, project_root=PROJECT_ROOT)

    # 打开前置摄像头（Mac 默认索引为 0）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查权限")
        return

    print("📷 摄像头已开启，每 3 秒进行一次检测，按 q 退出...")
    interval = 0  # 检测间隔（秒），可修改
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧")
            break

        # 每隔 interval 秒检测一次
        if time.time() - last_time >= interval:
            last_time = time.time()
            # 注意：frame 是 BGR 格式，与你的 infer 接口一致
            detections = detector.infer(frame)
            print(f"\n⏱️  {time.strftime('%H:%M:%S')} 检测结果：")
            if not detections:
                print("  未检测到目标")
            else:
                for det in detections:
                    bbox = ", ".join(f"{v:.1f}" for v in det.bbox)
                    print(f"  - {det.class_name} {det.risk_class} "
                          f"conf={det.confidence:.2f} bbox=[{bbox}]")

        # 显示实时画面（可选）
        cv2.imshow("Camera - Press 'Q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()