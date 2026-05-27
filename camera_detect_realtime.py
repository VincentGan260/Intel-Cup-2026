"""定时捕获摄像头图像，实时显示检测框（OpenVINO 加速）。"""
import time
import cv2
import yaml
from pathlib import Path
from src.vision.detection.detector import build_detector_from_config

PROJECT_ROOT = Path(__file__).resolve().parent


def draw_detections(frame, detections):
    """在图像帧上绘制检测框和标签。"""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        # 画矩形框（绿色）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 准备标签文本
        label = f"{det.class_name} {det.confidence:.2f} {det.risk_class}"
        # 计算文本大小并画背景
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
        # 写文本（黑色）
        cv2.putText(frame, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main():
    # 加载配置（复用已有的 detection.yaml）
    with open(PROJECT_ROOT / "configs/vision/detection.yaml", "r") as f:
        config = yaml.safe_load(f)

    detector = build_detector_from_config(config, project_root=PROJECT_ROOT)

    # 打开前置摄像头（Mac 默认 0）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查权限")
        return

    print("📷 摄像头已开启，每 3 秒进行一次检测，窗口实时显示框线，按 q 退出...")
    interval = 0  # 检测间隔（秒）
    last_detections = []      # 保存最近一次检测结果
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧")
            break

        # 每隔 interval 秒进行一次推理
        if time.time() - last_time >= interval:
            last_time = time.time()
            last_detections = detector.infer(frame)
            print(f"\n⏱️  {time.strftime('%H:%M:%S')} 检测结果：")
            if not last_detections:
                print("  未检测到目标")
            else:
                for det in last_detections:
                    bbox = ", ".join(f"{v:.1f}" for v in det.bbox)
                    print(f"  - {det.class_name} {det.risk_class} "
                          f"conf={det.confidence:.2f} bbox=[{bbox}]")

        # 在当前帧上绘制最近一次的检测结果
        if last_detections:
            draw_detections(frame, last_detections)

        # 显示带框的画面
        cv2.imshow("Camera - OpenVINO YOLO (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()