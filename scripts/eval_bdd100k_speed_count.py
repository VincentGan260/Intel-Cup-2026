from pathlib import Path
from ultralytics import YOLO
import time
import csv
import psutil
import statistics

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
MODEL_PATH = ROOT / "yolo26n.pt"
IMAGE_DIR = ROOT / "datasets/bdd100k_subset_500/images/val"
OUTPUT_DIR = ROOT / "runs/bdd100k_eval"
IMG_SIZE = 640
CONF_THRES = 0.25

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"没有找到权重文件：{MODEL_PATH}")

image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")))
if len(image_paths) == 0:
    raise FileNotFoundError(f"没有找到图片：{IMAGE_DIR}")

model = YOLO(str(MODEL_PATH))
print("模型类别如下：")
print(model.names)

PEDESTRIAN_NAMES = {"person", "pedestrian"}
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle", "motorbike", "bike", "bicycle"}

csv_path = OUTPUT_DIR / "edge_test_results.csv"
latencies_ms = []
fps_values = []
vehicle_counts = []
pedestrian_counts = []
total_counts = []
process = psutil.Process()

warmup_images = image_paths[:5]
print("开始热身推理...")
for img in warmup_images:
    model.predict(
        source=str(img),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device="cpu",
        verbose=False
    )

print("开始正式测试...")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_name",
        "latency_ms",
        "fps",
        "vehicle_count",
        "pedestrian_count",
        "total_object_count",
        "memory_mb"
    ])

    for idx, image_path in enumerate(image_paths, start=1):
        start_time = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device="cpu",
            verbose=False
        )
        end_time = time.perf_counter()

        latency_ms_val = (end_time - start_time) * 1000
        fps = 1000.0 / latency_ms_val if latency_ms_val > 0 else 0

        r = results[0]
        vehicle_count = 0
        pedestrian_count = 0
        total_object_count = 0

        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            total_object_count = len(cls_ids)
            for cls_id in cls_ids:
                cls_name = str(model.names.get(cls_id, str(cls_id))).lower()
                if cls_name in VEHICLE_NAMES:
                    vehicle_count += 1
                if cls_name in PEDESTRIAN_NAMES:
                    pedestrian_count += 1

        memory_mb = process.memory_info().rss / 1024 / 1024

        latencies_ms.append(latency_ms_val)
        fps_values.append(fps)
        vehicle_counts.append(vehicle_count)
        pedestrian_counts.append(pedestrian_count)
        total_counts.append(total_object_count)

        writer.writerow([
            image_path.name,
            round(latency_ms_val, 2),
            round(fps, 2),
            vehicle_count,
            pedestrian_count,
            total_object_count,
            round(memory_mb, 2)
        ])

        if idx % 50 == 0:
            print(f"已测试 {idx}/{len(image_paths)} 张")

print("\n========== 测试完成 ==========")
print(f"图片数量：{len(image_paths)}")
print(f"平均延迟：{statistics.mean(latencies_ms):.2f} ms")
print(f"中位延迟：{statistics.median(latencies_ms):.2f} ms")
print(f"最大延迟：{max(latencies_ms):.2f} ms")
print(f"最小延迟：{min(latencies_ms):.2f} ms")
print(f"平均 FPS：{statistics.mean(fps_values):.2f}")
print(f"最低 FPS：{min(fps_values):.2f}")
print(f"最高 FPS：{max(fps_values):.2f}")
print(f"平均车辆数：{statistics.mean(vehicle_counts):.2f}")
print(f"平均行人数：{statistics.mean(pedestrian_counts):.2f}")
print(f"平均目标数：{statistics.mean(total_counts):.2f}")
print(f"结果 CSV：{csv_path}")
