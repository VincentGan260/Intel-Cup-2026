from pathlib import Path
from ultralytics import YOLO
ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
MODEL_PATH = ROOT / "yolo26n.pt"
IMAGE_DIR = ROOT / "datasets/bdd100k_subset_500/images/val"
OUTPUT_DIR = ROOT / "runs/single_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"没有找到权重文件：{MODEL_PATH}")
image_list = sorted(IMAGE_DIR.glob("*.jpg"))
if not image_list:
    raise FileNotFoundError(f"没有找到测试图片：{IMAGE_DIR}")
model = YOLO(str(MODEL_PATH))
for i, image_path in enumerate(image_list):
    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.15,
        iou=0.45,
        device="cpu",
        save=True,
        save_conf=True,
        save_txt=False,
        project=str(OUTPUT_DIR),
        name="predict",
        exist_ok=True,
        show=False
    )
    boxes = results[0].boxes
    print(f"图片 {i+1}/{len(image_list)}: {image_path.name}")
    print(f"  检测到 {len(boxes)} 个目标")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"    - 类别: {model.names[cls]}, 置信度: {conf:.2f}")
    print()
print(f"\n所有结果已保存到：{OUTPUT_DIR / 'predict'}")