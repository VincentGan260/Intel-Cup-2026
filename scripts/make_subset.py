from pathlib import Path
import shutil
import random
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（原硬编码 Mac 路径已改为可移植）
SRC_IMG_DIR = ROOT / "datasets/bdd100k/test"
DST_IMG_DIR = ROOT / "datasets/bdd100k_subset_500/images/val"
NUM_IMAGES = 500
random.seed(42)
DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
images = sorted(list(SRC_IMG_DIR.glob("*.jpg")))
if len(images) == 0:
    raise FileNotFoundError(f"没有找到图片，请检查路径：{SRC_IMG_DIR}")
selected = random.sample(images, min(NUM_IMAGES, len(images)))
for img_path in selected:
    shutil.copy2(img_path, DST_IMG_DIR / img_path.name)
print(f"已复制 {len(selected)} 张图片到：{DST_IMG_DIR}")
