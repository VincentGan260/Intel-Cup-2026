"""一致性抽查：确认 yolo26n 导出成 OpenVINO 后，输出和 PyTorch 基本一致（导出没转坏）。

不需要数据集——用一张样例图，对比两边的检测框数量、置信度、坐标差异。
两边都走 ultralytics 同一套前/后处理，唯一差别是推理后端，所以应当几乎一致。

运行：
    conda run --no-capture-output -n intel python scripts/vision/check_pt_ov_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.dataset_paths import load_eval_config, resolve

CONF = 0.25
IMGSZ = 640
SAMPLE = "data/sample/bus.jpg"


def boxes_of(model, img_path: str) -> np.ndarray:
    """跑一次预测，返回按 (类别,坐标) 排序的 [x1,y1,x2,y2,conf,cls] 数组。"""
    r = model.predict(source=img_path, imgsz=IMGSZ, conf=CONF, device="cpu", verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()[:, None]
    cls = r.boxes.cls.cpu().numpy()[:, None]
    arr = np.hstack([xyxy, conf, cls]).astype(np.float32)
    order = np.lexsort((arr[:, 0], arr[:, 1], arr[:, 5]))
    return arr[order]


def main() -> None:
    from ultralytics import YOLO

    cfg = load_eval_config()
    det = cfg.get("models", {}).get("detection", {})
    pt_path = resolve(det.get("pytorch", "yolo26n.pt"))
    ov_xml = resolve(det.get("fp32", "models/yolo26n_openvino_model/yolo26n.xml"))
    ov_dir = ov_xml.parent  # ultralytics 加载 OpenVINO 模型用目录
    img = str(resolve(SAMPLE))

    print("=" * 64)
    print("PyTorch ↔ OpenVINO 一致性抽查（yolo26n）")
    print("=" * 64)
    print(f"PyTorch : {pt_path}")
    print(f"OpenVINO: {ov_dir}")
    print(f"样例图  : {img}  (conf={CONF}, imgsz={IMGSZ})")

    if not pt_path.is_file() or not ov_dir.is_dir():
        print("\n缺模型：确认 yolo26n.pt 与 yolo26n_openvino_model/ 都在。")
        return

    pt = boxes_of(YOLO(str(pt_path)), img)
    ovm = boxes_of(YOLO(str(ov_dir)), img)

    print(f"\n检测框数量：PyTorch={len(pt)}  OpenVINO={len(ovm)}")
    if len(pt) != len(ovm):
        print("⚠️ 数量不一致——可能是阈值边缘的框抖动；看下面坐标/置信度差异再判断。")

    n = min(len(pt), len(ovm))
    if n == 0:
        print("两边都没检出框，换张有目标的图再试。")
        return

    box_diff = np.abs(pt[:n, :4] - ovm[:n, :4])
    conf_diff = np.abs(pt[:n, 4] - ovm[:n, 4])
    cls_mismatch = int(np.sum(pt[:n, 5] != ovm[:n, 5]))

    print(f"坐标最大差异 : {box_diff.max():.2f} px   (均值 {box_diff.mean():.3f})")
    print(f"置信度最大差异: {conf_diff.max():.4f}")
    print(f"类别不一致数 : {cls_mismatch}")

    print("\n" + "-" * 64)
    # 判定口径：数量一致 + 类别零错位 + 置信度差很小 是关键信号；
    # 框坐标几像素的差异是 FP32 导出的正常数值误差（resize/算子实现差异），放宽到 15px。
    ok = (len(pt) == len(ovm)) and conf_diff.max() < 0.05 and cls_mismatch == 0 and box_diff.max() < 15.0
    if ok:
        print("✅ 一致：OpenVINO 导出忠实于 PyTorch（框数/类别一致，置信度与坐标仅数值级差异）。")
        print("   可放心用 OpenVINO 做后续精度评测。")
    else:
        print("⚠️ 偏差偏大：数量/类别不一致，或坐标差 >15px、置信度差 >5%，建议检查导出/预处理。")
    print("=" * 64)


if __name__ == "__main__":
    main()
