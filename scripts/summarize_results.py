from pathlib import Path
import pandas as pd

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
CSV_PATH = ROOT / "runs/bdd100k_eval/edge_test_results.csv"
OUT_PATH = ROOT / "runs/bdd100k_eval/summary_results.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"没有找到测试结果文件：{CSV_PATH}")

df = pd.read_csv(CSV_PATH)

summary = {
    "image_count": len(df),
    "mean_latency_ms": round(df["latency_ms"].mean(), 2),
    "median_latency_ms": round(df["latency_ms"].median(), 2),
    "max_latency_ms": round(df["latency_ms"].max(), 2),
    "min_latency_ms": round(df["latency_ms"].min(), 2),
    "mean_fps": round(df["fps"].mean(), 2),
    "min_fps": round(df["fps"].min(), 2),
    "max_fps": round(df["fps"].max(), 2),
    "mean_vehicle_count": round(df["vehicle_count"].mean(), 2),
    "mean_pedestrian_count": round(df["pedestrian_count"].mean(), 2),
    "mean_total_object_count": round(df["total_object_count"].mean(), 2),
    "mean_memory_mb": round(df["memory_mb"].mean(), 2),
    "max_memory_mb": round(df["memory_mb"].max(), 2)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("整体指标如下：")
print(summary_df)
print(f"整体指标已保存到：{OUT_PATH}")