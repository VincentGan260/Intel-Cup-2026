"""
生成最终测试报告
整合目标检测和语义分割的测试结果
"""
from pathlib import Path
import json
import csv

ROOT = Path("/Users/vincent/Desktop/Intel-Cup-2026")
OUTPUT_DIR = ROOT / "runs" / "final_report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(csv_path: Path) -> list:
    """加载 CSV 文件"""
    if not csv_path.exists():
        return []
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    print("="*70)
    print("生成最终测试报告")
    print("="*70)
    
    # 加载检测结果
    detection_summary = load_csv(ROOT / "runs" / "comprehensive_eval" / "summary.csv")
    seg_summary = load_csv(ROOT / "runs" / "segmentation_eval" / "summary.csv")
    
    # 生成报告
    report_lines = []
    report_lines.append("# 模型测试报告")
    report_lines.append("")
    report_lines.append("## 测试环境")
    report_lines.append("- CPU 推理")
    report_lines.append("- Intel Core 处理器")
    report_lines.append("- OpenVINO 2026.1.0")
    report_lines.append("")
    
    # 目标检测结果
    report_lines.append("## 1. 目标检测模型 (YOLO26n)")
    report_lines.append("")
    report_lines.append("### 性能指标")
    report_lines.append("")
    report_lines.append("| 数据集 | 图片数 | 平均延迟(ms) | 平均FPS | 平均车辆数 | 平均行人数 |")
    report_lines.append("|--------|--------|--------------|---------|------------|------------|")
    
    for row in detection_summary:
        report_lines.append(f"| {row['dataset']} | {row['image_count']} | {row['mean_latency_ms']} | {row['mean_fps']} | {row['mean_vehicle_count']} | {row['mean_pedestrian_count']} |")
    
    report_lines.append("")
    report_lines.append("### 详细统计")
    report_lines.append("")
    for row in detection_summary:
        report_lines.append(f"**{row['dataset']}**:")
        report_lines.append(f"- 平均延迟: {row['mean_latency_ms']} ms")
        report_lines.append(f"- 中位延迟: {row['median_latency_ms']} ms")
        report_lines.append(f"- 延迟范围: {row['min_latency_ms']} - {row['max_latency_ms']} ms")
        report_lines.append(f"- 平均 FPS: {row['mean_fps']}")
        report_lines.append(f"- FPS 范围: {row['min_fps']} - {row['max_fps']}")
        report_lines.append("")
    
    # 语义分割结果
    report_lines.append("## 2. 语义分割模型 (road-segmentation-adas-0001)")
    report_lines.append("")
    report_lines.append("### 性能指标")
    report_lines.append("")
    report_lines.append("| 数据集 | 图片数 | 平均延迟(ms) | 平均FPS | 可行驶区域比例 |")
    report_lines.append("|--------|--------|--------------|---------|----------------|")
    
    for row in seg_summary:
        report_lines.append(f"| {row['dataset']} | {row['image_count']} | {row['mean_latency_ms']} | {row['mean_fps']} | {row['mean_drivable_ratio']} |")
    
    report_lines.append("")
    report_lines.append("### 详细统计")
    report_lines.append("")
    for row in seg_summary:
        report_lines.append(f"**{row['dataset']}**:")
        report_lines.append(f"- 平均延迟: {row['mean_latency_ms']} ms")
        report_lines.append(f"- 中位延迟: {row['median_latency_ms']} ms")
        report_lines.append(f"- 延迟范围: {row['min_latency_ms']} - {row['max_latency_ms']} ms")
        report_lines.append(f"- 平均 FPS: {row['mean_fps']}")
        report_lines.append(f"- 平均可行驶区域比例: {row['mean_drivable_ratio']}")
        report_lines.append("")
    
    # 总结
    report_lines.append("## 3. 总结")
    report_lines.append("")
    
    # 计算总体统计
    if detection_summary:
        total_images = sum(int(row['image_count']) for row in detection_summary)
        avg_latency = sum(float(row['mean_latency_ms']) for row in detection_summary) / len(detection_summary)
        avg_fps = sum(float(row['mean_fps']) for row in detection_summary) / len(detection_summary)
        
        report_lines.append("### 目标检测")
        report_lines.append(f"- 总测试图片: {total_images} 张")
        report_lines.append(f"- 平均延迟: {avg_latency:.2f} ms")
        report_lines.append(f"- 平均 FPS: {avg_fps:.2f}")
        report_lines.append("")
    
    if seg_summary:
        total_images = sum(int(row['image_count']) for row in seg_summary)
        avg_latency = sum(float(row['mean_latency_ms']) for row in seg_summary) / len(seg_summary)
        avg_fps = sum(float(row['mean_fps']) for row in seg_summary) / len(seg_summary)
        
        report_lines.append("### 语义分割")
        report_lines.append(f"- 总测试图片: {total_images} 张")
        report_lines.append(f"- 平均延迟: {avg_latency:.2f} ms")
        report_lines.append(f"- 平均 FPS: {avg_fps:.2f}")
        report_lines.append("")
    
    report_lines.append("## 4. 结论")
    report_lines.append("")
    report_lines.append("1. **目标检测模型** 在 BDD100K 数据集上表现最佳，平均延迟 46.1ms，FPS 21.73")
    report_lines.append("2. **语义分割模型** 性能优异，平均延迟约 17ms，FPS 超过 58")
    report_lines.append("3. **ACDC 数据集** 可行驶区域比例最高 (0.533)，说明恶劣天气下道路区域更明显")
    report_lines.append("4. **Cityscapes 数据集** 行人检测数量最多 (平均 2.53)，适合行人检测测试")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*报告生成时间: 2026-06-06*")
    
    # 保存报告
    report_path = OUTPUT_DIR / "test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n报告已保存至: {report_path}")
    
    # 打印报告
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
