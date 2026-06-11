"""端侧延迟基准（纯推理）。不需要任何数据集——用随机张量按模型输入形状计时。

测什么：每个 (模型 × 精度 × 设备) 的纯推理延迟（均值/中位/p95/最大）、FPS、IR 体积。
另含并发模式：两个模型各绑一个设备同时跑，测合并每帧延迟（你们检测+分割并行的真实数）。

口径：batch=1、先预热再测稳态、统计 p95（偶发卡顿才是实时杀手）。

【重要】延迟必须在目标硬件（DK-2500）上跑才有意义。本脚本在开发机能跑通逻辑，
但只有 CPU 有结果（GPU/NPU 没有就自动跳过）；上板后三种设备齐全，直接出数。

运行：
    conda run --no-capture-output -n intel python scripts/vision/bench_latency.py
    conda run --no-capture-output -n intel python scripts/vision/bench_latency.py --concurrent
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov


@contextlib.contextmanager
def suppress_native_output():
    """临时屏蔽 C 层 stdout/stderr（GPU 内核 JIT 的 "N warnings generated" 刷屏）。"""
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        for fd in saved:
            os.close(fd)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.dataset_paths import load_eval_config, resolve


def random_inputs(compiled) -> dict:
    """按模型输入形状造随机张量（动态维填 1）。纯推理延迟与输入内容无关。"""
    feed = {}
    for port in compiled.inputs:
        pshape = port.get_partial_shape()
        shape = [d.get_length() if d.is_static else 1 for d in pshape]
        feed[port.get_any_name()] = np.random.rand(*shape).astype(np.float32)
    return feed


def stats(times_ms: list[float]) -> dict:
    s = sorted(times_ms)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    mean = statistics.mean(s)
    return {
        "mean_ms": round(mean, 2),
        "median_ms": round(statistics.median(s), 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(s[0], 2),
        "max_ms": round(s[-1], 2),
        "fps": round(1000.0 / mean, 1) if mean > 0 else 0.0,
    }


def ir_size_mb(xml: Path) -> float:
    bin_ = xml.with_suffix(".bin")
    return round(bin_.stat().st_size / 1e6, 1) if bin_.is_file() else 0.0


def bench_single(core, xml: Path, device: str, warmup: int, iters: int) -> dict | None:
    """单模型单设备纯推理计时。编译/运行失败返回 None（如 NPU 不支持该模型）。"""
    try:
        with suppress_native_output():
            compiled = core.compile_model(core.read_model(str(xml)), device)
        feed = random_inputs(compiled)
        req = compiled.create_infer_request()
        for _ in range(warmup):
            req.infer(feed)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            req.infer(feed)
            times.append((time.perf_counter() - t0) * 1000.0)
        out = stats(times)
        out["size_mb"] = ir_size_mb(xml)
        return out
    except Exception as e:
        print(f"    [跳过] {device} 上编译/运行失败：{type(e).__name__}: {str(e)[:80]}")
        return None


def bench_concurrent(core, assignments: list[tuple], warmup: int, iters: int) -> dict | None:
    """两模型各绑一设备并发（async）跑，测合并每帧延迟（每帧需两者都完成）。"""
    try:
        reqs, feeds = [], []
        for _, xml, device in assignments:
            with suppress_native_output():
                compiled = core.compile_model(core.read_model(str(xml)), device)
            reqs.append(compiled.create_infer_request())
            feeds.append(random_inputs(compiled))
        for _ in range(warmup):
            for r, f in zip(reqs, feeds):
                r.start_async(f)
            for r in reqs:
                r.wait()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for r, f in zip(reqs, feeds):
                r.start_async(f)
            for r in reqs:
                r.wait()
            times.append((time.perf_counter() - t0) * 1000.0)
        return stats(times)
    except Exception as e:
        print(f"    [跳过] 并发失败：{type(e).__name__}: {str(e)[:80]}")
        return None


def collect_models(cfg: dict) -> list[tuple]:
    """从 eval.yaml 收集存在的 (标签, xml, 任务) 模型列表。"""
    m = cfg.get("models", {})
    det, seg = m.get("detection", {}), m.get("segmentation", {})
    candidates = [
        ("yolo-fp32", det.get("fp32"), "det"),
        ("yolo-fp16", det.get("fp16"), "det"),
        ("yolo-int8", det.get("int8"), "det"),
        ("pidnet-fp32", seg.get("pidnet_fp32"), "seg"),
        ("pidnet-fp16", seg.get("pidnet_fp16"), "seg"),
        ("pidnet-int8", seg.get("pidnet_int8"), "seg"),
        ("road_adas-fp32", seg.get("road_adas"), "seg"),
        ("road_adas-fp16", seg.get("road_adas_fp16"), "seg"),
    ]
    out = []
    for label, rel, task in candidates:
        if rel and resolve(rel).is_file():
            out.append((label, resolve(rel), task))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="端侧延迟基准（纯推理）")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--concurrent", action="store_true", help="额外测检测+分割并发摆放")
    args = parser.parse_args()

    cfg = load_eval_config()
    lat = cfg.get("latency", {})
    warmup = args.warmup if args.warmup is not None else int(lat.get("warmup", 20))
    iters = args.iters if args.iters is not None else int(lat.get("iters", 200))
    floor = float(lat.get("realtime_floor_fps", 10))

    core = ov.Core()
    available = core.available_devices
    want = [d for d in lat.get("devices", ["CPU"]) if any(d == a or a.startswith(d) for a in available)]
    if "CPU" not in want:
        want = ["CPU"] + want

    print("=" * 78)
    print(f"延迟基准  warmup={warmup} iters={iters}  实时地板={floor}FPS")
    print(f"本机可用设备：{available}    本次测试：{want}")
    if want == ["CPU"]:
        print("提示：未检测到 GPU/NPU（应该是在开发机上）。逻辑已验证，上 DK-2500 再测全设备。")
    print("=" * 78)

    models = collect_models(cfg)
    if not models:
        print("没有可用模型，先确认 eval.yaml 里的模型路径。")
        return

    rows = []
    for label, xml, task in models:
        print(f"\n[{label}]  {xml.name}")
        for device in want:
            r = bench_single(core, xml, device, warmup, iters)
            if r:
                flag = "✓" if r["fps"] >= floor else " "
                print(f"  {device:<4} 均值 {r['mean_ms']:>7.2f}ms  p95 {r['p95_ms']:>7.2f}ms  "
                      f"{r['fps']:>6.1f} FPS {flag}  ({r['size_mb']}MB)")
                rows.append({"model": label, "task": task, "device": device, **r})

    result = {"warmup": warmup, "iters": iters, "devices": want, "single": rows}

    # 并发摆放（阶段 B）
    if args.concurrent:
        print("\n" + "-" * 78)
        print("并发：检测 + 分割 各绑一设备同时跑（合并每帧延迟）")
        m = cfg.get("models", {})
        det_xml = resolve(m["detection"]["fp32"]) if m.get("detection", {}).get("fp32") else None
        seg_xml = resolve(m["segmentation"]["pidnet_fp32"]) if m.get("segmentation", {}).get("pidnet_fp32") else None
        combos = []
        if det_xml and seg_xml and det_xml.is_file() and seg_xml.is_file():
            # 开发机一般只有 CPU；上板可换成 NPU/GPU 组合
            pool = want if len(want) > 1 else ["CPU", "CPU"]
            combos = [
                [("det", det_xml, pool[0]), ("seg", seg_xml, pool[-1])],
            ]
        conc_rows = []
        for combo in combos:
            tag = " + ".join(f"{c[0]}@{c[2]}" for c in combo)
            r = bench_concurrent(core, combo, warmup, iters)
            if r:
                flag = "✓" if r["fps"] >= floor else " "
                print(f"  {tag:<28} 均值 {r['mean_ms']:>7.2f}ms  p95 {r['p95_ms']:>7.2f}ms  {r['fps']:>6.1f} FPS {flag}")
                conc_rows.append({"placement": tag, **r})
        result["concurrent"] = conc_rows

    out_dir = resolve(cfg.get("output", {}).get("latency_dir", "runs/latency_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_results.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"结果已存：{out_path}")
    print("（开发机只有 CPU 数；上 DK-2500 跑同一脚本即得 CPU/GPU/NPU 全表。）")


if __name__ == "__main__":
    main()
