# GT-MRFN 训练使用说明

当前网络输入为视觉、雷达和GPS三模态，每个样本使用连续5帧。数据划分以完整外采session为单位，禁止同一session的窗口跨集合。

## 1. 检查每个外采session

```powershell
.\.venv\Scripts\python.exe scripts\check_dashboard_recording.py data\dashboard_recordings\<session>
```

人工复核后，确保`samples.jsonl`中的`label.risk_level`为`low`、`mid`或`high`，并保留质量检查生成的`quality_report.json`。

## 2. 生成滑窗数据集

```powershell
.\.venv\Scripts\python.exe scripts\build_gt_mrfn_dataset.py `
  data\dashboard_recordings\session_low_01 `
  data\dashboard_recordings\session_mid_01 `
  data\dashboard_recordings\session_high_01 `
  --output data\processed\gt_mrfn\samples.npz
```

输出包含`X/y/group_id/end_sample_id/feature_names`以及按模态拆分的数组。至少需要3个独立session；正式评估建议每个风险等级都有多个独立session。

## 3. 训练与评估

```powershell
.\.venv\Scripts\python.exe scripts\train_gt_mrfn.py `
  data\processed\gt_mrfn\samples.npz `
  --output runs\gt_mrfn\main
```

默认参数与设计报告一致：Adam、学习率0.001、batch 64、最多100 epoch、加权交叉熵；另启用15%模态缺失增强和patience=8早停。

产物包括`best.pt`、`split.json`、`metrics.json`和`history.json`。`best.pt`同时保存权重、特征顺序、窗口大小、训练集mu/sigma和评估元数据。

## 4. 导出ONNX（可选）

```powershell
.\.venv\Scripts\python.exe scripts\export_gt_mrfn_onnx.py `
  runs\gt_mrfn\main\best.pt `
  --output runs\gt_mrfn\main\gt_mrfn.onnx
```

ONNX环境缺少`onnx/onnxscript`时，先保留PyTorch CPU部署，不影响真实模型闭环。
