"""Synthetic forward/backward/save/reload acceptance test; not real training."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.risk_nn import GTMRFN, GTMRFNRuntime, load_feature_schema, save_model_package
from src.risk_nn.features import modality_slices

def main() -> int:
    torch.manual_seed(7); rng = np.random.default_rng(7); schema = load_feature_schema(); slices = modality_slices(schema)
    x = rng.normal(size=(24, schema.window_size, schema.feature_dim)).astype(np.float32)
    for sl in slices.values(): x[:, :, sl.stop - 1] = rng.random((24, schema.window_size)) > .15
    inputs = {m: torch.from_numpy(x[:, :, sl]) for m, sl in slices.items()}; y = torch.from_numpy(rng.integers(0, 3, 24))
    model = GTMRFN(schema.modality_dims); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits = model(inputs); loss = torch.nn.functional.cross_entropy(logits, y); optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert logits.shape == (24, 3) and np.isfinite(float(loss.detach()))
    flat = x.reshape(-1, schema.feature_dim); out = ROOT / "tmp" / "gt_mrfn_synthetic.pt"; out.parent.mkdir(exist_ok=True)
    save_model_package(out, model, schema, flat.mean(0), flat.std(0), synthetic_only=True)
    runtime = GTMRFNRuntime(out); row = {m: {"valid": True, "data": {n: .1 for n in schema.features[m]}} for m in schema.modalities}; prediction = runtime.predict_row(row)
    assert abs(sum(prediction.probabilities) - 1) < 1e-5
    print(f"PASS shape={tuple(logits.shape)} loss={float(loss.detach()):.4f} package={out}"); print(prediction); return 0
if __name__ == "__main__": raise SystemExit(main())
