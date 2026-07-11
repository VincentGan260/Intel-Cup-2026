"""Export a trained GT-MRFN package to ONNX with one input per modality."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.risk_nn import GTMRFN, load_feature_schema

class ExportWrapper(torch.nn.Module):
    def __init__(self, model, names): super().__init__(); self.model, self.names = model, names
    def forward(self, *values): return self.model(dict(zip(self.names, values)))

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("package", type=Path); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--schema", default="configs/gt_mrfn_features.yaml"); args=ap.parse_args()
    schema=load_feature_schema(args.schema); package=torch.load(args.package, map_location="cpu", weights_only=False)
    if tuple(package["feature_names"]) != schema.feature_names: raise ValueError("package与schema特征不一致")
    model=GTMRFN(package["modality_dims"], package["hidden_dim"]); model.load_state_dict(package["model_state"]); model.eval()
    names=tuple(schema.modalities); dummy=tuple(torch.zeros(1, schema.window_size, schema.modality_dims[n]) for n in names); args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(ExportWrapper(model, names), dummy, args.output, input_names=list(names), output_names=["logits"], dynamic_axes={**{n:{0:"batch"} for n in names}, "logits":{0:"batch"}}, opset_version=17)
    print(f"ONNX saved: {args.output}"); return 0
if __name__ == "__main__": raise SystemExit(main())
