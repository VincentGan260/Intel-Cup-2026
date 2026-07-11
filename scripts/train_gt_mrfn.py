"""Train report-aligned GT-MRFN from a grouped samples.npz dataset."""
from __future__ import annotations
import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.risk_nn import GTMRFN, load_feature_schema, save_model_package
from src.risk_nn.training import WindowDataset, classification_metrics, fit_normalizer, normalize, split_groups

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None); total, loss_sum, ys, preds = 0, 0.0, [], []
    for inputs, y in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}; y = y.to(device)
        if optimizer: optimizer.zero_grad()
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(inputs); loss = loss_fn(logits, y)
            if optimizer: loss.backward(); optimizer.step()
        total += len(y); loss_sum += float(loss.detach()) * len(y)
        ys.extend(y.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist())
    return loss_sum/max(1,total), np.asarray(ys), np.asarray(preds)

def main() -> int:
    ap = argparse.ArgumentParser(description="Train the three-modal GT-MRFN")
    ap.add_argument("dataset", type=Path); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--schema", default="configs/gt_mrfn_features.yaml"); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8); ap.add_argument("--hidden-dim", type=int, default=24)
    ap.add_argument("--modality-dropout", type=float, default=.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu"); args = ap.parse_args(); seed_all(args.seed)
    schema = load_feature_schema(args.schema); data = np.load(args.dataset, allow_pickle=False)
    x, y, groups = data["X"].astype(np.float32), data["y"].astype(np.int64), data["group_id"].astype(str)
    if x.shape[1:] != (schema.window_size, schema.feature_dim): raise ValueError(f"X维度{x.shape}与schema不符")
    if tuple(data["feature_names"].astype(str)) != schema.feature_names: raise ValueError("数据集特征顺序与schema不一致，请重新构建数据集")
    split = split_groups(groups, y, args.seed); masks = {k: np.isin(groups, v) for k,v in split.items()}
    if any(not mask.any() for mask in masks.values()): raise ValueError("train/val/test存在空集合")
    mu, sigma = fit_normalizer(x[masks["train"]], schema); xn = normalize(x, schema, mu, sigma)
    loaders = {k: DataLoader(WindowDataset(xn[m], y[m], schema, args.modality_dropout if k=="train" else 0), batch_size=args.batch_size, shuffle=k=="train") for k,m in masks.items()}
    counts = np.bincount(y[masks["train"]], minlength=3); weights = counts.sum() / np.maximum(counts, 1); weights /= weights.mean()
    device = torch.device(args.device); model = GTMRFN(schema.modality_dims, args.hidden_dim).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device)); optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best_loss, best_state, stale = [], float("inf"), None, 0; started = time.time()
    for epoch in range(1, args.epochs+1):
        train_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_loss, val_y, val_pred = run_epoch(model, loaders["val"], loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": float((val_y==val_pred).mean())})
        print(f"epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f} acc={history[-1]['val_accuracy']:.4f}")
        if val_loss < best_loss - 1e-5: best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if stale >= args.patience: print(f"early stopping at epoch {epoch}"); break
    assert best_state is not None; model.load_state_dict(best_state)
    test_loss, test_y, test_pred = run_epoch(model, loaders["test"], loss_fn, device)
    metrics = classification_metrics(test_y, test_pred, schema.class_names); metrics.update({"test_loss": test_loss, "class_counts_train": counts.tolist(), "class_weights": weights.tolist(), "elapsed_sec": time.time()-started})
    args.output.mkdir(parents=True, exist_ok=True)
    save_model_package(args.output/"best.pt", model.cpu(), schema, mu, sigma, training_config=vars(args), metrics=metrics)
    (args.output/"split.json").write_text(json.dumps({**split, "seed": args.seed}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output/"metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output/"history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best model: {args.output/'best.pt'}"); print(json.dumps(metrics, ensure_ascii=False, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
