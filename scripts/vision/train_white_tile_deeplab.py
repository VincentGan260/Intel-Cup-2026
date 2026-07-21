"""Fine-tune DeepLabV3-MobileNetV3 for white-tile road segmentation."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)
from torchvision.transforms import ColorJitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "white_tile_road" / "dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "vision" / "white_tile_deeplab"
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RoadDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        height: int,
        width: int,
        augment: bool,
    ) -> None:
        self.root = root
        self.height = int(height)
        self.width = int(width)
        self.augment = bool(augment)
        split_path = root / "splits" / f"{split}.txt"
        self.names = [
            line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.color_jitter = ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15,
            hue=0.03,
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name = self.names[index]
        image = cv2.imread(str(self.root / "images" / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(self.root / "masks" / f"{Path(name).stem}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or mask is None:
            raise RuntimeError(f"Cannot load sample {name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.augment and random.random() < 0.20:
            kernel = random.choice((3, 5))
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)

        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        if self.augment and random.random() < 0.85:
            image_tensor = self.color_jitter(image_tensor)
        image_tensor = (image_tensor - MEAN) / STD
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor, name


def build_model(pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = deeplabv3_mobilenet_v3_large(
        weights=weights,
        weights_backbone=None,
        aux_loss=True,
    )
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 2, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(
            model.aux_classifier[-1].in_channels, 2, kernel_size=1,
        )
    return model


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, ignore_index=255)
    valid = target != 255
    road_target = (target == 1).float()
    road_prob = torch.softmax(logits, dim=1)[:, 1]
    road_prob = road_prob * valid
    road_target = road_target * valid
    intersection = torch.sum(road_prob * road_target)
    denominator = torch.sum(road_prob) + torch.sum(road_target)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return ce + dice_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    intersection = 0
    union = 0
    true_positive = 0
    false_negative = 0

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            logits = outputs["out"]
            loss = segmentation_loss(logits, masks)
            if training and "aux" in outputs:
                loss = loss + 0.4 * segmentation_loss(outputs["aux"], masks)

        if training:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.detach()) * images.shape[0]
        predictions = torch.argmax(logits.detach(), dim=1)
        valid = masks != 255
        pred_road = (predictions == 1) & valid
        true_road = (masks == 1) & valid
        intersection += int(torch.sum(pred_road & true_road))
        union += int(torch.sum(pred_road | true_road))
        true_positive += int(torch.sum(pred_road & true_road))
        false_negative += int(torch.sum((~pred_road) & true_road))

    return {
        "loss": loss_sum / max(1, len(loader.dataset)),
        "road_iou": intersection / max(1, union),
        "road_recall": true_positive / max(1, true_positive + false_negative),
    }


@torch.no_grad()
def save_val_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for images, _, names in loader:
        logits = model(images.to(device))["out"]
        predictions = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        for prediction, name in zip(predictions, names):
            cv2.imwrite(str(output_dir / f"{Path(name).stem}.png"), prediction * 255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    args.output.mkdir(parents=True, exist_ok=True)

    train_dataset = RoadDataset(
        args.dataset, "train", args.height, args.width, augment=True,
    )
    val_dataset = RoadDataset(
        args.dataset, "val", args.height, args.width, augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_amp,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_amp,
        persistent_workers=args.workers > 0,
    )

    print(f"device={device}, train={len(train_dataset)}, val={len(val_dataset)}")
    model = build_model().to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial["model_state"])
        print(f"initialized from {args.init_checkpoint}")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = args.freeze_backbone_epochs <= 0

    head_parameters = list(model.classifier.parameters())
    if model.aux_classifier is not None:
        head_parameters.extend(model.aux_classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs),
    )

    best_iou = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
            print("Backbone unfrozen")

        train_metrics = run_epoch(model, train_loader, device, optimizer, use_amp)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, None, use_amp)
        scheduler.step()

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['road_iou']:.4f} "
            f"val_recall={val_metrics['road_recall']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_metrics": val_metrics,
            "input_size": [args.height, args.width],
            "classes": ["background", "road"],
        }
        torch.save(checkpoint, args.output / "last.pt")
        if val_metrics["road_iou"] > best_iou:
            best_iou = val_metrics["road_iou"]
            epochs_without_improvement = 0
            torch.save(checkpoint, args.output / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without IoU improvement")
                break

    (args.output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    print(f"done in {elapsed:.1f}s, best_val_iou={best_iou:.4f}")
    print(f"best checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
