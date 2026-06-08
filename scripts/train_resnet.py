import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet for indoor/outdoor.")
    parser.add_argument("--train_dir", default="train_set")
    parser.add_argument("--val_dir", default="val_set")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--gold_splits_dir", default="pool/splits")
    parser.add_argument("--use_gold", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int = 2) -> nn.Module:
    try:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    except Exception:
        from torchvision.models import resnet50

        model = resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, class_to_idx: dict[str, int], transform: transforms.Compose) -> None:
        self.items: list[tuple[Path, int]] = []
        self.transform = transform
        if not csv_path.exists():
            return
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label")
                path = row.get("path")
                if label is None or path is None:
                    continue
                if label not in class_to_idx:
                    continue
                img_path = Path(path)
                if not img_path.exists():
                    continue
                self.items.append((img_path, class_to_idx[label]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label: str) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=label, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_tf, val_tf = build_transforms()
    train_base = datasets.ImageFolder(args.train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_tf)

    class_to_idx = {name: idx for idx, name in enumerate(train_base.classes)}
    gold_train = CsvImageDataset(Path(args.gold_splits_dir) / "train.csv", class_to_idx, train_tf)
    gold_val = CsvImageDataset(Path(args.gold_splits_dir) / "val.csv", class_to_idx, val_tf)

    train_dataset = train_base
    if args.use_gold and len(gold_train) > 0:
        train_dataset = ConcatDataset([train_base, gold_train])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    gold_val_loader = None
    if args.use_gold and len(gold_val) > 0:
        gold_val_loader = DataLoader(
            gold_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
        )

    model = build_model(num_classes=len(train_base.classes))
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics_rows: list[dict[str, str | int | float]] = []

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, "Val pseudo")
        gold_metrics = None
        if gold_val_loader is not None:
            gold_metrics = evaluate(model, gold_val_loader, device, "Val gold")

        message = (
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if gold_metrics is not None:
            message += (
                f" | gold_val_loss={gold_metrics['loss']:.4f}"
                f" | gold_val_acc={gold_metrics['accuracy']:.4f}"
            )
        print(message)

        metric_source = gold_metrics["accuracy"] if gold_metrics is not None else val_metrics["accuracy"]
        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "gold_val_loss": gold_metrics["loss"] if gold_metrics is not None else "",
                "gold_val_accuracy": gold_metrics["accuracy"] if gold_metrics is not None else "",
                "selection_source": "gold" if gold_metrics is not None else "pseudo",
                "selection_accuracy": metric_source,
            }
        )
        if metric_source > best_acc:
            best_acc = metric_source
            ckpt_path = output_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_base.classes,
                    "val_accuracy": best_acc,
                    "val_source": "gold" if gold_metrics is not None else "pseudo",
                },
                ckpt_path,
            )

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "gold_val_loss",
                "gold_val_accuracy",
                "selection_source",
                "selection_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = {
        "best_val_accuracy": best_acc,
        "classes": train_base.classes,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "gold_splits_dir": args.gold_splits_dir,
        "used_gold": args.use_gold,
        "metrics_csv": str(metrics_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
