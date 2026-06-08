import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


class CsvImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
        root: Path,
    ) -> None:
        self.items: list[tuple[Path, int]] = []
        self.transform = transform
        self.root = root

        if not csv_path.exists():
            return

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row.get("label")
                raw_path = row.get("path")
                if label is None or raw_path is None:
                    continue
                if label not in class_to_idx:
                    continue
                path = Path(str(raw_path).replace("\\", "/"))
                if not path.is_absolute():
                    path = self.root / path
                path = path.resolve()
                if not path.exists():
                    continue
                self.items.append((path, class_to_idx[label]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root: str, transform: transforms.Compose, excluded_paths: set[Path]) -> None:
        super().__init__(root=root, transform=transform)
        excluded = {p.resolve() for p in excluded_paths}

        filtered_samples: list[tuple[str, int]] = []
        for path_str, class_idx in self.samples:
            candidate = Path(path_str).resolve()
            if candidate not in excluded:
                filtered_samples.append((path_str, class_idx))

        self.samples = filtered_samples
        self.imgs = filtered_samples
        self.targets = [class_idx for _, class_idx in filtered_samples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final frozen ResNet50 + logistic layer on pool + gold train/val."
    )
    parser.add_argument("--pool_dir", default="pool")
    parser.add_argument("--gold_splits_dir", default="pool/splits")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="checkpoints/final_logistic")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed for deterministic CUDA behavior in some kernels.
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


def build_model(num_classes: int) -> nn.Module:
    try:
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        from torchvision.models import resnet50

        model = resnet50(pretrained=True)

    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False

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


def load_split_paths(csv_path: Path, root: Path) -> set[Path]:
    paths: set[Path] = set()
    if not csv_path.exists():
        return paths

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_path = row.get("path")
            if raw_path is None:
                continue
            path = Path(str(raw_path).replace("\\", "/"))
            if not path.is_absolute():
                path = root / path
            paths.add(path.resolve())
    return paths


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = Path(".").resolve()
    pool_dir = Path(args.pool_dir).resolve()
    splits_dir = Path(args.gold_splits_dir).resolve()

    train_tf, val_tf = build_transforms()

    excluded_pool_paths = set()
    excluded_pool_paths |= load_split_paths(splits_dir / "train.csv", root)
    excluded_pool_paths |= load_split_paths(splits_dir / "val.csv", root)
    excluded_pool_paths |= load_split_paths(splits_dir / "test.csv", root)

    pool_dataset = FilteredImageFolder(str(pool_dir), transform=train_tf, excluded_paths=excluded_pool_paths)
    class_to_idx = pool_dataset.class_to_idx

    gold_train = CsvImageDataset(splits_dir / "train.csv", class_to_idx, train_tf, root)
    gold_val = CsvImageDataset(splits_dir / "val.csv", class_to_idx, train_tf, root)
    gold_test = CsvImageDataset(splits_dir / "test.csv", class_to_idx, val_tf, root)

    train_dataset = ConcatDataset([pool_dataset, gold_train, gold_val])
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check pool/splits paths.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    device = get_device()
    model = build_model(num_classes=len(pool_dataset.classes)).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, int | float]] = []
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
        metrics_rows.append({"epoch": epoch, "train_loss": train_loss})
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f}")

    ckpt_path = output_dir / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": pool_dataset.classes,
            "seed": args.seed,
            "train_size": len(train_dataset),
            "pool_size_used": len(pool_dataset),
            "gold_train_size": len(gold_train),
            "gold_val_size": len(gold_val),
            "gold_test_size_held_out": len(gold_test),
            "notes": "Trained on pool (excluding gold splits) + gold train + gold val. Gold test held out.",
        },
        ckpt_path,
    )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary = {
        "checkpoint": str(ckpt_path),
        "metrics_csv": str(metrics_path),
        "pool_dir": str(pool_dir),
        "gold_splits_dir": str(splits_dir),
        "seed": args.seed,
        "train_size": len(train_dataset),
        "pool_size_used": len(pool_dataset),
        "gold_train_size": len(gold_train),
        "gold_val_size": len(gold_val),
        "gold_test_size_held_out": len(gold_test),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
