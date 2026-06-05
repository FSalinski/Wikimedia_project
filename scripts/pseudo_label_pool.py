import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


INSIDE_PROMPTS = [
    "an indoor scene",
    "inside a building",
    "a room interior",
    "indoors",
]
OUTSIDE_PROMPTS = [
    "an outdoor scene",
    "outside in nature",
    "outdoors",
    "a landscape",
    "a street outdoors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-label the pool with CLIP.")
    parser.add_argument("--pool_dir", default="pool")
    parser.add_argument("--splits_dir", default="pool/splits")
    parser.add_argument("--train_dir", default="train_set")
    parser.add_argument("--val_dir", default="val_set")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene_prob", type=float, default=0.75)
    parser.add_argument("--scene_margin", type=float, default=0.30)
    parser.add_argument("--max_images", type=int, default=0)
    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def project_features(pooled: torch.Tensor, projection: torch.nn.Module) -> torch.Tensor:
    in_features = projection.in_features
    out_features = projection.out_features
    if pooled.shape[-1] == in_features:
        return projection(pooled)
    if pooled.shape[-1] == out_features:
        return pooled
    raise ValueError(
        f"Unexpected pooled shape {pooled.shape} for projection {in_features}->{out_features}"
    )


def encode_text(processor: CLIPProcessor, model: CLIPModel, prompts: list[str], device: str) -> torch.Tensor:
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.text_model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs[1]
        features = project_features(pooled, model.text_projection)
    return torch.nn.functional.normalize(features, dim=-1)


def classify(
    processor: CLIPProcessor,
    model: CLIPModel,
    image: Image.Image,
    class_text_features: dict[str, torch.Tensor],
    device: str,
) -> dict[str, float]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs[1]
        image_features = project_features(pooled, model.visual_projection)
    image_features = torch.nn.functional.normalize(image_features, dim=-1)

    logit_scale = model.logit_scale.exp()
    scores = {}
    for label, text_features in class_text_features.items():
        logits = logit_scale * image_features @ text_features.T
        scores[label] = logits.mean(dim=-1).item()

    labels = list(scores.keys())
    logits = torch.tensor([scores[label] for label in labels], device=image_features.device)
    probs = torch.softmax(logits, dim=0).tolist()
    return {label: prob for label, prob in zip(labels, probs)}


def read_split(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {str(Path(row["path"]).resolve()) for row in reader}


def write_labels(csv_path: Path, items: list[tuple[Path, str, float]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "score"])
        for path, label, score in items:
            writer.writerow([str(path), label, f"{score:.4f}"])


def copy_items(out_dir: Path, items: list[tuple[Path, str, float]]) -> None:
    for path, label, _ in items:
        target_dir = out_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        shutil.copy2(path, target_path)


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    pool_dir = Path(args.pool_dir)
    splits_dir = Path(args.splits_dir)
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)

    excluded = set()
    excluded |= read_split(splits_dir / "train.csv")
    excluded |= read_split(splits_dir / "val.csv")
    excluded |= read_split(splits_dir / "test.csv")

    pool_images = list((pool_dir / "inside").glob("*.jpg")) + list((pool_dir / "outside").glob("*.jpg"))
    pool_images = [p for p in pool_images if str(p.resolve()) not in excluded]

    if args.max_images > 0:
        pool_images = pool_images[: args.max_images]

    rng = random.Random(args.seed)
    rng.shuffle(pool_images)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_features = {
        "inside": encode_text(processor, model, INSIDE_PROMPTS, device),
        "outside": encode_text(processor, model, OUTSIDE_PROMPTS, device),
    }

    labeled: dict[str, list[tuple[Path, float]]] = {"inside": [], "outside": []}

    for path in tqdm(pool_images, desc="Pseudo-labeling"):
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue

        probs = classify(processor, model, image, text_features, device)
        inside_prob = probs["inside"]
        outside_prob = probs["outside"]
        max_prob = max(inside_prob, outside_prob)
        margin = abs(inside_prob - outside_prob)

        if max_prob < args.scene_prob:
            continue
        if margin < args.scene_margin:
            continue

        label = "inside" if inside_prob >= outside_prob else "outside"
        score = max_prob
        labeled[label].append((path, score))

    train_items: list[tuple[Path, str, float]] = []
    val_items: list[tuple[Path, str, float]] = []

    for label, items in labeled.items():
        rng.shuffle(items)
        split_idx = int(len(items) * args.train_ratio)
        train_items.extend([(p, label, score) for p, score in items[:split_idx]])
        val_items.extend([(p, label, score) for p, score in items[split_idx:]])

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    copy_items(train_dir, train_items)
    copy_items(val_dir, val_items)

    write_labels(train_dir / "labels.csv", train_items)
    write_labels(val_dir / "labels.csv", val_items)

    print(f"Train pseudo-labeled: {len(train_items)}")
    print(f"Val pseudo-labeled: {len(val_items)}")


if __name__ == "__main__":
    main()
