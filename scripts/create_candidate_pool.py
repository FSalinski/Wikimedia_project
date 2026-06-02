import argparse
import csv
import hashlib
import math
import os
from io import BytesIO

import requests
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


PHOTO_PROMPTS = [
    "a photograph",
    "a photo of a real scene",
    "a real-life photo",
    "a camera photo",
]
NONPHOTO_PROMPTS = [
    "an illustration",
    "a diagram",
    "a drawing",
    "a cartoon",
    "a screenshot",
    "a map",
    "a logo",
    "a chart",
    "a 3d render",
]

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
    parser = argparse.ArgumentParser(description="Create a candidate pool from WIT.")
    parser.add_argument("--dataset_name", default="wikimedia/wit_base")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="pool")
    parser.add_argument("--target_total", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--min_short_side", type=int, default=256)
    parser.add_argument("--max_aspect_ratio", type=float, default=3.0)
    parser.add_argument("--min_entropy", type=float, default=3.0)
    parser.add_argument("--min_bytes", type=int, default=5000)
    parser.add_argument("--photo_prob", type=float, default=0.70)
    parser.add_argument("--photo_margin", type=float, default=0.35)
    parser.add_argument("--scene_prob", type=float, default=0.60)
    parser.add_argument("--scene_margin", type=float, default=0.20)
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def image_entropy(image: Image.Image) -> float:
    gray = image.convert("L")
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in hist:
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def find_url(sample: dict) -> str | None:
    for key, value in sample.items():
        if isinstance(value, str) and value.startswith("http") and "image" in key:
            return value
    for key, value in sample.items():
        if isinstance(value, str) and value.startswith("http"):
            return value
    return None


def load_image(sample: dict, timeout: int) -> tuple[Image.Image | None, str | None, bytes | None]:
    img = sample.get("image")
    if isinstance(img, Image.Image):
        return img.convert("RGB"), None, None
    if isinstance(img, dict) and img.get("bytes"):
        try:
            pil_img = Image.open(BytesIO(img["bytes"])).convert("RGB")
            return pil_img, None, img["bytes"]
        except Exception:
            return None, None, None

    url = find_url(sample)
    if not url:
        return None, None, None

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None, url, None
        if len(response.content) < 1:
            return None, url, None
        pil_img = Image.open(BytesIO(response.content)).convert("RGB")
        return pil_img, url, response.content
    except Exception:
        return None, url, None


def quality_pass(
    image: Image.Image,
    min_short_side: int,
    max_aspect_ratio: float,
    min_entropy: float,
) -> bool:
    width, height = image.size
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side < min_short_side:
        return False
    if long_side / max(short_side, 1) > max_aspect_ratio:
        return False
    if image_entropy(image) < min_entropy:
        return False
    return True


def encode_text(processor: CLIPProcessor, model: CLIPModel, prompts: list[str], device: str) -> torch.Tensor:
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.text_model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs[1]
        features = project_features(pooled, model.text_projection)
    return torch.nn.functional.normalize(features, dim=-1)


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


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    ensure_dir(args.output_dir)
    inside_dir = os.path.join(args.output_dir, "inside")
    outside_dir = os.path.join(args.output_dir, "outside")
    ensure_dir(inside_dir)
    ensure_dir(outside_dir)

    labels_path = os.path.join(args.output_dir, "labels.csv")
    labels_file = open(labels_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(labels_file)
    writer.writerow(
        [
            "image_id",
            "url",
            "width",
            "height",
            "photo_prob",
            "inside_prob",
            "outside_prob",
            "weak_label",
        ]
    )

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text_features = {
        "photo": encode_text(processor, model, PHOTO_PROMPTS, device),
        "nonphoto": encode_text(processor, model, NONPHOTO_PROMPTS, device),
        "inside": encode_text(processor, model, INSIDE_PROMPTS, device),
        "outside": encode_text(processor, model, OUTSIDE_PROMPTS, device),
    }

    dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    target_per_class = args.target_total // 2
    counts = {"inside": 0, "outside": 0}
    seen_hashes: set[str] = set()
    saved = 0

    for sample in tqdm(dataset, desc="Building candidate pool"):
        if counts["inside"] >= target_per_class and counts["outside"] >= target_per_class:
            break

        image, url, raw_bytes = load_image(sample, timeout=args.timeout)
        if image is None:
            continue

        if raw_bytes is None:
            raw_bytes = image.tobytes()
        if len(raw_bytes) < args.min_bytes:
            continue

        image_hash = hashlib.sha1(raw_bytes).hexdigest()
        if image_hash in seen_hashes:
            continue
        seen_hashes.add(image_hash)

        if not quality_pass(image, args.min_short_side, args.max_aspect_ratio, args.min_entropy):
            continue

        photo_probs = classify(
            processor,
            model,
            image,
            {"photo": text_features["photo"], "nonphoto": text_features["nonphoto"]},
            device,
        )
        photo_prob = photo_probs["photo"]
        nonphoto_prob = photo_probs["nonphoto"]
        if photo_prob < args.photo_prob:
            continue
        if (photo_prob - nonphoto_prob) < args.photo_margin:
            continue

        scene_probs = classify(
            processor,
            model,
            image,
            {"inside": text_features["inside"], "outside": text_features["outside"]},
            device,
        )
        inside_prob = scene_probs["inside"]
        outside_prob = scene_probs["outside"]
        max_prob = max(inside_prob, outside_prob)
        margin = abs(inside_prob - outside_prob)
        if max_prob < args.scene_prob:
            continue
        if margin < args.scene_margin:
            continue

        weak_label = "inside" if inside_prob >= outside_prob else "outside"
        if counts[weak_label] >= target_per_class:
            continue

        image_id = sample.get("image_id") or sample.get("id") or f"img_{saved:07d}"
        file_name = f"{image_id}.jpg"
        save_dir = inside_dir if weak_label == "inside" else outside_dir
        save_path = os.path.join(save_dir, file_name)
        try:
            image.save(save_path, format="JPEG", quality=90)
        except Exception:
            continue

        writer.writerow(
            [
                image_id,
                url or "",
                image.size[0],
                image.size[1],
                f"{photo_prob:.4f}",
                f"{inside_prob:.4f}",
                f"{outside_prob:.4f}",
                weak_label,
            ]
        )
        labels_file.flush()

        counts[weak_label] += 1
        saved += 1

    labels_file.close()
    print("Done.")
    print(f"Saved: {saved}")
    print(f"Inside: {counts['inside']}, Outside: {counts['outside']}")


if __name__ == "__main__":
    main()
