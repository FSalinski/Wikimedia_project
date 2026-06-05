import argparse
import csv
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split gold labels into train/val/test.")
    parser.add_argument("--input_csv", default="labels_gold.csv")
    parser.add_argument("--out_dir", default="pool/splits")
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy_images", action="store_true")
    return parser.parse_args()


def read_labels(csv_path: Path) -> list[tuple[Path, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append((Path(row["path"]), row["label"]))
    return rows


def allocate_counts(total: int, label_counts: dict[str, int]) -> dict[str, int]:
    if total <= 0:
        return {label: 0 for label in label_counts}
    total_count = sum(label_counts.values())
    if total_count == 0:
        return {label: 0 for label in label_counts}

    raw = {label: (count / total_count) * total for label, count in label_counts.items()}
    base = {label: int(raw[label]) for label in label_counts}
    remainder = total - sum(base.values())
    if remainder > 0:
        frac = sorted(label_counts.keys(), key=lambda l: raw[l] - base[l], reverse=True)
        for label in frac[:remainder]:
            base[label] += 1
    return base


def write_split(csv_path: Path, items: list[tuple[Path, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for path, label in items:
            writer.writerow([str(path), label])


def copy_split(out_dir: Path, split_name: str, items: list[tuple[Path, str]]) -> None:
    for path, label in items:
        target_dir = out_dir / split_name / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        shutil.copy2(path, target_path)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_labels(input_csv)
    if not rows:
        print("No labels found.")
        return

    by_label: dict[str, list[Path]] = {}
    for path, label in rows:
        if not path.exists():
            continue
        by_label.setdefault(label, []).append(path)

    rng = random.Random(args.seed)
    for paths in by_label.values():
        rng.shuffle(paths)

    label_counts = {label: len(paths) for label, paths in by_label.items()}
    test_counts = allocate_counts(args.test_size, label_counts)

    test_items: list[tuple[Path, str]] = []
    remaining: dict[str, list[Path]] = {}
    for label, paths in by_label.items():
        take = min(test_counts.get(label, 0), len(paths))
        test_paths = paths[:take]
        rest_paths = paths[take:]
        test_items.extend([(p, label) for p in test_paths])
        remaining[label] = rest_paths

    remaining_counts = {label: len(paths) for label, paths in remaining.items()}
    val_counts = allocate_counts(args.val_size, remaining_counts)

    val_items: list[tuple[Path, str]] = []
    train_items: list[tuple[Path, str]] = []
    for label, paths in remaining.items():
        take = min(val_counts.get(label, 0), len(paths))
        val_paths = paths[:take]
        train_paths = paths[take:]
        val_items.extend([(p, label) for p in val_paths])
        train_items.extend([(p, label) for p in train_paths])

    write_split(out_dir / "train.csv", train_items)
    write_split(out_dir / "val.csv", val_items)
    write_split(out_dir / "test.csv", test_items)

    if args.copy_images:
        copy_split(out_dir, "train", train_items)
        copy_split(out_dir, "val", val_items)
        copy_split(out_dir, "test", test_items)

    print(f"Train: {len(train_items)}")
    print(f"Val: {len(val_items)}")
    print(f"Test: {len(test_items)}")


if __name__ == "__main__":
    main()
