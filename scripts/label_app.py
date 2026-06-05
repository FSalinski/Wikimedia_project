import argparse
import csv
import os
import random
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual labeling app for indoor/outdoor.")
    parser.add_argument("--input_dir", default="pool", help="Folder with inside/ and outside/ subfolders.")
    parser.add_argument("--output_csv", default="labels_gold.csv")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_size", type=int, default=900)
    return parser.parse_args()


def load_images(input_dir: Path) -> list[Path]:
    inside = list((input_dir / "inside").glob("*.jpg"))
    outside = list((input_dir / "outside").glob("*.jpg"))
    return inside + outside


def read_existing_labels(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["path"]: row["label"] for row in reader}


def save_label(csv_path: Path, path: Path, label: str) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["path", "label"])
        writer.writerow([str(path), label])


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    image_paths = load_images(input_dir)
    random.Random(args.seed).shuffle(image_paths)

    existing = read_existing_labels(output_csv)
    queue = [p for p in image_paths if str(p) not in existing]
    queue = queue[: args.limit]

    if not queue:
        print("No images to label.")
        return

    index = 0

    root = tk.Tk()
    root.title("Indoor vs Outdoor Labeling")

    image_label = tk.Label(root)
    image_label.pack()

    info_var = tk.StringVar()
    info_label = tk.Label(root, textvariable=info_var)
    info_label.pack()

    status_var = tk.StringVar()
    status_label = tk.Label(root, textvariable=status_var)
    status_label.pack()

    def update_image() -> None:
        nonlocal index
        if index >= len(queue):
            status_var.set("Done. You can close the window.")
            return

        path = queue[index]
        img = Image.open(path).convert("RGB")
        img.thumbnail((args.max_size, args.max_size))
        tk_img = ImageTk.PhotoImage(img)
        image_label.configure(image=tk_img)
        image_label.image = tk_img
        info_var.set(f"{index + 1}/{len(queue)} - {path.name}")

    def set_label(label: str) -> None:
        nonlocal index
        if index >= len(queue):
            return
        path = queue[index]
        save_label(output_csv, path, label)
        index += 1
        update_image()

    def skip() -> None:
        nonlocal index
        if index >= len(queue):
            return
        index += 1
        update_image()

    def back() -> None:
        nonlocal index
        if index <= 0:
            return
        index -= 1
        update_image()

    button_frame = tk.Frame(root)
    button_frame.pack(pady=8)

    tk.Button(button_frame, text="Inside (I)", command=lambda: set_label("inside")).grid(row=0, column=0, padx=4)
    tk.Button(button_frame, text="Outside (O)", command=lambda: set_label("outside")).grid(row=0, column=1, padx=4)
    tk.Button(button_frame, text="Skip (S)", command=skip).grid(row=0, column=2, padx=4)
    tk.Button(button_frame, text="Back (B)", command=back).grid(row=0, column=3, padx=4)

    root.bind("i", lambda _: set_label("inside"))
    root.bind("o", lambda _: set_label("outside"))
    root.bind("s", lambda _: skip())
    root.bind("b", lambda _: back())

    status_var.set(f"Output: {output_csv}")
    update_image()
    root.mainloop()


if __name__ == "__main__":
    main()
