"""
Dataset Splitter — Train/Val/Test
---------------------------------
Splits a YOLO formatted dataset into train/val/test folders.

Input Format Expected:
dataset/
 ├── images/
 └── labels/

Usage Example:
split_dataset("dataset", "dataset_split", train=0.8, val=0.1, test=0.1)
"""

from pathlib import Path
import shutil
import random
from typing import Optional, Tuple


def split_dataset(input_dir: str,
                  output_dir: str,
                  train: float = 0.8,
                  val: float = 0.1,
                  test: float = 0.1,
                  seed: int = 42,
                  copy: bool = True) -> None:

    assert abs((train + val + test) - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    random.seed(seed)
    input_path = Path(input_dir)
    img_dir = input_path / "images"
    lbl_dir = input_path / "labels"

    assert img_dir.exists() and lbl_dir.exists(), "Input must contain images/ and labels/"

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    random.shuffle(images)

    total = len(images)
    n_train = int(total * train)
    n_val = int(total * val)
    n_test = total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split_name, imgs in splits.items():
        out_img = Path(output_dir) / split_name / "images"
        out_lbl = Path(output_dir) / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img in imgs:
            lbl = lbl_dir / f"{img.stem}.txt"
            (shutil.copy if copy else shutil.move)(img, out_img / img.name)

            if lbl.exists():
                (shutil.copy if copy else shutil.move)(lbl, out_lbl / lbl.name)

    print(f"\n[✓] Dataset splitting complete.")
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    split_dataset("dataset_yolo", "dataset_split", 0.8, 0.1, 0.1)