"""
Dataset Converter → Standard YOLO Format
-----------------------------------------
Converts messy/nested dataset structures into standard YOLO format:

output/
 ├── images/
 └── labels/

Features:
- Recursively searches for images + txt labels
- Moves or copies into proper YOLO flat layout
- Ensures filename pairing consistency
"""

import shutil
from pathlib import Path
from typing import Optional

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def is_image(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTS


def convert_to_yolo(input_dir: str,
                    output_dir: str,
                    copy: bool = True) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    img_out = output_path / "images"
    lbl_out = output_path / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images = [f for f in input_path.rglob("*") if is_image(f)]
    labels = {f.stem: f for f in input_path.rglob("*.txt")}

    moved, missing = 0, 0

    for img in images:
        label = labels.get(img.stem)
        img_dest = img_out / img.name

        if copy:
            shutil.copy(img, img_dest)
        else:
            shutil.move(img, img_dest)

        if label:
            lbl_dest = lbl_out / f"{img.stem}.txt"
            shutil.copy(label, lbl_dest) if copy else shutil.move(label, lbl_dest)
            moved += 1
        else:
            missing += 1

    print(f"[OK] Conversion complete.")
    print(f"Images found: {len(images)}")
    print(f"Images paired with labels: {moved}")
    print(f"Missing label files: {missing}")


if __name__ == "__main__":
    inp = input("Input dataset path: ")
    out = input("Output YOLO dataset path: ")
    convert_to_yolo(inp, out)