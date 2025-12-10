"""
Dataset Scanner
----------------
Scans an input dataset directory and detects structure type, 
counts images/labels, missing pairs, and reports dataset health.

Supports:
- Flat YOLO style (images/ and labels/)
- Nested/messy folder structures
- Partial / missing labels detection
"""

import os
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def find_files(root: Path, extensions: List[str]) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in extensions]


def classify_structure(image_dirs: set, label_dirs: set) -> str:
    if len(image_dirs) == 1 and len(label_dirs) == 1:
        return "flat_yolo"
    elif len(image_dirs) > 1 or len(label_dirs) > 1:
        return "nested"
    return "unknown"


def scan_dataset(dataset_path: str) -> Dict:
    root = Path(dataset_path)

    images = find_files(root, IMAGE_EXTS)
    labels = find_files(root, [".txt"])
    label_map = {f.stem: f for f in labels}

    # Build a function to check if label file has valid YOLO lines
    def has_valid_label(file: Path) -> bool:
        try:
            lines = file.read_text().strip().split("\n")
        except:
            return False

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    float(parts[1]); float(parts[2])
                    float(parts[3]); float(parts[4])
                    return True
                except ValueError:
                    continue
        return False

    # NEW: count only files with valid labels
    valid_label_files = [f for f in labels if has_valid_label(f)]

    missing_labels = [img for img in images if img.stem not in label_map]
    image_dirs = {img.parent for img in images}
    label_dirs = {lbl.parent for lbl in valid_label_files}

    result = {
        "total_images": len(images),
        "total_labels": len(valid_label_files),     # <-- FIXED
        "missing_pairs": len(missing_labels),
        "missing_label_files": [str(i) for i in missing_labels[:20]],
        "structure_type": classify_structure(image_dirs, label_dirs),
        "image_dirs": list(map(str, image_dirs)),
        "label_dirs": list(map(str, label_dirs)),
        "has_problems": len(missing_labels) > 0
    }

    return result



if __name__ == "__main__":
    import pprint
    path = input("Enter dataset directory: ")
    pprint.pprint(scan_dataset(path))