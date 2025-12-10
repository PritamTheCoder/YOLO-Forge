"""
repair_labels.py
----------------
Label repair utility for YOLO annotation files.

Capabilities:
 - Fix malformed lines & numeric parsing issues
 - Clip bbox values to [0,1] range
 - Remove empty/invalid bboxes (zero/negative size)
 - Rewrite cleaned labels safely
 - Optional backup original labels

Usage example:
    repair_labels("dataset_yolo/labels", backup=True)

Integrates before augmentation or training.
"""

from pathlib import Path
from typing import List


def _clean_line(line: str):
    """Parses a YOLO line → returns (cls,x,y,w,h) or None if invalid."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    try:
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
    except ValueError:
        return None

    # clip values into valid range
    x, y = max(0, min(1, x)), max(0, min(1, y))
    w, h = max(0, min(1, w)), max(0, min(1, h))

    # Coordinates must be strictly inside (0,1)
    if not (0 < x < 1 and 0 < y < 1):
        return None

    # width/height must be positive AND ensure box doesn't exceed bounds
    if not (0 < w < 1 and 0 < h < 1):
        return None

    return cls, x, y, w, h


def repair_labels(label_dir: str, backup: bool = True) -> None:
    label_dir = Path(label_dir)
    files = list(label_dir.glob("*.txt"))

    fixed_count, removed_files = 0, 0

    for file in files:
        lines = file.read_text().strip().split("\n")

        cleaned = []
        for line in lines:
            parsed = _clean_line(line)
            if parsed:
                cls, x, y, w, h = parsed
                cleaned.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # no valid labels left → remove file
        if len(cleaned) == 0:
            removed_files += 1
            file.unlink()
            continue

        if backup:
            file.with_suffix(".txt.bak").write_text("\n".join(lines))

        file.write_text("\n".join(cleaned))
        fixed_count += 1

    print(f"\n[✓] Label repair finished")
    print(f"Files fixed     : {fixed_count}")
    print(f"Files removed   : {removed_files} (no valid labels)\n")


if __name__ == "__main__":
    repair_labels("dataset_yolo/labels")