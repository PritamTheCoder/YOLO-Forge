"""
validators.py
--------------
Sanity validators for dataset + config + annotations.

Used before converting, splitting, or augmenting.
"""

from pathlib import Path
import yaml


# ============================== #
# Dataset validation utilities   #
# ============================== #

def validate_yolo_structure(path: str) -> bool:
    path = Path(path)
    img = path / "images"
    lbl = path / "labels"

    valid = img.exists() and lbl.exists()

    if not valid:
        print("[ERROR] Invalid YOLO dataset structure.")
        print("Expected: dataset/images & dataset/labels")
    else:
        print("[OK] YOLO structure verified")

    return valid


def validate_bbox_format(line: str) -> bool:
    parts = line.split()
    if len(parts) != 5:
        return False

    try:
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
    except:
        return False

    return (0 <= x <= 1) and (0 <= y <= 1) and w > 0 and h > 0


def validate_label_file(file: Path) -> bool:
    return all(validate_bbox_format(l) for l in file.read_text().splitlines())


# ============================== #
# Config validation              #
# ============================== #

def validate_pipeline_config(config_path: str) -> dict:
    """Load + verify YAML config. Raises errors if malformed."""
    try:
        config = yaml.safe_load(open(config_path))
    except Exception:
        raise ValueError("Invalid YAML format in pipeline config")

    required_root_keys = ["dataset", "steps"]
    for k in required_root_keys:
        if k not in config:
            raise ValueError(f"Missing required config section: {k}")

    print("[OK] Config structure looks valid")
    return config


# ============================== #
# Quick batch label validation   #
# ============================== #

def validate_labels_dir(label_dir: str):
    label_dir = Path(label_dir)
    invalid_files = []

    for f in label_dir.glob("*.txt"):
        if not validate_label_file(f):
            invalid_files.append(f.name)

    if invalid_files:
        print("[ERROR] Invalid annotations found:")
        for f in invalid_files[:10]:
            print("  ", f)
        print(f"Total invalid files: {len(invalid_files)}")
        return False

    print("[OK] All label files valid")
    return True


if __name__ == "__main__":
    validate_yolo_structure("dataset")
    validate_labels_dir("dataset/labels")
