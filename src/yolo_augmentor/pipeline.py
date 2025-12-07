"""
pipeline.py
-----------
Automated end-to-end pipeline for dataset preparation.

Flow:
1. Scan
2. Convert → YOLO
3. Repair labels
4. Split train/val/test
5. Augment dataset (from augment config)
"""

import yaml
from pathlib import Path

from data.scan_dataset import scan_dataset
from data.convert_to_yolo import convert_to_yolo
from data.split_dataset import split_dataset
from data.repair_labels import repair_labels
from validators import validate_pipeline_config, validate_yolo_structure

# Import augmentor
from aug.augment_dataset import YOLOAugmenterV2


def run_pipeline(config_path: str):

    cfg = validate_pipeline_config(config_path)
    dataset = cfg["dataset"]
    steps = cfg["steps"]
    opt = cfg.get("options", {})
    seed = opt.get("seed", 42)

    input_dir = dataset["input_dir"]
    workspace = Path(dataset["workspace_dir"])
    output_dir = Path(dataset["output_dir"])

    workspace.mkdir(exist_ok=True)

    # ------------------ 1. SCAN ------------------
    if steps.get("scan", True):
        print("\n[1] Scanning dataset...")
        scan_result = scan_dataset(input_dir)
        print(scan_result)

    # ------------------ 2. CONVERT ------------------
    if steps.get("convert_to_yolo", True):
        print("\n[2] Converting dataset to YOLO format...")
        convert_to_yolo(input_dir, workspace, copy=opt.get("copy_instead_of_move", True))

    # ------------------ 3. REPAIR LABELS ------------------
    if steps.get("repair_labels", True):
        print("\n[3] Repairing YOLO labels...")
        repair_labels(str(workspace / "labels"),
                      backup=opt.get("backup_labels_before_repair", True))

    # ------------------ 4. SPLIT ------------------
    if steps["split"]["enabled"]:
        print("\n[4] Splitting dataset...")
        split_dataset(str(workspace),
                      str(workspace / "split"),
                      steps["split"]["train"],
                      steps["split"]["val"],
                      steps["split"]["test"],
                      seed)

    # ------------------ 5. AUGMENT ------------------
    if steps["augment"]["enabled"]:
        print("\n[5] Augmentation running...")
        aug = YOLOAugmenterV2(steps["augment"]["config"])
        aug.run()
        print("[OK] Augmentation complete")

    print("\n[OK] Pipeline Done Successfully.")
    print(f"Output prepared at → {output_dir}\n")
