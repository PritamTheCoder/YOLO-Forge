"""
pipeline.py 
-----------
Automated end-to-end pipeline for dataset preparation.

Flow:
1. Scan
2. Convert → YOLO
3. Repair labels
4. Augment dataset (to TEMP location)
5. Split train/val/test (from augmented TEMP data to final output)
"""

import yaml
import shutil
from pathlib import Path
import logging

from .data.scan_dataset import scan_dataset
from .data.convert_to_yolo import convert_to_yolo
from .data.split_dataset import split_dataset
from .data.repair_labels import repair_labels
from .validators import validate_pipeline_config
from .reports.report_generator import generate_report

# Import augmentor
from .aug.augment_dataset import YOLOAugmenterV2


def run_pipeline(config_path: str):
    """Execute the complete data preparation pipeline with automatic cleanup."""
    
    cfg = validate_pipeline_config(config_path)
    dataset = cfg["dataset"]
    steps = cfg["steps"]
    opt = cfg.get("options", {})
    logging_cfg = cfg.get("logging", {})
    seed = opt.get("seed", 42)

    input_dir = dataset["input_dir"]
    workspace = Path(dataset["workspace_dir"])
    output_dir = Path(dataset["output_dir"])
    
    # Ensure logging directory exists
    if logging_cfg.get("save_logs", True):
        log_dir = Path(logging_cfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Logging directory ready: {log_dir}")

    workspace.mkdir(exist_ok=True)
    
    # Track if we should cleanup workspace
    auto_cleanup = opt.get("auto_cleanup_workspace", False)

    try:
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

        # ------------------ 4. AUGMENT ------------------
        augment_output_dir = None
        if steps.get("augment", {}).get("enabled", False):
            print("\n[4] Augmentation running...")
            aug_config_path = steps["augment"]["config"]
            
            # Load augmentation config to get output directory
            with open(aug_config_path, 'r', encoding='utf-8') as f:
                aug_cfg = yaml.safe_load(f)
            
            # CRITICAL FIX: Get the augmentation output directory
            augment_output_dir = Path(aug_cfg["dataset"]["output_images_dir"]).parent
            
            # Check if augmentation output conflicts with final split output
            if augment_output_dir.resolve() == output_dir.resolve() or \
               augment_output_dir.resolve() in [output_dir.resolve() / split_name 
                                                for split_name in ['train', 'val', 'test']]:
                print("\n[WARNING] Augmentation output directory conflicts with split output!")
                print(f"  Augment output: {augment_output_dir}")
                print(f"  Final output: {output_dir}")
                print("  Creating temporary augmentation directory...")
                
                # Use a temp directory inside workspace
                temp_aug_dir = workspace / "augmented_temp"
                temp_aug_dir.mkdir(exist_ok=True)
                
                # Modify augmentation config temporarily
                aug_cfg["dataset"]["output_images_dir"] = str(temp_aug_dir / "images")
                aug_cfg["dataset"]["output_labels_dir"] = str(temp_aug_dir / "labels")
                
                # Save modified config to temporary file
                temp_config_path = workspace / "temp_aug_config.yaml"
                with open(temp_config_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(aug_cfg, f)
                
                # Use temporary config
                aug_config_path = str(temp_config_path)
                augment_output_dir = temp_aug_dir
                print(f"  Using temporary augmentation directory: {augment_output_dir}")
            
            aug = YOLOAugmenterV2(aug_config_path)
            aug.run()
            print("[OK] Augmentation complete")
        
        # ------------------ 5. SPLIT ------------------
        if steps.get("split", {}).get("enabled", False):
            print("\n[5] Splitting dataset...")
            
            # Determine input directory for split
            if augment_output_dir and augment_output_dir.exists():
                split_input = str(augment_output_dir)
                print(f"    Using augmented data from: {split_input}")
            else:
                split_input = str(workspace)
                print(f"    Using workspace data from: {split_input}")
            
            # Verify split input is different from output
            split_input_resolved = Path(split_input).resolve()
            output_dir_resolved = output_dir.resolve()
            
            if split_input_resolved == output_dir_resolved or \
               split_input_resolved in [output_dir_resolved / name for name in ['train', 'val', 'test']]:
                raise ValueError(
                    f"Split input ({split_input}) cannot be the same as or inside output directory ({output_dir}). "
                    f"This would cause files to be copied to themselves!"
                )
            
            # Split into final output directory
            split_dataset(
                split_input,
                str(output_dir),
                steps["split"]["train"],
                steps["split"]["val"],
                steps["split"]["test"],
                seed,
                copy=True
            )
            print("[OK] Dataset split successful")
        
        # --------------------- 6. FINAL REPORT ---------------------
        print("\n[6] Generating final dataset report...")
        
        report_save_dir = Path(output_dir) / "report"
        report_save_dir.mkdir(parents=True, exist_ok=True)
        
        report = generate_report(
            yolo_root=str(output_dir/"train"),
            out_dir=str(report_save_dir),
            samples=30,
        )
        
        print("\n Report generated successfully!")
        print("\n Dataset Processing Completed!")
        print("Report Summary:")
        print(f"  -> HTML Report: {report['html']}")
        print(f"  -> Stats JSON: {report['stats_json']}\n")
        print(f"   - Output Folder: {report['report_dir']}")
        print(f"   - Plots: {len(report['plots'])}")
        print(f"   - Sample Tiles: {len(report['sample_images'])}")
          
        # =============== PIPELINE COMPLETED ==========================
        print("\n[OK] Pipeline Done Successfully.")
        print(f"Final dataset ready at → {output_dir}\n")
    
    finally:
        # Cleanup workspace if enabled
        if auto_cleanup and workspace.exists():
            print(f"\n[CLEANUP] Removing workspace directory: {workspace}")
            try:
                shutil.rmtree(workspace)
                print("[OK] Workspace cleaned up successfully")
            except Exception as e:
                print(f"[WARNING] Failed to cleanup workspace: {e}")