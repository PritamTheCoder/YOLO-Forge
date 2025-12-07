"""
YOLO Dataset Augmentation System 
 1. Proper custom transform instantiation (factory functions)
 2. Better error handling and validation logging
 3. Fixed coordinate system issues
 4. Comprehensive discard tracking by reason
"""

import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from typing import List, Dict, Tuple, Optional, Callable
import logging
from datetime import datetime
import random
import shutil
from custom_transforms import (
    BboxMultiBlurAndShearTransform,
    BboxExtremeShearOcclude,
    NearBboxExtremeBrighten,
    ConcentratedNoiseTransform,
    BallBlendAndShapeBiasTransform,
    BallPixelLevelOcclusion,
    GradientPatchTransform,
    BboxGaussianOccludeShearTransform
)


class CustomTransformRegistry:
    """Registry for custom augmentation transforms."""

    def __init__(self, logger):
        self.transforms = {}
        self.logger = logger
        self._register_builtin_transforms()

    def _register_builtin_transforms(self):
        """Register built-in custom transforms as CLASS TYPES (not instances)."""
        # Store the class itself, instantiate later with params
        self.register('BboxMultiBlurAndShearTransform', BboxMultiBlurAndShearTransform)
        self.register('BboxExtremeShearOcclude', BboxExtremeShearOcclude)
        self.register('NearBboxExtremeBrighten', NearBboxExtremeBrighten)
        self.register('ConcentratedNoiseTransform', ConcentratedNoiseTransform)
        self.register('BallBlendAndShapeBiasTransform', BallBlendAndShapeBiasTransform)
        self.register('BallPixelLevelOcclusion', BallPixelLevelOcclusion)
        self.register('GradientPatchTransform', GradientPatchTransform)
        self.register('BboxGaussianOccludeShearTransform', BboxGaussianOccludeShearTransform)
        
        self.logger.info(f"Registered {len(self.transforms)} custom transforms")

    def register(self, name: str, transform_class):
        """Register a transform class."""
        self.transforms[name] = transform_class

    def get(self, name: str):
        """Get transform class by name."""
        return self.transforms.get(name)


class YOLOAugmenterV2:
    """Enhanced augmentation system with comprehensive debugging."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._validate_config()
        self._setup_directories()
        self.custom_registry = CustomTransformRegistry(self.logger)
        
        # Detailed statistics tracking
        self.stats = {
            'total_processed': 0,
            'total_generated': 0,
            'total_attempted': 0,
            'discarded_total': 0,
            'discarded_black_frames': 0,
            'discarded_extreme_brightness': 0,
            'discarded_invalid_bboxes': 0,
            'discarded_transform_failed': 0,
            'discarded_empty_labels': 0,
            'pass_stats': {}
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration with sensible defaults."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        for k in ['dataset', 'validation', 'augment_passes']:
            if k not in config:
                raise ValueError(f"Missing required config key: {k}")

        # More lenient QC defaults
        qc = config.get('quality_control', {})
        qc.setdefault('black_frame_threshold', 0.90)
        qc.setdefault('min_brightness', 3)
        qc.setdefault('max_brightness', 252)
        qc.setdefault('target_discard_rate', 0.08)
        config['quality_control'] = qc
        
        return config

    def _setup_logging(self):
        """Setup comprehensive logging for debugging."""
        log_dir = Path(self.config['dataset'].get('logs_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"augmentation_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 80)
        self.logger.info("YOLO Dataset Augmentation System - FIXED VERSION")
        self.logger.info("=" * 80)

    def _validate_config(self):
        """Validate configuration and warn about strict settings."""
        qc = self.config['quality_control']
        
        if qc['black_frame_threshold'] > 0.95:
            self.logger.warning(f"  black_frame_threshold is very high ({qc['black_frame_threshold']:.2f}). "
                              "May reject too many augmented images!")
        
        if qc['min_brightness'] < 5 or qc['max_brightness'] > 250:
            self.logger.warning(f"  Brightness range [{qc['min_brightness']}, {qc['max_brightness']}] "
                              "is very strict. Consider relaxing to [5, 250]")
        
        val = self.config['validation']
        if val['min_bbox_width'] < 0.005 or val['min_bbox_height'] < 0.005:
            self.logger.warning(f"  Minimum bbox size is very small. May allow invalid detections.")

    def _setup_directories(self):
        """Create output directories."""
        out_img = Path(self.config['dataset']['output_images_dir'])
        out_lbl = Path(self.config['dataset']['output_labels_dir'])
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directories ready: {out_img}, {out_lbl}")

    # ===============================================
    # Validation / QC utilities
    # ===============================================
    def _is_black_frame(self, image):
        """Check if image is mostly black/dark."""
        qc = self.config['quality_control']
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dark_ratio = np.sum(gray < qc['min_brightness']) / gray.size
        is_black = dark_ratio > qc['black_frame_threshold']
        
        if is_black:
            self.logger.debug(f"Black frame detected: {dark_ratio:.2%} dark pixels")
        
        return is_black

    def _has_extreme_brightness(self, image):
        """Check if image has extreme brightness values."""
        qc = self.config['quality_control']
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        is_extreme = mean_brightness < qc['min_brightness'] or mean_brightness > qc['max_brightness']
        
        if is_extreme:
            self.logger.debug(f"Extreme brightness: {mean_brightness:.1f} "
                            f"(valid range: [{qc['min_brightness']}, {qc['max_brightness']}])")
        
        return is_extreme

    def _validate_bboxes(self, bboxes: List[List[float]]) -> bool:
        """Validate bbox dimensions and coordinates."""
        if not bboxes:
            self.logger.debug("No bboxes to validate")
            return False
            
        v = self.config['validation']
        
        for idx, (x, y, w, h) in enumerate(bboxes):
            # Check minimum size
            if w < v['min_bbox_width'] or h < v['min_bbox_height']:
                self.logger.debug(f"Bbox {idx} too small: w={w:.4f}, h={h:.4f}")
                return False
            
            # Check coordinates are in valid range
            tolerance = v['coord_tolerance']
            if not (v['coord_min'] - tolerance <= x <= v['coord_max'] + tolerance):
                self.logger.debug(f"Bbox {idx} x-coord out of range: {x:.4f}")
                return False
            
            if not (v['coord_min'] - tolerance <= y <= v['coord_max'] + tolerance):
                self.logger.debug(f"Bbox {idx} y-coord out of range: {y:.4f}")
                return False
            
            # Check that bbox doesn't extend beyond image
            if x - w/2 < -tolerance or x + w/2 > 1 + tolerance:
                self.logger.debug(f"Bbox {idx} extends beyond image horizontally")
                return False
            
            if y - h/2 < -tolerance or y + h/2 > 1 + tolerance:
                self.logger.debug(f"Bbox {idx} extends beyond image vertically")
                return False
        
        return True

    # ===============================================
    # File utilities
    # ===============================================
    def _load_yolo_labels(self, label_path: Path):
        """Load YOLO format labels with error handling."""
        if not label_path.exists():
            self.logger.debug(f"Label file not found: {label_path}")
            return [], []
        
        bboxes, labels = [], []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        self.logger.warning(f"Invalid label format in {label_path}:{line_num} - "
                                          f"expected 5 values, got {len(parts)}")
                        continue
                    
                    try:
                        labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])
                    except ValueError as e:
                        self.logger.warning(f"Error parsing {label_path}:{line_num} - {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading {label_path}: {e}")
            return [], []
        
        return bboxes, labels

    def _save_yolo_labels(self, path: Path, bboxes, labels):
        """Save YOLO format labels."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for cls, bbox in zip(labels, bboxes):
                    f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")
        except Exception as e:
            self.logger.error(f"Error writing {path}: {e}")

    # ===============================================
    # Augmentation Core
    # ===============================================
    def _build_pipeline(self, pipeline_cfg, pass_type):
        """Build augmentation pipeline (Albumentations or custom)."""
        if pass_type == 'custom':
            return pipeline_cfg  # Return raw config for custom handling
        
        transforms = []
        for t in pipeline_cfg:
            name = t['transform']
            params = t.get('params', {})
            try:
                T = getattr(A, name)
                transforms.append(T(**params))
            except AttributeError:
                self.logger.warning(f"Transform '{name}' not found in Albumentations")
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=self.config['validation']['min_visibility']
            )
        )

    def _apply_custom_pipeline(self, image, bboxes, labels, cfgs):
        """Apply custom transforms pipeline with proper instantiation."""
        out_img = image.copy()
        out_boxes = bboxes[:]
        out_labels = labels[:]
        
        for t_cfg in cfgs:
            name = t_cfg['transform']
            params = t_cfg.get('params', {}).copy()  # Copy to avoid modifying original
            p = params.pop('p', 1.0)  # Extract probability
            
            # Random skip based on probability
            if random.random() > p:
                continue
            
            # Get transform CLASS from registry
            transform_class = self.custom_registry.get(name)
            if not transform_class:
                self.logger.warning(f"Custom transform '{name}' not registered. Skipping.")
                continue
            
            try:
                # Instantiate the transform with parameters
                transform_instance = transform_class(**params)
                
                # Apply the transform
                result = transform_instance.apply(out_img, out_boxes, out_labels)
                
                if result is None:
                    self.logger.debug(f"Transform '{name}' returned None")
                    return None
                
                out_img, out_boxes, out_labels = result
                
            except Exception as e:
                self.logger.error(f"Transform '{name}' failed: {e}", exc_info=True)
                return None
        
        return out_img, out_boxes, out_labels

    def _augment_one(self, img, bboxes, labels, pipeline, pass_type, pass_name):
        """Apply augmentation to one image with comprehensive validation."""
        self.stats['total_attempted'] += 1
        
        try:
            # Apply augmentation
            if pass_type == 'custom':
                result = self._apply_custom_pipeline(img, bboxes, labels, pipeline)
            else:
                result = pipeline(image=img, bboxes=bboxes, class_labels=labels)
                result = (result['image'], result['bboxes'], result['class_labels'])
            
            if not result:
                self.logger.debug(f"[{pass_name}] Transform returned None")
                self.stats['discarded_transform_failed'] += 1
                return None
            
            aug_img, aug_boxes, aug_labels = result
            
            # Validation checks
            if not aug_boxes or not aug_labels:
                self.logger.debug(f"[{pass_name}] Empty labels after augmentation")
                self.stats['discarded_empty_labels'] += 1
                return None
            
            if self._is_black_frame(aug_img):
                self.stats['discarded_black_frames'] += 1
                return None
            
            if self._has_extreme_brightness(aug_img):
                self.stats['discarded_extreme_brightness'] += 1
                return None
            
            if not self._validate_bboxes(aug_boxes):
                self.stats['discarded_invalid_bboxes'] += 1
                return None
            
            return aug_img, aug_boxes, aug_labels
            
        except Exception as e:
            self.logger.error(f"[{pass_name}] Augmentation failed: {e}", exc_info=True)
            self.stats['discarded_transform_failed'] += 1
            return None

    def _float_mult(self, m):
        """Handle fractional multiplier with probabilistic rounding."""
        base = int(m)
        frac = m - base
        return base + (1 if random.random() < frac else 0)

    # ===============================================
    # Main execution
    # ===============================================
    def run(self):
        """Main augmentation pipeline execution."""
        # Load input images
        img_dir = Path(self.config['dataset']['input_images_dir'])
        lbl_dir = Path(self.config['dataset']['input_labels_dir'])
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        imgs = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in img_exts]
        
        if not imgs:
            self.logger.error(f" No images found in {img_dir}")
            return
        
        num_orig = len(imgs)
        self.stats['total_processed'] = num_orig
        self.logger.info(f" Found {num_orig} input images")
        
        # Check labels
        labels_found = sum(1 for img in imgs if (lbl_dir / f"{img.stem}.txt").exists())
        self.logger.info(f" Found {labels_found}/{num_orig} corresponding label files")
        
        if labels_found == 0:
            self.logger.error(f" No label files found in {lbl_dir}! Check your paths.")
            return
        
        # Calculate target
        target_total = self.config['dataset'].get('target_total_images', None)
        if target_total:
            target_total = int(target_total)
            req_gen = target_total - num_orig
            self.logger.info(f" Target: {target_total} total images, need to generate {req_gen}")
        else:
            req_gen = None
            self.logger.info(" Using multipliers directly (no target_total_images set)")
        
        # Auto-compute multipliers if target is set
        if req_gen and req_gen > 0:
            weights = [float(p.get('weight', 1.0)) for p in self.config['augment_passes']]
            w_sum = sum(weights)
            per_pass = [int(round(req_gen * (w / w_sum))) for w in weights]
            delta = req_gen - sum(per_pass)
            
            # Distribute remainder
            i = 0
            while delta != 0:
                per_pass[i % len(per_pass)] += 1 if delta > 0 else -1
                delta = req_gen - sum(per_pass)
                i += 1
                if i > 1000:  # Safety
                    break
            
            for idx, p in enumerate(self.config['augment_passes']):
                m = per_pass[idx] / float(num_orig) if num_orig > 0 else 0
                p['count_multiplier'] = float(max(0.0, m))
                self.logger.info(f"  Pass '{p['name']}': multiplier={m:.3f} (target {per_pass[idx]} images)")
        
        # Copy originals
        out_img = Path(self.config['dataset']['output_images_dir'])
        out_lbl = Path(self.config['dataset']['output_labels_dir'])
        
        self.logger.info(" Copying original images...")
        for img_path in tqdm(imgs, desc="Copying originals"):
            shutil.copy2(img_path, out_img / img_path.name)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)
        
        # Run augmentation passes
        remaining = req_gen if req_gen else float('inf')
        
        for pass_idx, p_cfg in enumerate(self.config['augment_passes'], 1):
            if remaining <= 0:
                break
            
            name = p_cfg['name']
            ptype = p_cfg.get('type', 'standard')
            multiplier = float(p_cfg.get('count_multiplier', 1.0))
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f" Pass {pass_idx}/{len(self.config['augment_passes'])}: {name}")
            self.logger.info(f"   Type: {ptype}, Multiplier: {multiplier:.3f}")
            self.logger.info(f"{'='*60}")
            
            pipeline = self._build_pipeline(p_cfg['pipeline'], ptype)
            gen_this = 0
            disc_this = 0
            
            for img_path in tqdm(imgs, desc=f"Pass: {name}"):
                if remaining <= 0:
                    break
                
                # Load image and labels
                img = cv2.imread(str(img_path))
                if img is None:
                    self.logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                boxes, labels = self._load_yolo_labels(lbl_dir / f"{img_path.stem}.txt")
                
                if not boxes:
                    self.logger.debug(f"No labels for {img_path.stem}, skipping")
                    continue
                
                # Generate augmentations
                n = self._float_mult(multiplier)
                
                for aug_idx in range(n):
                    if remaining <= 0:
                        break
                    
                    result = self._augment_one(img, boxes, labels, pipeline, ptype, name)
                    
                    if not result:
                        disc_this += 1
                        continue
                    
                    aug_img, aug_boxes, aug_labels = result
                    
                    # Save augmented image
                    name_base = f"{img_path.stem}_{name}_{gen_this:04d}"
                    out_path = out_img / f"{name_base}{img_path.suffix}"
                    
                    cv2.imwrite(
                        str(out_path),
                        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    )
                    
                    self._save_yolo_labels(
                        out_lbl / f"{name_base}.txt",
                        aug_boxes,
                        aug_labels
                    )
                    
                    gen_this += 1
                    self.stats['total_generated'] += 1
                    if req_gen:
                        remaining -= 1
            
            self.stats['pass_stats'][name] = {
                'generated': gen_this,
                'discarded': disc_this
            }
            
            self.logger.info(f"[OK] Pass '{name}': {gen_this} generated, {disc_this} discarded")
        
        # Post-augmentation random discard
        disc_rate = self.config['quality_control']['target_discard_rate']
        imgs_out = [f for f in out_img.iterdir() if f.suffix.lower() in img_exts]
        gen_only = [f for f in imgs_out if f.stem not in [x.stem for x in imgs]]
        to_disc = int(len(gen_only) * disc_rate)
        
        if to_disc > 0:
            self.logger.info(f"\n  Randomly discarding {to_disc} images ({disc_rate:.1%})...")
            random.shuffle(gen_only)
            for f in gen_only[:to_disc]:
                try:
                    lbl = out_lbl / f"{f.stem}.txt"
                    f.unlink(missing_ok=True)
                    lbl.unlink(missing_ok=True)
                    self.stats['discarded_total'] += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {f}: {e}")
        
        # Final statistics
        final_imgs = len(list(out_img.glob("*.*")))
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[OK] AUGMENTATION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Original images:     {num_orig}")
        self.logger.info(f"Generated images:    {self.stats['total_generated']}")
        self.logger.info(f"Final total:         {final_imgs}")
        self.logger.info(f"Target (if set):     {target_total if target_total else 'N/A'}")
        self.logger.info("\n Discard Statistics:")
        self.logger.info(f"  Total attempted:          {self.stats['total_attempted']}")
        self.logger.info(f"  Black frames:             {self.stats['discarded_black_frames']}")
        self.logger.info(f"  Extreme brightness:       {self.stats['discarded_extreme_brightness']}")
        self.logger.info(f"  Invalid bboxes:           {self.stats['discarded_invalid_bboxes']}")
        self.logger.info(f"  Transform failures:       {self.stats['discarded_transform_failed']}")
        self.logger.info(f"  Empty labels:             {self.stats['discarded_empty_labels']}")
        self.logger.info(f"  Random post-discard:      {to_disc}")
        
        total_discarded = (self.stats['discarded_black_frames'] + 
                          self.stats['discarded_extreme_brightness'] +
                          self.stats['discarded_invalid_bboxes'] +
                          self.stats['discarded_transform_failed'] +
                          self.stats['discarded_empty_labels'])
        
        if self.stats['total_attempted'] > 0:
            discard_pct = (total_discarded / self.stats['total_attempted']) * 100
            self.logger.info(f"\n  Overall discard rate:     {discard_pct:.1f}%")
            
            if discard_pct > 30:
                self.logger.warning("  High discard rate! Consider:")
                self.logger.warning("   - Relaxing quality_control thresholds")
                self.logger.warning("   - Reducing augmentation intensity")
                self.logger.warning("   - Checking transform parameters")
        
        self.logger.info("\n" + "=" * 80)


def main():
    import argparse
    p = argparse.ArgumentParser(description='YOLO Dataset Augmenter - Fixed Version')
    p.add_argument('--config', type=str, default='config_aug.yaml',
                   help='Path to configuration YAML file')
    args = p.parse_args()
    
    augmenter = YOLOAugmenterV2(args.config)
    augmenter.run()


if __name__ == "__main__":
    main()