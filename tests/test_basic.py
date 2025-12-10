"""
test_basic.py
=============
Basic pytest suite for YOLO-Forge pipeline validation.
Tests core functionality without requiring large datasets.

Run with:
    pytest tests/test_basic.py -v
    pytest tests/test_basic.py -v --tb=short  # shorter traceback
    pytest tests/test_basic.py::test_name -v  # run specific test
"""

import pytest
import yaml
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import tempfile


# ============================================================================
# FIXTURES - Create test data
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample RGB image (640x480)."""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = temp_dir / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_yolo_dataset(temp_dir):
    """Create minimal YOLO dataset structure with 3 images."""
    dataset_dir = temp_dir / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    
    # Create 3 test images
    for i in range(3):
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        img_path = images_dir / f"image_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Create corresponding label with 1-2 bboxes
        label_path = labels_dir / f"image_{i:03d}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 0.5 0.5 0.2 0.3\n")
            if i % 2 == 0:
                f.write(f"0 0.3 0.7 0.15 0.25\n")
    
    # Create classes.txt
    classes_file = dataset_dir / "classes.txt"
    classes_file.write_text("ball\n")
    
    return dataset_dir


@pytest.fixture
def sample_aug_config(temp_dir):
    """Create minimal augmentation config."""
    config = {
        'dataset': {
            'input_images_dir': str(temp_dir / 'input' / 'images'),
            'input_labels_dir': str(temp_dir / 'input' / 'labels'),
            'output_images_dir': str(temp_dir / 'output' / 'images'),
            'output_labels_dir': str(temp_dir / 'output' / 'labels'),
        },
        'validation': {
            'min_bbox_width': 0.01,
            'min_bbox_height': 0.01,
            'min_visibility': 0.1,
            'coord_min': 0.0,
            'coord_max': 1.0,
            'coord_tolerance': 0.05
        },
        'quality_control': {
            'black_frame_threshold': 0.90,
            'min_brightness': 5,
            'max_brightness': 250,
            'target_discard_rate': 0.05
        },
        'augment_passes': [
            {
                'name': 'test_pass',
                'type': 'standard',
                'count_multiplier': 1.0,
                'pipeline': [
                    {
                        'transform': 'HorizontalFlip',
                        'params': {'p': 0.5}
                    }
                ]
            }
        ]
    }
    
    config_path = temp_dir / 'aug_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def sample_pipeline_config(temp_dir):
    """Create minimal pipeline config."""
    config = {
        'dataset': {
            'input_dir': str(temp_dir / 'input'),
            'workspace_dir': str(temp_dir / 'workspace'),
            'output_dir': str(temp_dir / 'output')
        },
        'steps': {
            'scan': True,
            'convert_to_yolo': True,
            'repair_labels': True,
            'augment': {
                'enabled': False
            },
            'split': {
                'enabled': True,
                'train': 0.7,
                'val': 0.2,
                'test': 0.1
            }
        },
        'options': {
            'seed': 42,
            'copy_instead_of_move': True,
            'auto_cleanup_workspace': False
        }
    }
    
    config_path = temp_dir / 'pipeline_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


# ============================================================================
# TEST: Configuration Validation
# ============================================================================

def test_validate_pipeline_config(sample_pipeline_config):
    """Test pipeline config validation."""
    from src.yolo_augmentor.validators import validate_pipeline_config
    
    config = validate_pipeline_config(str(sample_pipeline_config))
    
    assert 'dataset' in config
    assert 'steps' in config
    assert 'input_dir' in config['dataset']
    assert 'output_dir' in config['dataset']


def test_invalid_config_missing_keys(temp_dir):
    """Test that invalid configs raise errors."""
    from src.yolo_augmentor.validators import validate_pipeline_config
    
    # Config missing required 'steps' key
    invalid_config = {'dataset': {'input_dir': 'test'}}
    config_path = temp_dir / 'invalid.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(invalid_config, f)
    
    with pytest.raises(ValueError, match="Missing required config section"):
        validate_pipeline_config(str(config_path))


# ============================================================================
# TEST: YOLO Label Validation
# ============================================================================

def test_valid_yolo_label():
    """Test validation of correctly formatted YOLO labels."""
    from src.yolo_augmentor.validators import validate_bbox_format
    
    valid_labels = [
        "0 0.5 0.5 0.3 0.4",
        "1 0.25 0.75 0.1 0.2",
        "2 0.9 0.1 0.05 0.15"
    ]
    
    for label in valid_labels:
        assert validate_bbox_format(label) is True


def test_invalid_yolo_labels():
    """Test detection of invalid YOLO label formats."""
    from src.yolo_augmentor.validators import validate_bbox_format
    
    invalid_labels = [
        "0 0.5 0.5",              # Too few values
        "0 1.5 0.5 0.3 0.4",      # x out of range
        "0 0.5 0.5 -0.1 0.4",     # negative width
        "abc 0.5 0.5 0.3 0.4",    # non-numeric class
        ""                        # empty
    ]
    
    for label in invalid_labels:
        assert validate_bbox_format(label) is False


def test_repair_labels_fixes_issues(temp_dir):
    """Test that repair_labels fixes malformed labels."""
    from src.yolo_augmentor.data.repair_labels import repair_labels
    
    labels_dir = temp_dir / "labels"
    labels_dir.mkdir()
    
    # Create label with issues
    label_file = labels_dir / "test.txt"
    label_file.write_text(
        "0 0.5 0.5 0.3 0.4\n"      # valid
        "1 1.5 0.5 0.3 0.4\n"      # x out of range
        "2 0.5 0.5 -0.1 0.4\n"     # negative width
        "0 0.3 0.7 0.2 0.15\n"     # valid
    )
    
    repair_labels(str(labels_dir), backup=False)
    
    # Check repaired content
    repaired = label_file.read_text().strip().split('\n')
    assert len(repaired) == 2  # Only 2 valid lines remain


# ============================================================================
# TEST: Dataset Scanning
# ============================================================================

def test_scan_dataset_flat_structure(sample_yolo_dataset):
    """Test scanning a properly structured YOLO dataset."""
    from src.yolo_augmentor.data.scan_dataset import scan_dataset
    
    result = scan_dataset(str(sample_yolo_dataset))
    
    assert result['total_images'] == 3
    assert result['total_labels'] == 3
    assert result['missing_pairs'] == 0
    assert result['structure_type'] == 'flat_yolo'
    assert result['has_problems'] is False


def test_scan_dataset_missing_labels(temp_dir):
    """Test detection of missing label files."""
    from src.yolo_augmentor.data.scan_dataset import scan_dataset
    
    dataset_dir = temp_dir / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    
    # Create 3 images but only 2 labels
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i}.jpg"), img)
    
    for i in range(2):  # Only 2 labels
        (labels_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    
    result = scan_dataset(str(dataset_dir))
    
    assert result['total_images'] == 3
    assert result['total_labels'] == 2
    assert result['missing_pairs'] == 1
    assert result['has_problems'] is True


# ============================================================================
# TEST: Dataset Conversion
# ============================================================================

def test_convert_to_yolo(temp_dir):
    """Test conversion of nested dataset to flat YOLO structure."""
    from src.yolo_augmentor.data.convert_to_yolo import convert_to_yolo
    
    # Create nested input structure
    input_dir = temp_dir / "input"
    nested_img_dir = input_dir / "subfolder" / "images"
    nested_lbl_dir = input_dir / "subfolder" / "labels"
    nested_img_dir.mkdir(parents=True)
    nested_lbl_dir.mkdir(parents=True)
    
    # Create test files
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(nested_img_dir / "test.jpg"), img)
    (nested_lbl_dir / "test.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    
    output_dir = temp_dir / "output"
    
    convert_to_yolo(str(input_dir), str(output_dir), copy=True)
    
    # Verify flat structure created
    assert (output_dir / "images" / "test.jpg").exists()
    assert (output_dir / "labels" / "test.txt").exists()


# ============================================================================
# TEST: Dataset Splitting
# ============================================================================

def test_split_dataset(sample_yolo_dataset, temp_dir):
    """Test splitting dataset into train/val/test."""
    from src.yolo_augmentor.data.split_dataset import split_dataset
    
    output_dir = temp_dir / "split_output"
    
    split_dataset(
        str(sample_yolo_dataset),
        str(output_dir),
        train=0.6,
        val=0.2,
        test=0.2,
        seed=42,
        copy=True
    )
    
    # Check split directories exist
    assert (output_dir / "train" / "images").exists()
    assert (output_dir / "train" / "labels").exists()
    assert (output_dir / "val" / "images").exists()
    assert (output_dir / "val" / "labels").exists()
    assert (output_dir / "test" / "images").exists()
    assert (output_dir / "test" / "labels").exists()
    
    # Count files (3 total images should be split)
    train_imgs = len(list((output_dir / "train" / "images").glob("*.jpg")))
    val_imgs = len(list((output_dir / "val" / "images").glob("*.jpg")))
    test_imgs = len(list((output_dir / "test" / "images").glob("*.jpg")))
    
    assert train_imgs + val_imgs + test_imgs == 3


# ============================================================================
# TEST: Custom Transforms (Basic)
# ============================================================================

def test_base_transform_yolo_conversion():
    """Test YOLO coordinate conversion utilities."""
    from src.yolo_augmentor.aug.custom_transforms import BaseTransform
    
    # Test pixel to YOLO conversion
    img_shape = (480, 640)
    bbox_yolo = BaseTransform.pixels_to_yolo(100, 100, 200, 200, img_shape)
    
    assert 0 <= bbox_yolo[0] <= 1  # x_center
    assert 0 <= bbox_yolo[1] <= 1  # y_center
    assert bbox_yolo[2] > 0         # width
    assert bbox_yolo[3] > 0         # height
    
    # Test YOLO to pixel conversion
    x1, y1, x2, y2 = BaseTransform.yolo_to_pixels(
        [0.5, 0.5, 0.3, 0.4], 
        img_shape
    )
    
    assert x1 < x2
    assert y1 < y2


def test_bbox_clipping():
    """Test bbox clipping to image boundaries."""
    from src.yolo_augmentor.aug.custom_transforms import BaseTransform
    
    img_shape = (480, 640)
    
    # Test clipping coordinates outside image
    x1, y1, x2, y2 = BaseTransform.clip_bbox_to_image(
        -50, -50, 700, 500, img_shape
    )
    
    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= 640
    assert y2 <= 480


def test_radial_mask_generation():
    """Test radial mask generation utility."""
    from src.yolo_augmentor.aug.utils import radial_mask
    
    mask = radial_mask(100, 100, 50, 50, radius=30, falloff='gaussian')
    
    assert mask.shape == (100, 100)
    assert mask.dtype == np.float32
    assert 0 <= mask.max() <= 1
    assert mask[50, 50] > mask[0, 0]  # Center brighter than edge


# ============================================================================
# TEST: Augmentation System (Integration)
# ============================================================================

def test_augmentation_config_loading(sample_aug_config):
    """Test augmentation config can be loaded."""
    from src.yolo_augmentor.aug.augment_dataset import YOLOAugmenterV2
    
    augmenter = YOLOAugmenterV2(str(sample_aug_config))
    
    assert augmenter.config is not None
    assert 'dataset' in augmenter.config
    assert 'augment_passes' in augmenter.config
    assert len(augmenter.config['augment_passes']) > 0


def test_custom_transform_registry():
    """Test custom transform registration."""
    import logging
    from src.yolo_augmentor.aug.augment_dataset import CustomTransformRegistry
    
    logger = logging.getLogger(__name__)
    registry = CustomTransformRegistry(logger)
    
    # Check built-in transforms registered
    assert 'BboxMultiBlurAndShearTransform' in registry.transforms
    assert 'NearBboxExtremeBrighten' in registry.transforms
    assert len(registry.transforms) >= 8


# ============================================================================
# TEST: COCO to YOLO Conversion
# ============================================================================

def test_coco_bbox_conversion():
    """Test COCO bbox to YOLO format conversion."""
    from src.yolo_augmentor.data.coco_to_yolo import _xywh_to_yolo
    
    # COCO format: [x_min, y_min, width, height] in pixels
    x, y, w, h = 100, 100, 50, 50
    img_w, img_h = 640, 480
    
    x_c, y_c, w_norm, h_norm = _xywh_to_yolo(x, y, w, h, img_w, img_h)
    
    # Check normalized values
    assert 0 < x_c < 1
    assert 0 < y_c < 1
    assert 0 < w_norm < 1
    assert 0 < h_norm < 1


def test_coco_segmentation_to_bbox():
    """Test bbox extraction from polygon segmentation."""
    from src.yolo_augmentor.data.coco_to_yolo import _bbox_from_segmentation
    
    # Polygon: [x0, y0, x1, y1, x2, y2, ...]
    polygon = [100, 100, 200, 100, 200, 200, 100, 200]
    
    bbox = _bbox_from_segmentation(polygon)
    
    assert bbox is not None
    assert len(bbox) == 4
    x_min, y_min, width, height = bbox
    assert width > 0
    assert height > 0


# ============================================================================
# TEST: Report Generation (Basic)
# ============================================================================

def test_report_generation(sample_yolo_dataset, temp_dir):
    """Test basic report generation."""
    from src.yolo_augmentor.reports.report_generator import generate_report
    
    report_dir = temp_dir / "report"
    
    result = generate_report(
        str(sample_yolo_dataset),
        out_dir=str(report_dir),
        samples=2
    )
    
    # Check report files created
    assert (report_dir / "report_index.html").exists()
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "class_distribution.png").exists()
    
    # Check result structure
    assert 'summary' in result
    assert result['summary']['total_images'] == 3
    assert 'html' in result


# ============================================================================
# TEST: Utility Functions
# ============================================================================

def test_rectangular_gradient_mask():
    """Test rectangular gradient mask generation."""
    from src.yolo_augmentor.aug.utils import rectangular_gradient_mask
    
    mask = rectangular_gradient_mask(
        h=480, w=640,
        bbox_x1=200, bbox_y1=150, bbox_x2=400, bbox_y2=350,
        outer_x1=150, outer_y1=100, outer_x2=450, outer_y2=400,
        falloff='gaussian',
        edge_only=False
    )
    
    assert mask.shape == (480, 640)
    assert mask.dtype == np.float32
    assert 0 <= mask.max() <= 1


def test_apply_masked_blend():
    """Test masked blending utility."""
    from src.yolo_augmentor.aug.utils import apply_masked_blend
    
    base = np.ones((100, 100, 3), dtype=np.uint8) * 100
    overlay = np.ones((100, 100, 3), dtype=np.uint8) * 200
    mask = np.ones((100, 100), dtype=np.float32) * 0.5
    
    result = apply_masked_blend(base, overlay, mask, alpha=1.0)
    
    assert result.shape == base.shape
    assert result.dtype == np.uint8
    # Should be blend of 100 and 200
    assert 100 < result[50, 50, 0] < 200


# ============================================================================
# TEST: Edge Cases
# ============================================================================

def test_empty_label_file_handling(temp_dir):
    """Test handling of empty label files."""
    from src.yolo_augmentor.data.repair_labels import repair_labels
    
    labels_dir = temp_dir / "labels"
    labels_dir.mkdir()
    
    # Create empty label file
    empty_label = labels_dir / "empty.txt"
    empty_label.write_text("")
    
    repair_labels(str(labels_dir), backup=False)
    
    # Empty file should be removed
    assert not empty_label.exists()


def test_zero_width_bbox_rejection():
    """Test that zero-width bboxes are rejected."""
    from src.yolo_augmentor.validators import validate_bbox_format
    
    # Zero width
    assert validate_bbox_format("0 0.5 0.5 0.0 0.4") is False
    
    # Zero height
    assert validate_bbox_format("0 0.5 0.5 0.3 0.0") is False


# ============================================================================
# PERFORMANCE / SMOKE TESTS
# ============================================================================

@pytest.mark.slow
def test_augmentation_runs_without_crash(sample_yolo_dataset, temp_dir):
    """Smoke test: augmentation completes without crashing."""
    from src.yolo_augmentor.aug.augment_dataset import YOLOAugmenterV2
    
    # Create config
    config = {
        'dataset': {
            'input_images_dir': str(sample_yolo_dataset / 'images'),
            'input_labels_dir': str(sample_yolo_dataset / 'labels'),
            'output_images_dir': str(temp_dir / 'aug_out' / 'images'),
            'output_labels_dir': str(temp_dir / 'aug_out' / 'labels'),
        },
        'validation': {
            'min_bbox_width': 0.01,
            'min_bbox_height': 0.01,
            'min_visibility': 0.1,
            'coord_min': 0.0,
            'coord_max': 1.0,
            'coord_tolerance': 0.05
        },
        'quality_control': {
            'black_frame_threshold': 0.95,
            'min_brightness': 3,
            'max_brightness': 252,
            'target_discard_rate': 0.0
        },
        'augment_passes': [
            {
                'name': 'basic_flip',
                'type': 'standard',
                'count_multiplier': 0.5,
                'pipeline': [
                    {'transform': 'HorizontalFlip', 'params': {'p': 1.0}}
                ]
            }
        ]
    }
    
    config_path = temp_dir / 'aug_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    augmenter = YOLOAugmenterV2(str(config_path))
    
    # Should not crash
    try:
        augmenter.run()
        success = True
    except Exception as e:
        success = False
        print(f"Augmentation failed: {e}")
    
    assert success


# ============================================================================
# SUMMARY TEST
# ============================================================================

def test_complete_mini_pipeline(temp_dir):
    """Integration test: mini pipeline from input to split output."""
    # 1. Create input data
    input_dir = temp_dir / "input"
    img_dir = input_dir / "images"
    lbl_dir = input_dir / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    
    for i in range(5):
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)
        (lbl_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    
    # 2. Convert
    from src.yolo_augmentor.data.convert_to_yolo import convert_to_yolo
    workspace = temp_dir / "workspace"
    convert_to_yolo(str(input_dir), str(workspace), copy=True)
    
    # 3. Repair
    from src.yolo_augmentor.data.repair_labels import repair_labels
    repair_labels(str(workspace / "labels"), backup=False)
    
    # 4. Split
    from src.yolo_augmentor.data.split_dataset import split_dataset
    output_dir = temp_dir / "output"
    split_dataset(str(workspace), str(output_dir), 0.6, 0.2, 0.2, seed=42)
    
    # 5. Verify
    assert (output_dir / "train" / "images").exists()
    assert (output_dir / "val" / "images").exists()
    assert (output_dir / "test" / "images").exists()
    
    total = sum([
        len(list((output_dir / split / "images").glob("*.jpg")))
        for split in ['train', 'val', 'test']
    ])
    
    assert total == 5  # All images accounted for


if __name__ == "__main__":
    pytest.main([__file__, "-v"])