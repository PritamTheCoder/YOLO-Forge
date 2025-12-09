# YOLO-Forge: Advanced Dataset Augmentation & Pipeline Tool

**YOLO-Forge** is a production-grade, end-to-end dataset engineering pipeline designed specifically for training robust object detection models. While it handles standard dataset preparation tasks (conversion, splitting, repair), its core strength lies in its **specialized augmentation engine**.

Designed for high-velocity, small-object scenarios (e.g., sports ball tracking, drone surveillance, industrial inspection), YOLO-Forge applies augmentations **locally** (within/around bounding boxes) rather than globally. This preserves background context while generating difficult training samples that simulate motion blur, extreme lighting, occlusion, and shape deformation.

-----

## 🚀 Key Capabilities

### 1\. Specialized Augmentation Engine

Unlike standard libraries (Albumentations, imgaug) that apply transforms to the entire image, YOLO-Forge includes a custom registry of **Bbox-Aware Transforms**. These target the object of interest specifically:

  * **Motion Simulation:** Directional motion blur and shear based on object trajectory.
  * **Occlusion Modeling:** Geometric, organic, and pixel-level occlusions to simulate partial visibility.
  * **Lighting & Environment:** Localized brightness, glare (halo effects), and shadow injection.
  * **Adversarial Noise:** Concentrated Gaussian noise and texture degradation on the object itself.

### 2\. End-to-End Automation

A single command runs the entire lifecycle:

1.  **Scan:** Audits dataset health (missing labels, corrupt images).
2.  **Convert:** Normalizes directory structures (flat/nested) to standard YOLO format.
3.  **Repair:** Fixes malformed label files, clips coordinates to `[0,1]`, and removes empty annotations.
4.  **Augment:** Generates synthetic data based on configurable weights and pipelines.
5.  **Split:** Stratified random split into Train/Val/Test sets.

### 3\. Quality Control (QC)

Built-in safety rails prevent "garbage-in, garbage-out":

  * **Black Frame Detection:** Automatically discards augmented images that are too dark/empty.
  * **Bbox Validation:** Rejects invalid geometries (negative areas, out-of-bounds).
  * **Logging:** Comprehensive audit logs for every image generated or discarded.

-----

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo-forge.git
cd yolo-forge

# Install dependencies
pip install -r requirements.txt
```

*Requires: `numpy`, `opencv-python`, `PyYAML`, `albumentations`, `tqdm`*

-----

## ⚡ Quick Start

### 1\. The "All-in-One" Pipeline

The most efficient way to use YOLO-Forge is via the pipeline command, which reads a master config file.

```bash
python -m src.yolo_augmentor.cli pipeline --config configs/pipeline_config.yaml
```

### 2\. Individual Modules

You can also run specific utilities independently:

**Scan a dataset for errors:**

```bash
python -m src.yolo_augmentor.cli scan --path /path/to/raw_data
```

**Convert nested folders to YOLO format:**

```bash
python -m src.yolo_augmentor.cli convert --input /raw/data --output /clean/data
```

**Augment an existing dataset:**

```bash
python -m src.yolo_augmentor.cli augment --config configs/config_aug_extreme.yaml
```

-----

## 🎨 Augmentation Registry

YOLO-Forge features **8+ Custom Bbox-Oriented Augmentations** alongside standard global transforms.

| Transform Name | Description | Target Use Case |
| :--- | :--- | :--- |
| **`BboxMultiBlurAndShear`** | Applies diverse blur kernels (Gaussian, Motion) and shearing *only* to the object. | Fast-moving objects (balls, drones). |
| **`BboxExtremeShearOcclude`** | Aggressive geometric deformation + partial blocking. | Objects undergoing rapid direction changes. |
| **`NearBboxExtremeBrighten`** | Adds a bright "halo" or glare *around* the object, washing out edges. | Outdoor sports, sun glare, stadium lights. |
| **`ConcentratedNoiseTransform`** | Injects high-frequency noise centered on the object, fading radially. | Low-light sensors, ISO grain simulation. |
| **`BallBlendAndShapeBias`** | Subtle warping and alpha-blending with background colors. | Camouflage, objects blending into texture. |
| **`BallPixelLevelOcclusion`** | Random pixel dropouts (salt-and-pepper) inside the bbox. | Sensor dust, dead pixels, transmission errors. |
| **`GradientPatchTransform`** | Smooth directional lighting gradients over the object area. | Dynamic shadows and uneven lighting. |
| **`BboxGaussianOccludeShear`** | Soft, fog-like occlusion patches overlaid on the object. | Atmospheric interference (fog, smoke). |

### Configuration Example (`config_aug.yaml`)

Control the intensity and probability of every transform:

```yaml
augment_passes:
  - name: "motion_blur_pass"
    type: "custom"
    weight: 0.3
    pipeline:
      - transform: "BboxMultiBlurAndShearTransform"
        params:
          blur_types: ["motion"]
          shear_x_range: [-30, 30]
          bbox_prob: 0.9
```

-----

## ⚙️ Configuration Reference

### Pipeline Config (`pipeline_config.yaml`)

Controls the workflow steps.

```yaml
steps:
  scan: true
  convert_to_yolo: true
  repair_labels: true
  augment: 
    enabled: true
    config: "configs/config_aug_extreme.yaml"
  split:
    enabled: true
    train: 0.8
    val: 0.1
    test: 0.1
```

### Augmentation Config (`config_aug.yaml`)

Controls generation logic.

  * **`target_total_images`**: The total desired dataset size. The system calculates necessary multipliers automatically.
  * **`quality_control`**: Thresholds for discarding "bad" augmentations (e.g., `black_frame_threshold: 0.90`).
  * **`augment_passes`**: A list of augmentation strategies. You can mix "custom" (YOLO-Forge specific) and "standard" (Albumentations) passes.

-----

## 📊 Directory Structure

The system enforces a standard YOLO layout for compatibility with YOLOv5/v8/v10/v11.

**Input (Raw):**
Can be messy, nested, or flat. The `convert` step standardizes this.

**Output (Processed):**

```text
final_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

-----

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.

-----

**Developed for high-precision Computer Vision engineering.**
*Generate difficult data today, train robust models tomorrow.*