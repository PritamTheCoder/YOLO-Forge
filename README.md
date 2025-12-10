# YOLO-Forge

![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)
[![Docker Pulls](https://img.shields.io/docker/pulls/aurelian1111/yolo-forge)](https://hub.docker.com/r/aurelian1111/yolo-forge)
![Docker Image Size](https://img.shields.io/docker/image-size/aurelian1111/yolo-forge/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/pritamthecoder/yolo-forge)](https://github.com/pritamthecoder/yolo-forge/graphs/commit-activity)

**End-to-End Dataset Engineering, Augmentation & Automation for Object Detection**

YOLO-Forge is a production-ready dataset pipeline designed for training robust object detection models (YOLOv5/v8/v11). Unlike standard augmentation libraries that process global frames, YOLO-Forge features a **Bbox-Aware Engine** optimized for small-object tracking, drone surveillance, and industrial vision.

It modifies regions *inside and around* the object to simulate motion, occlusion, and sensor noise without destroying background context.

-----

## 🏗️ Architecture & Workflow

YOLO-Forge automates the "messy" side of computer vision data prep through a strictly typed, sequential pipeline:

```mermaid
    A[Scan] --> B[Convert]
    B --> C[Repair]
    C --> D[Augment]
    D --> E[Split]
    E --> F[Report]
```

| Stage | Description |
| :--- | :--- |
| **1. Scan** | Validates directory structure, detects missing labels, and reports initial dataset health. |
| **2. Convert** | Normalizes diverse input formats (nested folders, flat files) into standard YOLO structure. |
| **3. Repair** | Fixes invalid labels, normalizes coordinates to `[0,1]`, and removes broken/corrupt files. |
| **4. Augment** | **(Core)** Generates synthetic samples using motion, glare, occlusion, and warping. |
| **5. Split** | Automatically splits data into Train/Val/Test sets based on configurable ratios. |
| **6. Report** | Generates HTML reports, class distribution histograms, and health metrics. |

-----

## ⚡ Quick Start

### Option A: Docker (Recommended)

*Best for production consistency. No environment setup required.*

**1. Pull the Image**

```bash
docker pull aurelian1111/yolo-forge:latest
```

**2. Run the Pipeline**
Ensure your data resides in a folder (e.g., `data/`) containing `images/` and `labels/`.

**Linux / MacOS:**

```bash
docker run --rm -it \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  aurelian1111/yolo-forge:latest \
  pipeline --config configs/pipeline_config.yaml
```

**Windows (PowerShell):**

```powershell
docker run --rm -it `
  -v "C:\path\to\dataset:/data" `
  -v "C:\path\to\output:/output" `
  aurelian1111/yolo-forge:latest `
  pipeline --config configs/pipeline_config.yaml
```

### Option B: Local Installation

*Best for development and debugging.*

```bash
# Clone and Install
git clone https://github.com/YOUR-USERNAME/yolo-forge.git
cd yolo-forge
pip install -r requirements.txt

# Run Pipeline
python -m src.yolo_augmentor.cli pipeline --config configs/pipeline_config.yaml
```

-----

## 🎨 Bbox-Aware Augmentation Engine

YOLO-Forge specializes in difficult vision scenarios. It includes **8+ custom augmentation modules** that target the bounding box area specifically.

| Transform | Effect | Use Case |
| :--- | :--- | :--- |
| **Multi-Blur + Shear** | Motion simulation | Fast moving objects (soccer balls, drones). |
| **Occlusion Warp** | Object blocking & distortion | Objects moving behind trees/poles. |
| **Bright Halo Boost** | Lens glare simulation | Stadium lights, sun glare. |
| **Concentrated Noise** | Low-light sensor simulation | Nighttime surveillance, ISO grain. |
| **Pixel-drop Occlusion** | Transmission artifacts | Dead pixels, signal interference. |
| **Gaussian Fog Patch** | Weather interference | Fog, smoke, steam. |
| **Shape Bias + Blending** | Texture camouflage | Objects blending into complex backgrounds. |
| **Gradient Center Patch** | Light gradients | Dynamic shadows. |

-----

## 🛠️ Standalone Tools

YOLO-Forge exposes individual modules for specific tasks without running the full pipeline.

### COCO → YOLO Converter

Convert COCO JSON annotations to YOLO `.txt` format instantly.

```bash
python -m src.tools.coco2yolo \
  --json annotations.json \
  --img_dir path/to/images \
  --output labels_yolo/
```

### Manual Scan & Repair

Audit a dataset without altering it, or run a repair pass to fix coordinate errors.

```bash
# Scan only
python -m src.yolo_augmentor.cli scan --path /data/dataset

# Repair labels
python -m src.yolo_augmentor.cli repair --input /data/raw --output /data/clean
```

-----

## ⚙️ Configuration Reference

### Pipeline Config (`pipeline_config.yaml`)

Controls which steps of the lifecycle are active.

```yaml
dataset:
  input_dir: "/data"       # Mapped to container volume
  output_dir: "/output"    # Results location
  workspace_dir: "workspace"

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
  report:
    enabled: true
    samples: 30
```

### Augmentation Profile (`config_aug_extreme.yaml`)

Controls the intensity of synthetic generation.

```yaml
dataset:
  # Note: Paths must match container paths if using Docker
  input_images_dir: "/data/images"
  input_labels_dir: "/data/labels"
  output_images_dir: "/output/aug/images"
  output_labels_dir: "/output/aug/labels"
  
  # Target size of the final dataset
  target_total_images: 200

quality_control:
  black_frame_threshold: 0.90 # Discard images that become too dark
```

-----

## 📊 Outputs & Reporting

The system ensures a strictly organized output directory ready for training.

```text
output/
├── train/              # Ready for YOLO training
│   ├── images/
│   └── labels/
├── val/
├── test/
└── report/
    ├── index.html               <-- Interactive Dataset Report
    ├── summary.json             <-- Machine readable metrics
    ├── class_distribution.png
    ├── bbox_hist.png            <-- Area/Ratio analysis
    └── instances_per_class.png
```

**The HTML Report includes:**

  * Class balance visualization.
  * Bounding box aspect ratio & area histograms (crucial for anchor box tuning).
  * Visual grid of augmented samples.
  * Dataset health metrics.

-----

## 👨‍💻 Development

To build the Docker image locally:

```bash
docker build -t yolo-forge .
```

To run the CLI help menu:

```bash
python -m src.yolo_augmentor.cli --help
```

-----

## 🛡️ License

MIT License. Free for commercial and research use.

> *"Forge your data like steel. The harsher the training, the stronger the model."*

By - **PritamTheCoder** | **Pritam Thapa**

-----