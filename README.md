# 🔥 YOLO-Forge  
**Dataset Preparation & Augmentation Suite for YOLO Object Detection**  
📦 Scan → Convert → Repair → Split → Augment — *All in one pipeline.*

---

### Why YOLO-Forge?
YOLO datasets are messy in the real world — mixed formats, broken labels, unbalanced splits, low-quality images, missing boxes.

**YOLO-Forge solves this with automation.**

Designed especially for **small object datasets (ball, puck, shuttle, cork etc.)** and **fast-motion blur scenarios** with heavy bbox-aware augmentation techniques.

---

## ✨ Key Features

| Module | Capabilities |
|-------|-------------|
| 🔍 Scan | Detect dataset structure, count images/labels/classes |
| 🔄 Convert | Convert raw dataset → standard YOLO format |
| 🩺 Repair | Fix invalid labels, remove broken entries, normalize coords |
| 🔪 Split | Train/Val/Test splits automatically |
| ⚙ Augment | Extreme bbox-aware transformation pipeline |
| 🛡 QC Check | Detect corrupt/over-dark/over-bright images *(NEW)* |
| 🔗 CLI + Config | One-command full pipeline automation |

---

## 📁 Project Structure

```
yolo-forge/
├─ configs/
│   ├─ pipeline_config.yaml
│   ├─ augment_default.yaml
│   └─ augment_extreme.yaml
│
├─ src/yolo_augmentor/
│   ├─ pipeline.py
│   ├─ cli.py
│   ├─ validators.py
│   ├─ qc/image_qc.py               <- NEW MODULE
│   ├─ data/
│   └─ aug/
```

---

## 🚀 Installation

```bash
git clone https://github.com/<yourname>/yolo-forge.git
cd yolo-forge
pip install -r requirements.txt
```

---

## 🧭 Usage

### 1. Scan dataset structure

```bash
yolo-forge scan --path data_raw/
```

---

### 2. Convert → YOLO structured format

```bash
yolo-forge convert --input data_raw --output workspace
```

---

### 3. Repair annotations

```bash
yolo-forge repair --labels workspace/labels
```

---

### 4. Train/Val/Test split

```bash
yolo-forge split --input workspace --output split
```

---

### 5. Run augmentation only

```bash
yolo-forge augment --config configs/augment_default.yaml
```

---

### 6. End-to-end pipeline in ONE command

```bash
yolo-forge pipeline --config configs/pipeline_config.yaml
```

Output generated at:

```
final_dataset/
```

---

## 🔥 Example augmentation results

> *Show before/after images here later for portfolio impact*

---

## 📜 License

MIT

---

## 🤝 Contributions

PRs welcome. Recommended areas:

- Dashboard / web UI
- COCO/Pascal → YOLO converters
- New augmentation modules
- More QC metrics

---

## ⭐ If you like this project — give it a star!
