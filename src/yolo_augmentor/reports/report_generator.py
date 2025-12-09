"""
report_generator.py

Generates a self-contained HTML visual report for a YOLO-formatted dataset.

Usage:
    from yolo_augmentor.reports.report_generator import generate_report

    generate_report("output_dataset", out_dir="reports/last_run", samples=24)

Report contains:
 - Summary counts & health checks (missing labels, corrupt images, resolution stats)
 - Class distribution bar chart
 - BBox area histogram (px if available) and normalized area histogram
 - Aspect ratio histogram
 - Per-image bbox counts scatterplot
 - Sample tiled images with bbox overlays
 - Saves PNG charts and report_index.html (embeds images via relative links)
"""

import os
from pathlib import Path
import base64
import io
from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
import json
from PIL import Image

# --- helper readers -----------------------------------

def _read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception:
        return []
    if not text:
        return []
    items = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            items.append((cls, x, y, w, h))
        except Exception:
            continue
    return items


def _img_to_base64(path: Path, max_width: int = 1200) -> str:
    # load using PIL then encode to base64 JPEG for embedding
    try:
        im = Image.open(path)
        w, h = im.size
        if w > max_width:
            scale = max_width / w
            im = im.resize((int(w * scale), int(h * scale)))
        buffered = io.BytesIO()
        im.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _draw_bboxes(img_path: Path, labels: List[Tuple[int, float, float, float, float]], classes: List[str]) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        # return blank white image of small size
        return np.ones((200, 300, 3), dtype=np.uint8) * 255
    h, w = img.shape[:2]
    out = img.copy()
    rng = np.random.RandomState(abs(hash(img_path.name)) % (2**32))
    for (cls, cx, cy, bw, bh) in labels:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label_text = classes[cls] if cls < len(classes) else str(cls)
        cv2.putText(out, label_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# --- plotting helpers -----------------------------------

def _save_figure(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_class_distribution(class_counts: Counter, class_names: List[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(class_counts) == 0:
        ax.text(0.5, 0.5, "No labeled objects found", ha="center", va="center")
    else:
        ids = list(class_counts.keys())
        counts = [class_counts[i] for i in ids]
        labels = [class_names[i] if i < len(class_names) else str(i) for i in ids]
        ax.bar(range(len(ids)), counts)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Class distribution")
    _save_figure(fig, out_path)


def _plot_hist(values: List[float], title: str, out_path: Path, log_scale: bool = False):
    fig, ax = plt.subplots(figsize=(6, 4))
    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.hist(values, bins=40, log=log_scale)
        ax.set_title(title)
    _save_figure(fig, out_path)


# --- main generator -----------------------------------

def generate_report(yolo_root: str, out_dir: str = "reports/last_run", samples: int = 24) -> Dict:
    """
    Generate a visual and metrics report for a YOLO dataset folder.

    yolo_root should contain:
      - images/   (image files)
      - labels/   (txt files)
      - classes.txt (optional)

    Output: directory containing graphs and a report_index.html (human friendly)
    Returns: summary dict
    """
    root = Path(yolo_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images_dir = root / "images"
    labels_dir = root / "labels"
    classes_file = root / "classes.txt"

    # load classes
    classes = []
    if classes_file.exists():
        classes = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    # gather image files
    image_files = sorted([p for p in images_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    label_files = sorted([p for p in labels_dir.glob("*.txt")])

    # health counters
    corrupt_images = []
    low_res_images = []
    under_exposed = []
    over_exposed = []
    missing_label_for_image = []
    empty_label_files = []

    class_counts = Counter()
    bbox_areas = []
    bbox_areas_norm = []
    aspect_ratios = []
    bboxes_per_image = []

    # helper function to compute brightness
    def _brightness(img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    # scan labels and images
    for img_path in image_files:
        # labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_label_for_image.append(str(img_path.name))
            items = []
        else:
            items = _read_yolo_labels(label_path)
            if len(items) == 0:
                empty_label_files.append(str(label_path.name))

        # load image
        img = cv2.imread(str(img_path))
        if img is None:
            corrupt_images.append(str(img_path.name))
            continue

        h, w = img.shape[:2]
        if w < 200 or h < 200:
            low_res_images.append(str(img_path.name))

        bright = _brightness(img)
        if bright < 15:
            under_exposed.append(str(img_path.name))
        elif bright > 240:
            over_exposed.append(str(img_path.name))

        # collect bbox stats
        bboxes_per_image.append(len(items))
        for (cls, cx, cy, bw, bh) in items:
            class_counts[cls] += 1
            # pixel area if dims available
            area_px = (bw * w) * (bh * h)
            bbox_areas.append(area_px)
            bbox_areas_norm.append(bw * bh)
            aspect_ratios.append((bw * w) / (bh * h) if (bh * h) > 0 else 0)

    # overall metrics
    total_images = len(image_files)
    total_labels = sum(class_counts.values())
    avg_bboxes_per_image = (sum(bboxes_per_image) / total_images) if total_images else 0
    percent_missing_labels = (len(missing_label_for_image) / total_images * 100) if total_images else 0
    percent_corrupt = (len(corrupt_images) / total_images * 100) if total_images else 0
    percent_low_res = (len(low_res_images) / total_images * 100) if total_images else 0

    # produce plots
    _plot_class_distribution(class_counts, classes, out / "class_distribution.png")
    _plot_hist(bbox_areas, "Bounding box area (pixels)", out / "bbox_area_px_hist.png", log_scale=True)
    _plot_hist(bbox_areas_norm, "Bounding box area (normalized)", out / "bbox_area_norm_hist.png")
    _plot_hist(aspect_ratios, "BBox aspect ratio (w/h)", out / "bbox_aspect_ratio_hist.png")
    _plot_hist(list(class_counts.values()), "Instances per class", out / "instances_per_class.png")

    # sample images tiled with bbox overlay
    sample_n = min(samples, len(image_files))
    sample_paths = image_files[:sample_n]
    tiles = []
    for p in sample_paths:
        lab = labels_dir / f"{p.stem}.txt"
        labels = _read_yolo_labels(lab) if lab.exists() else []
        drawn = _draw_bboxes(p, labels, classes)
        # save small version to reports folder
        thumb_path = out / f"sample_{p.stem}.jpg"
        cv2.imwrite(str(thumb_path), drawn)
        tiles.append(thumb_path.name)

    # create a small JSON summary for easy machine parsing
    summary = {
        "total_images": total_images,
        "total_labelled_instances": total_labels,
        "classes_found": len(class_counts),
        "avg_bboxes_per_image": avg_bboxes_per_image,
        "missing_label_files": len(missing_label_for_image),
        "empty_label_files": len(empty_label_files),
        "corrupt_images": len(corrupt_images),
        "low_res_images": len(low_res_images),
        "under_exposed_images": len(under_exposed),
        "over_exposed_images": len(over_exposed),
        "percent_missing_labels": round(percent_missing_labels, 2),
        "percent_corrupt": round(percent_corrupt, 2),
        "percent_low_res": round(percent_low_res, 2),
    }
    # write summary json
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # build HTML report
    html_lines = []
    html_lines.append("<html><head><meta charset='utf-8'><title>YOLO-Forge Dataset Report</title></head><body>")
    html_lines.append(f"<h1>YOLO-Forge Dataset Report</h1>")
    html_lines.append("<h2>Summary</h2>")
    html_lines.append("<ul>")
    for k, v in summary.items():
        html_lines.append(f"<li><strong>{k}</strong>: {v}</li>")
    html_lines.append("</ul>")

    html_lines.append("<h2>Plots</h2>")
    for fname in ["class_distribution.png", "bbox_area_px_hist.png", "bbox_area_norm_hist.png", "bbox_aspect_ratio_hist.png", "instances_per_class.png"]:
        if (out / fname).exists():
            html_lines.append(f"<h3>{fname.replace('_',' ')}</h3>")
            html_lines.append(f"<img src='{fname}' width='800'>")

    html_lines.append("<h2>Sample images (with bounding boxes)</h2>")
    html_lines.append("<div style='display:flex;flex-wrap:wrap'>")
    for t in tiles:
        html_lines.append(f"<div style='margin:8px'><img src='{t}' width='300'></div>")
    html_lines.append("</div>")

    # data health details
    html_lines.append("<h2>Data health (details)</h2>")
    html_lines.append(f"<p>Missing label pairing: {len(missing_label_for_image)}</p>")
    if missing_label_for_image:
        html_lines.append("<details><summary>Missing label filenames (first 50)</summary><pre>")
        html_lines.append("\n".join(missing_label_for_image[:50]))
        html_lines.append("</pre></details>")

    html_lines.append(f"<p>Empty label files: {len(empty_label_files)}</p>")
    if empty_label_files:
        html_lines.append("<details><summary>Empty label files (first 50)</summary><pre>")
        html_lines.append("\n".join(empty_label_files[:50]))
        html_lines.append("</pre></details>")

    html_lines.append(f"<p>Corrupt images: {len(corrupt_images)}</p>")
    html_lines.append(f"<p>Low resolution images: {len(low_res_images)}</p>")
    if corrupt_images:
        html_lines.append("<details><summary>Corrupt images</summary><pre>")
        html_lines.append("\n".join(corrupt_images[:50]))
        html_lines.append("</pre></details>")

    html_lines.append("</body></html>")

    html_path = out / "report_index.html"
    html_path.write_text("\n".join(html_lines), encoding="utf-8")

    stats_json_path = Path(out_dir) / "summary.json"
    with open(stats_json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    return {
        "report_dir": str(out.resolve()),
        "summary": summary,
        "stats_json": str(stats_json_path.resolve()),
        "plots": [str((out / p).resolve()) for p in ["class_distribution.png", "bbox_area_px_hist.png", "bbox_area_norm_hist.png", "bbox_aspect_ratio_hist.png", "instances_per_class.png"]],
        "sample_images": [str((out / t).resolve()) for t in tiles],
        "html": str(html_path.resolve())
    }
