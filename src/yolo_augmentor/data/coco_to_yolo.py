"""
coco_to_yolo.py

Usage (example):
    from yolo_augmentor.data.coco_to_yolo import convert_coco_to_yolo

    summary = convert_coco_to_yolo(
        coco_json="path/to/instances.json",
        images_root="path/to/images_root",
        output_dir="path/to/output_yolo",
        copy_images=True,
        min_area=10.0,
        exclude_crowd=True,
        keep_empty_images=True
    )
"""

import json
from pathlib import Path
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def _xywh_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert COCO bbox (x,y,w,h) to YOLO normalized (xc,yc,w,h)."""
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h


def _bbox_from_segmentation(seg) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute bbox from polygon segmentation.
    'seg' is one segmentation (list of floats [x0,y0,x1,y1,...]).
    Returns COCO-style (x_min, y_min, width, height) or None if invalid.
    """
    try:
        xs = seg[0::2]
        ys = seg[1::2]
        if not xs or not ys:
            return None
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return x_min, y_min, x_max - x_min, y_max - y_min
    except Exception:
        return None


def convert_coco_to_yolo(
    coco_json: str,
    images_root: str,
    output_dir: str,
    copy_images: bool = True,
    min_area: float = 0.0,
    exclude_crowd: bool = True,
    keep_empty_images: bool = True
) -> Dict:
    """
    Convert COCO annotation JSON to YOLO folder structure.
    For each image, writes: output_dir/labels/<image_stem>.txt
    Copies (or moves) images to: output_dir/images/

    Returns a summary dict.
    """

    coco_json_path = Path(coco_json)
    images_root = Path(images_root)
    out = Path(output_dir)
    images_out = out / "images"
    labels_out = out / "labels"
    out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    # map category id -> name, then to contiguous index 0..N-1
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_class = {cid: i for i, cid in enumerate(sorted_cat_ids)}

    # write classes.txt (class order is contiguous index order)
    classes_txt = out / "classes.txt"
    with classes_txt.open("w", encoding="utf-8") as fh:
        for cid in sorted_cat_ids:
            fh.write(cat_id_to_name[cid] + "\n")

    # group annotations by image id
    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    images_processed = 0
    annotations_converted = 0
    empty_images = 0
    skipped_annotations = 0
    missing_image_files = 0

    for img_id, img_info in images.items():
        file_name = img_info.get("file_name")
        img_w = img_info.get("width")
        img_h = img_info.get("height")
        if not file_name or not img_w or not img_h:
            # skip malformed image entry
            continue

        # locate source image
        src_img_path = images_root / file_name
        if not src_img_path.exists():
            # try to find by name in images_root recursively
            alt = list(images_root.rglob(file_name))
            if alt:
                src_img_path = alt[0]
            else:
                # image file not found; skip writing labels for it
                missing_image_files += 1
                continue

        # copy/move image to output/images
        dst_img_path = images_out / src_img_path.name
        if copy_images:
            shutil.copy2(src_img_path, dst_img_path)
        else:
            shutil.move(str(src_img_path), str(dst_img_path))
        images_processed += 1

        # aggregate annotations for this image
        ann_list = anns_by_img.get(img_id, [])
        yolo_lines: List[str] = []

        for ann in ann_list:
            # optionally skip crowd
            if exclude_crowd and ann.get("iscrowd", 0) == 1:
                skipped_annotations += 1
                continue

            bbox = ann.get("bbox")
            if bbox and len(bbox) >= 4:
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                seg = ann.get("segmentation")
                bbox_from_seg = None
                if seg:
                    # segmentation can be list of polygons -> pick first polygon that produces valid bbox
                    if isinstance(seg, list):
                        for poly in seg:
                            candidate = _bbox_from_segmentation(poly)
                            if candidate:
                                bbox_from_seg = candidate
                                break
                    # other segmentation formats are ignored for simplicity
                if bbox_from_seg is None:
                    skipped_annotations += 1
                    continue
                x, y, w, h = bbox_from_seg

            # area filter
            area = ann.get("area", w * h)
            if area < min_area:
                skipped_annotations += 1
                continue

            # clip to image bounds
            x = max(0.0, min(x, img_w))
            y = max(0.0, min(y, img_h))
            w = max(0.0, min(w, img_w - x))
            h = max(0.0, min(h, img_h - y))
            if w <= 0 or h <= 0:
                skipped_annotations += 1
                continue

            # convert to yolo normalized coords
            x_c, y_c, nw, nh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
            # safety clip
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            cat_id = ann.get("category_id")
            if cat_id not in cat_id_to_class:
                skipped_annotations += 1
                continue
            cls_idx = cat_id_to_class[cat_id]
            yolo_lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}")
            annotations_converted += 1

        # write label file (even if empty optionally)
        label_file = labels_out / f"{dst_img_path.stem}.txt"
        if yolo_lines:
            label_file.write_text("\n".join(yolo_lines), encoding="utf-8")
        else:
            empty_images += 1
            if keep_empty_images:
                label_file.write_text("", encoding="utf-8")
            else:
                # remove copied image if we don't keep empty images
                if dst_img_path.exists():
                    dst_img_path.unlink()

    summary = {
        "images_processed": images_processed,
        "annotations_converted": annotations_converted,
        "empty_images": empty_images,
        "skipped_annotations": skipped_annotations,
        "missing_image_files": missing_image_files,
        "classes_file": str(classes_txt),
        "images_out": str(images_out),
        "labels_out": str(labels_out)
    }

    return summary
