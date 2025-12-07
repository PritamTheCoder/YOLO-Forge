"""
Quality Control Utility
Detects corrupt images, extremely dark/bright frames,
low resolution images, blank frames etc.
"""

import cv2
import os
from pathlib import Path
from typing import Dict, List


def is_corrupt(img_path: str) -> bool:
    img = cv2.imread(img_path)
    return img is None


def brightness_level(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_under_over_exposed(img, min_b=15, max_b=240) -> str:
    b = brightness_level(img)
    if b < min_b: return "underexposed"
    if b > max_b: return "overexposed"
    return "ok"


def resolution_check(img, min_w=320, min_h=320) -> bool:
    h, w = img.shape[:2]
    return w >= min_w and h >= min_h


def qc_scan_images(path: str,
                   min_res=(320, 320),
                   brightness=(15, 240)) -> Dict:

    path = Path(path)
    results = {"corrupt": [], "under": [], "over": [], "low_res": [], "clean": []}

    for img_path in path.glob("*.*"):
        img_path = str(img_path)

        if is_corrupt(img_path):
            results["corrupt"].append(img_path)
            continue

        img = cv2.imread(img_path)

        # exposure check
        exposure = is_under_over_exposed(img, brightness[0], brightness[1])
        if exposure == "underexposed": results["under"].append(img_path)
        elif exposure == "overexposed": results["over"].append(img_path)

        # resolution
        if not resolution_check(img, *min_res):
            results["low_res"].append(img_path)

        if exposure == "ok" and resolution_check(img, *min_res):
            results["clean"].append(img_path)

    return results


def print_qc_report(results: Dict):
    print("\n===== QC Report =====")
    for k, v in results.items():
        print(f"{k}: {len(v)}")
    print("=====================\n")


if __name__ == "__main__":
    res = qc_scan_images("images/")
    print_qc_report(res)