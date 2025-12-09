"""
Extended CLI with hybrid control:
- Individual tasks (scan/convert/split/augment/repair)
- Full automation pipeline via config
"""

import argparse

from .pipeline import run_pipeline
from .data.scan_dataset import scan_dataset
from .data.convert_to_yolo import convert_to_yolo
from .data.split_dataset import split_dataset
from .data.repair_labels import repair_labels
from .aug.augment_dataset import YOLOAugmenterV2


def build_parser():
    parser = argparse.ArgumentParser(prog="yolo-forge")
    sub = parser.add_subparsers(dest="command")

    # ------------------- scan -------------------
    c = sub.add_parser("scan")
    c.add_argument("--path", required=True)

    # ------------------- convert -------------------
    c = sub.add_parser("convert")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--move", action="store_true")

    # ------------------- split -------------------
    c = sub.add_parser("split")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--train", type=float, default=0.8)
    c.add_argument("--val", type=float, default=0.1)
    c.add_argument("--test", type=float, default=0.1)

    # ------------------- repair -------------------
    c = sub.add_parser("repair")
    c.add_argument("--labels", required=True)

    # ------------------- augment -------------------
    c = sub.add_parser("augment")
    c.add_argument("--config", default="../configs/config_aug.yaml", required=True)
    
    # --------------------report ---------------------
    r = sub.add_parser("report")
    r.add_argument("--dataset", required=True)
    r.add_argument("--out", default="reports/last_run")
    r.add_argument("--samples", type=int, default=24)

    # ------------------- pipeline (full automation) -------------------
    p = sub.add_parser("pipeline")
    p.add_argument("--config", default="../configs/pipeine_config.yaml", required=True)

    # ------------------- convert-coco entry (standalone tool) -----------
    c = sub.add_parser("convert-coco")
    c.add_argument("--json", required=True)
    c.add_argument("--images_root", required=True)
    c.add_argument("--out", required=True)
    
    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "scan":
        print(scan_dataset(args.path))

    elif args.command == "convert":
        convert_to_yolo(args.input, args.output, copy=not args.move)

    elif args.command == "split":
        split_dataset(args.input, args.output, args.train, args.val, args.test)

    elif args.command == "repair":
        repair_labels(args.labels)

    elif args.command == "augment":
        YOLOAugmenterV2(args.config).run()
    
    elif args.command == "report":
        from .reports.report_generator import generate_report
        res = generate_report(args.dataset, out_dir=args.out, samples=args.samples)
        print("Report generated:", res["html"])

    elif args.command == "pipeline":
        run_pipeline(args.config)

    elif args.command == "convert-coco":
        from .data.coco_to_yolo import convert_coco_to_yolo
        summary = convert_coco_to_yolo(args.json, args.images_root, args.out, copy_images=True)
        print("COCO->YOLO summary:", summary)
    
    else:
        print("\nUse one of the following:")
        print(" scan | convert | split | repair | augment | pipeline\n")


if __name__ == "__main__":
    main()
