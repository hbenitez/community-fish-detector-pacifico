#!/usr/bin/env python3
"""Score one or more detectors on a held-out split, with identical metrics.

Answers the question the study is actually asking: does fine-tuning on Eastern
Pacific reef footage beat the general-purpose Community Fish Detector?  Both
models see the same images and the same ground truth, and both are scored by
scripts/detection_metrics.py.

Detections are collected at a very low confidence threshold because mAP
integrates over the whole precision/recall curve; thresholding early would
silently truncate it.  The fixed-threshold precision/recall/F1 is applied
afterwards, to the same detections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detection_metrics import (  # noqa: E402
    CATEGORY_ID, coco_map, load_ground_truth, precision_recall_f1,
)

MAP_THRESHOLD = 0.001


def predict_yolo(weights: Path, images: list[Path], imgsz: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for path in images:
        r = model.predict(str(path), conf=MAP_THRESHOLD, imgsz=imgsz,
                          verbose=False)[0]
        boxes = r.boxes
        out[path.name] = [
            (xyxy.tolist(), float(c))
            for xyxy, c in zip(boxes.xyxy.cpu(), boxes.conf.cpu())
        ]
    return out


def predict_rfdetr(weights: Path, images: list[Path]) -> dict:
    """Load exactly as rf_detr_batch_inference.py does, via its own detector."""
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rf_detr_batch_inference import (  # type: ignore
        MODEL_TYPE_MAP, detect_model_info_from_checkpoint,
    )

    info = detect_model_info_from_checkpoint(str(weights))
    model_type = info.get("model_type")
    resolution = info.get("resolution")
    if model_type is None:
        raise SystemExit(f"Could not determine model type from {weights}")
    print(f"  RF-DETR type={model_type} resolution={resolution}")

    cls = MODEL_TYPE_MAP[model_type]
    model = (cls(resolution=resolution, pretrain_weights=str(weights))
             if resolution else cls(pretrain_weights=str(weights)))

    out = {}
    for path in images:
        with Image.open(path) as im:
            det = model.predict(im.convert("RGB"), threshold=MAP_THRESHOLD)
        out[path.name] = [
            (xyxy.tolist(), float(c))
            for xyxy, c in zip(det.xyxy, det.confidence)
        ]
    return out


def to_coco_detections(per_image: dict, ground_truth: dict) -> list[dict]:
    id_by_name = {img["file_name"]: img["id"] for img in ground_truth["images"]}
    out = []
    for name, dets in per_image.items():
        image_id = id_by_name.get(name)
        if image_id is None:
            continue
        for (x1, y1, x2, y2), score in dets:
            out.append({
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", nargs=3,
                    metavar=("LABEL", "TYPE", "PATH"), required=True,
                    help="Repeatable. TYPE is 'yolo' or 'rfdetr'.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Operating point for precision/recall/F1.")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--iou-sweep", type=float, nargs="+",
                    default=[0.10, 0.25, 0.30, 0.50, 0.75],
                    help="IoU thresholds for the localisation sweep.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    images_dir = args.dataset_dir / "images" / args.split
    labels_dir = args.dataset_dir / "labels" / args.split
    if not images_dir.is_dir():
        raise SystemExit(f"No such split: {images_dir}")

    ground_truth = load_ground_truth(images_dir, labels_dir)
    images = [images_dir / img["file_name"] for img in ground_truth["images"]]
    print(f"{args.split}: {len(ground_truth['images'])} images, "
          f"{len(ground_truth['annotations'])} boxes\n")

    results = {}
    for label, kind, path in args.model:
        print(f"Running {label} ({kind})...")
        weights = Path(path)
        if kind == "yolo":
            per_image = predict_yolo(weights, images, args.imgsz)
        elif kind == "rfdetr":
            per_image = predict_rfdetr(weights, images)
        else:
            raise SystemExit(f"Unknown model type: {kind}")

        detections = to_coco_detections(per_image, ground_truth)
        results[label] = {
            "weights": str(weights),
            "type": kind,
            "n_detections": len(detections),
            **coco_map(ground_truth, detections),
            **precision_recall_f1(ground_truth, detections, args.conf, args.iou),
            "iou_sweep": {
                f"{t:.2f}": precision_recall_f1(ground_truth, detections,
                                                args.conf, t)
                for t in args.iou_sweep
            },
        }
        print(f"  {len(detections)} detections above {MAP_THRESHOLD}\n")

    header = (f"{'model':<28} {'mAP50':>7} {'mAP50-95':>9} "
              f"{'P':>7} {'R':>7} {'F1':>7}")
    print("=" * len(header))
    print(header)
    print(f"{'':28} {'':>7} {'':>9} "
          f"{'@' + str(args.conf):>7} {'@' + str(args.conf):>7} {'':>7}")
    print("-" * len(header))
    for label, r in results.items():
        print(f"{label:<28} {r['mAP50']:7.4f} {r['mAP50_95']:9.4f} "
              f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}")
    print("=" * len(header))

    # A model whose boxes follow a different convention from the annotator
    # loses matches to the IoU threshold rather than to missed fish.  Sweeping
    # the threshold separates "did not find it" from "drew it differently".
    print(f"\nRecall at conf {args.conf}, by IoU threshold:")
    head = f"  {'model':<28}" + "".join(f"{t:>8.2f}" for t in args.iou_sweep)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, r in results.items():
        row = "".join(f"{r['iou_sweep'][f'{t:.2f}']['recall']:8.3f}"
                      for t in args.iou_sweep)
        print(f"  {label:<28}{row}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"split": args.split,
             "images": len(ground_truth["images"]),
             "boxes": len(ground_truth["annotations"]),
             "conf_threshold": args.conf,
             "iou_threshold": args.iou,
             "results": results}, indent=2), encoding="utf-8")
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
