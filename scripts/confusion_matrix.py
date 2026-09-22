#!/usr/bin/env python3
"""Confusion matrix and per-species recall on a held-out split.

Both models are single-class detectors, so there is no class confusion to
report: the matrix is detection against background, and the background/
background cell is undefined because a detector is not asked to enumerate the
places a fish is not.

The per-species breakdown is the more informative half.  The models predict
only `fish`, but every ground-truth box still carries the species the
annotator assigned, so recall can be reported per species even though the
model never names one.  That answers a question the detector's own output
cannot: which species does it miss?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from detection_metrics import _iou  # noqa: E402
from detectors import predict  # noqa: E402


def load_species_ground_truth(via_json: Path, images_dir: Path) -> dict:
    """{filename: [(xyxy, species), ...]} for images present in the split."""
    data = json.loads(via_json.read_text(encoding="utf-8"))
    present = {p.name for p in images_dir.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

    sizes: dict[str, tuple[int, int]] = {}
    out: dict[str, list] = {}
    for meta in data["_via_img_metadata"].values():
        fn = meta["filename"]
        if fn not in present:
            continue
        if fn not in sizes:
            with Image.open(images_dir / fn) as im:
                sizes[fn] = im.size
        w, h = sizes[fn]

        boxes = []
        for r in meta.get("regions", []):
            shape = r.get("shape_attributes", {})
            if shape.get("name") != "rect":
                continue
            # Clamp exactly as build_splits.py does, so these boxes are the
            # same ones the YOLO labels encode.
            x1 = max(0.0, float(shape["x"]))
            y1 = max(0.0, float(shape["y"]))
            x2 = min(float(w), x1 + float(shape["width"]))
            y2 = min(float(h), y1 + float(shape["height"]))
            if x2 <= x1 or y2 <= y1:
                continue
            species = r.get("region_attributes", {}).get("name", "").strip()
            boxes.append(([x1, y1, x2, y2], species or "unlabelled"))
        out[fn] = boxes
    return out


def match(gt_boxes, detections, conf: float, iou_thr: float):
    """Greedy confidence-ordered matching; returns (matched gt idx, tp, fp)."""
    dets = sorted([d for d in detections if d[1] >= conf],
                  key=lambda d: d[1], reverse=True)
    matched: set[int] = set()
    tp = fp = 0
    for box, _ in dets:
        best_iou, best_idx = 0.0, -1
        for idx, (gt_box, _) in enumerate(gt_boxes):
            if idx in matched:
                continue
            v = _iou(box, gt_box)
            if v > best_iou:
                best_iou, best_idx = v, idx
        if best_iou >= iou_thr:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
    return matched, tp, fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--via-json", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", nargs=3,
                    metavar=("LABEL", "TYPE", "PATH"), required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/detections"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    images_dir = args.dataset_dir / "images" / args.split
    gt = load_species_ground_truth(args.via_json, images_dir)
    images = sorted(images_dir / fn for fn in gt)

    totals = Counter(sp for boxes in gt.values() for _, sp in boxes)
    print(f"{args.split}: {len(gt)} images, {sum(totals.values())} boxes, "
          f"{len(totals)} species\n")

    results = {}
    for label, kind, path in args.model:
        print(f"Running {label} ({kind})...")
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        per_image = predict(kind, Path(path), images, args.imgsz,
                            args.cache_dir / f"{slug}-{args.split}.json")

        tp = fp = 0
        hit: Counter = Counter()
        for fn, gt_boxes in gt.items():
            matched, t, f = match(gt_boxes, per_image.get(fn, []),
                                  args.conf, args.iou)
            tp += t
            fp += f
            for idx in matched:
                hit[gt_boxes[idx][1]] += 1
        fn_count = sum(totals.values()) - tp

        print(f"\n  Confusion matrix (conf {args.conf}, IoU {args.iou})")
        print(f"                     actual fish   actual background")
        print(f"    predicted fish   {tp:11d}   {fp:17d}")
        print(f"    predicted bg     {fn_count:11d}   {'n/a':>17}")
        print(f"    (background/background is undefined for detection)")

        print(f"\n  Recall by species")
        print(f"    {'species':<30} {'boxes':>6} {'found':>6} {'recall':>7}")
        rows = {}
        for sp, n in totals.most_common():
            r = hit[sp] / n if n else 0.0
            rows[sp] = {"boxes": n, "found": hit[sp], "recall": r}
            print(f"    {sp:<30} {n:6d} {hit[sp]:6d} {r:7.3f}")

        results[label] = {
            "weights": path, "type": kind,
            "confusion": {"tp": tp, "fp": fp, "fn": fn_count},
            "by_species": rows,
        }
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"split": args.split, "conf_threshold": args.conf,
             "iou_threshold": args.iou,
             "species_totals": dict(totals), "results": results},
            indent=2), encoding="utf-8")
        print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
