#!/usr/bin/env python3
"""Shared detection metrics, so every model is scored by the same code.

The imported scripts each carried their own copy of compute_iou and the
TP/FP/FN loop, and reported precision/recall/F1 at a fixed threshold for the
baseline while reading mAP50 out of ultralytics for the trained model.  Those
are different quantities and cannot be compared.  Everything here is computed
once, from one set of ground truth, for whichever model produced the
detections.

mAP comes from pycocotools -- the reference implementation -- rather than a
hand-rolled one, so the headline numbers are not this repository's opinion of
what mAP means.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

from PIL import Image

CATEGORY_ID = 1
CATEGORY_NAME = "fish"


def load_ground_truth(images_dir: Path, labels_dir: Path) -> dict:
    """Build a COCO ground-truth dict from a YOLO split.

    Images with no boxes are kept: they are background negatives and a model
    that fires on them must be penalised for it.
    """
    images, annotations = [], []
    ann_id = 1

    for img_id, img_path in enumerate(sorted(images_dir.iterdir()), start=1):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(img_path) as im:
            width, height = im.size
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, xc, yc, w, h = (float(v) for v in line.split())
            bw, bh = w * width, h * height
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY_ID,
                "bbox": [(xc * width) - bw / 2, (yc * height) - bh / 2, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }


def coco_map(ground_truth: dict, detections: list[dict]) -> dict[str, float]:
    """mAP50 and mAP50-95 via pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"mAP50": 0.0, "mAP50_95": 0.0}

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        gt_path.write_text(json.dumps(ground_truth))
        # pycocotools is chatty; its progress output is not a result.
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(gt_path))
            coco_dt = coco_gt.loadRes(detections)
            ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()

    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0])}


def _iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def precision_recall_f1(
    ground_truth: dict,
    detections: list[dict],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> dict[str, float]:
    """Greedy confidence-ordered matching at one operating point.

    Retained because the imported scripts reported precision/recall/F1 this
    way; it is one point on the curve mAP integrates, not a substitute for it.
    """
    gt_by_image: dict[int, list] = {img["id"]: [] for img in ground_truth["images"]}
    for a in ground_truth["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_image[a["image_id"]].append([x, y, x + w, y + h])

    det_by_image: dict[int, list] = {img["id"]: [] for img in ground_truth["images"]}
    for d in detections:
        if d["score"] < conf_threshold:
            continue
        x, y, w, h = d["bbox"]
        det_by_image[d["image_id"]].append(([x, y, x + w, y + h], d["score"]))

    tp = fp = fn = 0
    for img_id, gt_boxes in gt_by_image.items():
        dets = sorted(det_by_image[img_id], key=lambda t: t[1], reverse=True)
        matched: set[int] = set()
        for box, _ in dets:
            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched:
                    continue
                v = _iou(box, gt_box)
                if v > best_iou:
                    best_iou, best_idx = v, idx
            if best_iou >= iou_threshold:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}
