#!/usr/bin/env python3
"""Model loading and inference, shared by evaluate.py and confusion_matrix.py.

Detections are cached to disk keyed by model and split, because RF-DETR
inference over a split takes minutes and several analyses run over the same
predictions.  Delete the cache file to force a re-run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Confidence floor for collecting detections.  mAP integrates the whole
# precision/recall curve, so thresholding here would truncate it; the
# operating-point threshold is applied later, to these same detections.
MAP_THRESHOLD = 0.001


def predict_yolo(weights: Path, images: list[Path], imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for path in images:
        r = model.predict(str(path), conf=MAP_THRESHOLD, imgsz=imgsz,
                          verbose=False)[0]
        out[path.name] = [
            (xyxy.tolist(), float(c))
            for xyxy, c in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu())
        ]
    return out


def predict_rfdetr(weights: Path, images: list[Path]) -> dict:
    """Load exactly as rf_detr_batch_inference.py does, via its own helpers."""
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rf_detr_batch_inference import (  # type: ignore
        MODEL_TYPE_MAP, detect_model_info_from_checkpoint,
    )

    info = detect_model_info_from_checkpoint(str(weights))
    model_type, resolution = info.get("model_type"), info.get("resolution")
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


def predict(kind: str, weights: Path, images: list[Path],
            imgsz: int = 640, cache: Path | None = None) -> dict:
    if cache is not None and cache.is_file():
        print(f"  using cached detections: {cache}")
        return json.loads(cache.read_text())

    if kind == "yolo":
        out = predict_yolo(weights, images, imgsz)
    elif kind == "rfdetr":
        out = predict_rfdetr(weights, images)
    else:
        raise SystemExit(f"Unknown model type: {kind}")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out
