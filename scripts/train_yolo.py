#!/usr/bin/env python3
"""Fine-tune a YOLO detector on a dataset config produced by build_splits.py.

Replaces entrenar_peces.py: paths come from the config rather than being
hardcoded, and `val` is a genuine held-out split.  `test` is deliberately not
touched here -- it is read once, by the evaluation step, for reported numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="Dataset YAML from build_splits.py")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="640 matches the RF-DETR baseline's training "
                         "resolution; 1024 needs >16GB at batch 8.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu; ultralytics picks one if unset.")
    ap.add_argument("--name", default="pacifico_fish_1class")
    ap.add_argument("--project", type=Path, default=Path("runs/detect"),
                    help="Resolved to an absolute path: ultralytics joins a "
                         "relative project onto its own runs_dir setting.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"Dataset config not found: {args.data}\n"
              f"Run scripts/build_splits.py first.", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        name=args.name,
        project=str(args.project.resolve()),
        seed=args.seed,
        deterministic=True,
        val=True,
        plots=True,
    )
    print(f"\nRun written to {args.project.resolve() / args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
