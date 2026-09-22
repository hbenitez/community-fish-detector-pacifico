#!/usr/bin/env python3
"""Build leakage-free train/val/test splits from VIA annotations.

Frames are sampled one per 3 seconds from continuous video transects, so
neighbouring frames are near-duplicates.  A random split would place those
near-duplicates on both sides of the train/val boundary and inflate every
metric.  Instead each video is cut into contiguous temporal blocks, whole
blocks are assigned to a split, and a guard band of frames is dropped either
side of every boundary where the assignment changes.

By default all boxes collapse to a single "fish" class: the annotations are
60% `not_defined`, which is unusable for species classification but perfectly
valid as a fish/no-fish positive.

`--task species` instead gives its own class to every species with at least
`--min-boxes` annotations, groups the remaining identified species as
`other_fish`, and keeps `not_defined` as an explicit `unidentified` class.
Dropping the unidentified boxes is not an option: they would become background,
teaching the model that fish are background.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

FRAME_RE = re.compile(r"^(?P<video>.+)_(?P<index>\d+)\.jpg$", re.IGNORECASE)

# One entry per contiguous block, applied to every video in order.
DEFAULT_PATTERN = ("train", "val", "train", "train", "test", "train")

UNIDENTIFIED_SOURCE = "not_defined"
UNIDENTIFIED_CLASS = "unidentified"
OTHER_CLASS = "other_fish"


def build_class_map(annotations, task: str, min_boxes: int):
    """Return (name -> class id function, ordered class names)."""
    if task == "fish":
        return (lambda name: 0), ["fish"]

    counts = Counter(
        r.get("region_attributes", {}).get("name", "").strip()
        for regions in annotations.values() for r in regions
    )
    named = sorted(
        (n for n, c in counts.items()
         if n and n != UNIDENTIFIED_SOURCE and c >= min_boxes),
        key=lambda n: -counts[n],
    )
    names = named + [OTHER_CLASS, UNIDENTIFIED_CLASS]
    index = {n: i for i, n in enumerate(names)}

    def class_map(name: str):
        name = name.strip()
        if name == UNIDENTIFIED_SOURCE:
            return index[UNIDENTIFIED_CLASS]
        if name in index:
            return index[name]
        # Identified, but too rare to carry its own class.
        return index[OTHER_CLASS]

    return class_map, names


def parse_via(via_path: Path) -> dict[str, list[dict]]:
    """Return {filename: [region, ...]} for rect regions only."""
    with via_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "_via_img_metadata" not in data:
        raise ValueError(f"{via_path} is not a VIA project file")

    out: dict[str, list[dict]] = {}
    for meta in data["_via_img_metadata"].values():
        regions = [
            r for r in meta.get("regions", [])
            if r.get("shape_attributes", {}).get("name") == "rect"
        ]
        out[meta["filename"]] = regions
    return out


def group_by_video(filenames) -> dict[str, list[tuple[int, str]]]:
    """Group frames by source video, ordered by frame index."""
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for fn in filenames:
        m = FRAME_RE.match(fn)
        if m is None:
            raise ValueError(
                f"Cannot parse a video and frame index from {fn!r}; "
                "splits would not be leakage-free."
            )
        groups[m.group("video")].append((int(m.group("index")), fn))
    for frames in groups.values():
        frames.sort()
    return dict(groups)


def assign_blocks(
    frames: list[tuple[int, str]],
    pattern: tuple[str, ...],
    guard: int,
) -> tuple[dict[str, list[str]], list[str]]:
    """Cut one video into contiguous blocks and drop guard bands.

    Returns (split -> filenames, dropped filenames).
    """
    n = len(frames)
    n_blocks = len(pattern)
    if n < n_blocks:
        raise ValueError(f"Only {n} frames for {n_blocks} blocks")

    # Block boundaries, as even as the frame count allows.
    edges = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
    labels = [""] * n
    for block, split in enumerate(pattern):
        for i in range(edges[block], edges[block + 1]):
            labels[i] = split

    # Drop `guard` frames either side of every change in assignment.  The
    # frames are 3s apart, so this opens a real temporal gap between splits.
    dropped = set()
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            for j in range(max(0, i - guard), min(n, i + guard)):
                dropped.add(j)

    result: dict[str, list[str]] = defaultdict(list)
    dropped_files = []
    for i, (_, fn) in enumerate(frames):
        if i in dropped:
            dropped_files.append(fn)
        else:
            result[labels[i]].append(fn)
    return dict(result), dropped_files


def find_image(filename: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def to_yolo(regions, img_w: int, img_h: int, class_map) -> list[str]:
    """Convert VIA rects to normalised YOLO lines, clamped to the image."""
    lines = []
    for r in regions:
        shape = r["shape_attributes"]
        name = r.get("region_attributes", {}).get("name", "").strip()
        class_id = class_map(name)
        if class_id is None:
            continue

        x1 = max(0.0, float(shape["x"]))
        y1 = max(0.0, float(shape["y"]))
        x2 = min(float(img_w), x1 + float(shape["width"]))
        y2 = min(float(img_h), y1 + float(shape["height"]))
        if x2 <= x1 or y2 <= y1:
            continue

        lines.append(
            f"{class_id} "
            f"{((x1 + x2) / 2) / img_w:.6f} {((y1 + y2) / 2) / img_h:.6f} "
            f"{(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--via-json", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, nargs="+", required=True,
                    help="One or more directories to search for frames.")
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="Where the YOLO tree is built (gitignored).")
    ap.add_argument("--splits-dir", type=Path, required=True,
                    help="Where the split manifests are written (committed).")
    ap.add_argument("--yaml-out", type=Path, required=True)
    ap.add_argument("--guard", type=int, default=2,
                    help="Frames dropped either side of a split boundary.")
    ap.add_argument("--pattern", nargs="+", default=list(DEFAULT_PATTERN),
                    choices=["train", "val", "test"],
                    help="Split assignment per contiguous block.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy images instead of symlinking them.")
    ap.add_argument("--task", choices=["fish", "species"], default="fish",
                    help="Single 'fish' class, or per-species classes.")
    ap.add_argument("--min-boxes", type=int, default=60,
                    help="Boxes a species needs to get its own class "
                         "(--task species). Below this it joins other_fish.")
    args = ap.parse_args()

    pattern = tuple(args.pattern)
    annotations = parse_via(args.via_json)
    groups = group_by_video(annotations)

    class_map, class_names = build_class_map(
        annotations, args.task, args.min_boxes)
    print(f"Task: {args.task}  ({len(class_names)} classes: "
          f"{', '.join(class_names)})")

    splits: dict[str, list[str]] = defaultdict(list)
    all_dropped: list[str] = []
    print(f"Blocks per video: {' '.join(pattern)}   guard band: {args.guard} frames\n")
    for video, frames in sorted(groups.items()):
        assigned, dropped = assign_blocks(frames, pattern, args.guard)
        for split, files in assigned.items():
            splits[split].extend(files)
        all_dropped.extend(dropped)
        counts = "  ".join(f"{s}={len(assigned.get(s, [])):4}"
                           for s in ("train", "val", "test"))
        print(f"  {video}: {len(frames):4} frames  ->  {counts}  dropped={len(dropped)}")

    # Build the YOLO tree.
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = args.dataset_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Counter] = {s: Counter() for s in splits}
    missing: list[str] = []

    for split, filenames in splits.items():
        manifest = []
        for fn in sorted(filenames):
            src = find_image(fn, args.images_root)
            if src is None:
                missing.append(fn)
                continue

            dst = args.dataset_dir / "images" / split / fn
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())

            with Image.open(src) as im:
                w, h = im.size
            lines = to_yolo(annotations[fn], w, h, class_map)

            label = args.dataset_dir / "labels" / split / (Path(fn).stem + ".txt")
            label.write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")

            stats[split]["images"] += 1
            stats[split]["boxes"] += len(lines)
            for ln in lines:
                stats[split][f"class{ln.split()[0]}"] += 1
            if not lines:
                stats[split]["background"] += 1
            manifest.append(fn)

        (args.splits_dir / f"{split}.txt").write_text(
            "\n".join(manifest) + "\n", encoding="utf-8")

    if missing:
        print(f"\nWARNING: {len(missing)} annotated frames not found on disk, "
              f"e.g. {missing[:3]}", file=sys.stderr)

    args.yaml_out.parent.mkdir(parents=True, exist_ok=True)
    dataset_rel = Path(args.dataset_dir).resolve()
    args.yaml_out.write_text(
        f"# Generated by scripts/build_splits.py -- do not edit by hand.\n"
        f"path: {dataset_rel}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {len(class_names)}\n"
        f"names:\n"
        + "".join(f"  {i}: '{n}'\n" for i, n in enumerate(class_names)),
        encoding="utf-8",
    )

    print("\n" + "=" * 62)
    total_i = total_b = 0
    for split in ("train", "val", "test"):
        s = stats.get(split, Counter())
        total_i += s["images"]
        total_b += s["boxes"]
        print(f"  {split:5}  {s['images']:4} images  {s['boxes']:5} boxes  "
              f"{s['background']:4} background")
    print(f"  {'total':5}  {total_i:4} images  {total_b:5} boxes  "
          f"({len(all_dropped)} dropped to guard bands)")
    print("=" * 62)
    if len(class_names) > 1:
        print("\n  class                          train    val   test")
        for i, n in enumerate(class_names):
            print(f"  {i} {n:<28} "
                  + "".join(f"{stats.get(sp, Counter())[f'class{i}']:6}"
                            for sp in ("train", "val", "test")))

    print(f"\nSplit manifests: {args.splits_dir}")
    print(f"Dataset tree:    {args.dataset_dir}")
    print(f"Dataset config:  {args.yaml_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
