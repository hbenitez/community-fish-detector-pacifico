#!/usr/bin/env python3
"""Subsample video frames at a given rate using ffmpeg, one folder per video."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def subsample(video_path: Path, output_dir: Path, fps: float) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (install ffmpeg).")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    video_out_dir = output_dir / video_path.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-qscale:v", "2",
            str(video_out_dir / "frame_%06d.jpg"),
        ],
        check=True,
    )

    return video_out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos",
        type=Path,
        nargs="+",
        help="Video files to subsample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Base directory for per-video frame folders (default: data/).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Frames per second to extract (e.g. 1 for one frame per second).",
    )
    args = parser.parse_args()

    for video in args.videos:
        out = subsample(video, args.output_dir, args.fps)
        count = sum(1 for _ in out.glob("frame_*.jpg"))
        print(f"{video.name}: {count} frames -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
