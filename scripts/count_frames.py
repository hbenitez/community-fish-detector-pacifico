#!/usr/bin/env python3
"""Count the total number of video frames in a media file using ffprobe."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def count_frames(video_path: Path) -> int:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH (install ffmpeg).")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"ffprobe returned no frame count for {video_path}")
    return int(output)


def get_duration(video_path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH (install ffmpeg).")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"ffprobe returned no duration for {video_path}")
    return float(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "data" / "videos" / "0514.mp4",
        help="Path to the video file (defaults to data/videos/0514.mp4).",
    )
    parser.add_argument(
        "--duration",
        action="store_true",
        help="Also print the video duration in seconds.",
    )
    args = parser.parse_args()

    frames = count_frames(args.video)
    if args.duration:
        duration = get_duration(args.video)
        print(f"frames: {frames}")
        print(f"duration: {duration:.3f}s")
    else:
        print(frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
