#!/usr/bin/env python3
"""Mine HomeSentinel C2/C3 temporal candidates for later human annotation.

The output is a review queue, not an evaluation set.  Automatic captions and
object records are never promoted to ground-truth answers by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ttt_frame.predictive_dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT,
    build_homesentinel_temporal_manifest,
    write_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--merged-captions",
        type=Path,
        help="defaults to <data-root>/merged_captions.json",
    )
    parser.add_argument(
        "--objects-dir",
        type=Path,
        help="defaults to <data-root>/benchmark_tmp",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-gap",
        "--min-gap-seconds",
        dest="min_gap_seconds",
        type=float,
        default=10.0,
        help="minimum non-overlapping interval between selected observations",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 writes every candidate")
    parser.add_argument(
        "--video-filename",
        default="raw_video.mp4",
        help="source media name inside each video directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.min_gap_seconds) or args.min_gap_seconds < 0 or args.limit < 0:
        raise SystemExit("--min-gap and --limit must be nonnegative")
    manifest = build_homesentinel_temporal_manifest(
        args.data_root,
        merged_captions_path=args.merged_captions,
        objects_dir=args.objects_dir,
        min_gap_seconds=args.min_gap_seconds,
        limit=args.limit,
        video_filename=args.video_filename,
    )
    write_manifest(manifest, args.output)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"wrote candidate review queue: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
