#!/usr/bin/env python3
"""Render observed-prefix frames for already frozen, post-scoring answer examples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PIL import Image, ImageDraw
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_features import causal_video_frames


def render(root: Path, examples_path: Path, output: Path, limit: int = 8) -> dict:
    if output.exists():
        raise FileExistsError("use a new context output directory")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("output must be outside benchmark")
    examples = json.loads(examples_path.read_text())
    ids = [row["id"] for row in examples[:limit]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("need nonempty unique example IDs")
    cases = {case.id: case for case in load_forecast_cases(root, ids=ids)}
    output.mkdir(parents=True)
    records = []
    # None of these times contains cutoff/future media. This is posthoc inspection.
    grid = [("context", 1740.), ("recent", 0.), ("recent", 10.),
            ("recent", 20.), ("recent", 25.), ("recent", 29.5)]
    for sid in ids:
        case = cases[sid]
        decoded = {}
        for role, media in (("context", case.context_path), ("recent", case.recent_path)):
            times = [time for r, time in grid if r == role]
            frames = list(causal_video_frames(media, times, max_side=448))
            if len(frames) != len(times):
                raise ValueError("missing observed frames")
            for requested, (im, actual) in zip(times, frames):
                if actual > requested + 1e-9:
                    raise ValueError("noncausal decoded frame")
                decoded[(role, requested)] = im, actual
        canvas = Image.new("RGB", (960, 570), "#16181d")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"{sid} | observed prefix only | cutoff = 1830 s", fill="white")
        frame_records = []
        for i, (role, requested) in enumerate(grid):
            im, actual = decoded[(role, requested)]
            prefix_requested = requested + (1800 if role == "recent" else 0)
            prefix_actual = actual + (1800 if role == "recent" else 0)
            thumb = im.convert("RGB").copy()
            thumb.thumbnail((312, 238))
            x, y = (i % 3) * 320 + 4, (i // 3) * 265 + 38
            draw.text((x, y), f"t={prefix_actual:.2f}s (request {prefix_requested:.1f})", fill="white")
            canvas.paste(thumb, (x+(312-thumb.width)//2, y+20+(238-thumb.height)//2))
            frame_records.append({"role": role, "requested_prefix_seconds": prefix_requested,
                "actual_prefix_seconds": prefix_actual, "decoded_pixel_sha256": hashlib.sha256(im.convert("RGB").tobytes()).hexdigest(),
                "image_size": im.size})
        path = output / f"{sid}.jpg"
        canvas.save(path, quality=88)
        records.append({"id": sid, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frames": frame_records})
    manifest = {"schema": "ideax_posthoc_context_v1", "analysis_only": True, "future_media_read": False,
                "example_source": str(examples_path), "example_source_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": records,
                "warning": "Post-scoring example review, not extra model evidence or human verification of future labels."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    result = render(args.root, args.examples, args.output_dir, args.limit)
    print(json.dumps({"cases": len(result["cases"]), "future_media_read": False}))
