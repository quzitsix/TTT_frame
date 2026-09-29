#!/usr/bin/env python3
"""Build a chronological Spatial-TTT memory with restartable checkpoints.

The regular ``spatial_videoqa ingest`` command saves only after every input
video has finished.  A 1080-second recording is long enough that losing the
last hour of work is inconvenient, so this driver commits one checkpoint per
60-second source clip and keeps only the newest checkpoint.  It is an online
fast-weight write; it does not train the official slow weights.

Example::

    python scripts/build_spatial_memory.py \
      --output runs/q9_full_history_spatial_official_4s8f \
      --progress runs/q9_full_history_spatial_official_4s8f.progress \
      --device cuda:0 --chunk-seconds 4 --frames-per-chunk 8

If the process stops, run the same command again.  The latest completed clip
is loaded and only the remaining clips are processed.  Use
``--reset-progress`` to deliberately start over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

# Make the repository root importable when this file is executed as
# ``python scripts/build_spatial_memory.py`` rather than with ``-m``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_frame.spatial_videoqa import SpatialModelConfig, SpatialVideoConfig, SpatialVideoMemory

from scripts.watch_supermemory import DEFAULT_MEDIA_INDEX, DEFAULT_MEDIA_ROOT, DEFAULT_MEMORY_ID
from scripts.watch_supermemory import load_clips, select_clips


DEFAULT_MODEL = "/data/quzitsix/models/Qwen3-VL-2B-Instruct"
DEFAULT_CHECKPOINT = "/data/quzitsix/models/Spatial-TTT-nano/model.safetensors"


def _remove_tree(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"expected directory: {path}")
        shutil.rmtree(path)


def _atomic_save(memory: SpatialVideoMemory, progress: Path) -> None:
    """Save one checkpoint and atomically replace ``progress/latest``."""

    progress.mkdir(parents=True, exist_ok=True)
    temporary = progress / ".next"
    previous = progress / ".previous"
    _remove_tree(temporary)
    _remove_tree(previous)
    memory.save(temporary)
    latest = progress / "latest"
    if latest.exists():
        latest.rename(previous)
    temporary.rename(latest)
    _remove_tree(previous)


def _read_completed(progress: Path) -> int:
    marker = progress / "progress.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            value = payload.get("completed_clips")
            if type(value) is int and value >= 0:
                return value
        except (OSError, ValueError, TypeError):
            pass
    latest = progress / "latest" / "memory.json"
    if latest.is_file():
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            value = payload.get("stats", {}).get("sessions")
            if type(value) is int and value >= 0:
                return value
        except (OSError, ValueError, TypeError):
            pass
    return 0


def _write_marker(progress: Path, *, completed: int, report: dict) -> None:
    marker = progress / "progress.json"
    payload = {
        "completed_clips": completed,
        "last_report": report,
        "updated_unix": time.time(),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _config(args: argparse.Namespace) -> SpatialVideoConfig:
    spatial = SpatialModelConfig(
        num_heads=args.num_heads,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        base_lr=args.base_lr,
        ttt_scale_init=0.0,
        seed=args.seed,
    )
    return SpatialVideoConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        local_files_only=not args.allow_download,
        spatial_checkpoint=args.spatial_checkpoint,
        chunk_seconds=args.chunk_seconds,
        frames_per_chunk=args.frames_per_chunk,
        max_side=args.max_side,
        max_chunks=args.max_chunks,
        spatial=spatial,
    )


def build(args: argparse.Namespace) -> int:
    output = Path(args.output)
    progress = Path(args.progress)
    if args.reset_progress:
        _remove_tree(progress)
    if output.exists() and not args.overwrite:
        raise ValueError(f"output already exists: {output}; use --overwrite to replace it")

    clips = select_clips(
        load_clips(
            Path(args.media_index),
            memory_id=args.memory_id,
            media_root=Path(args.media_root),
        ),
        args.start,
        args.end,
    )
    completed = _read_completed(progress)
    if completed > len(clips):
        raise ValueError(f"progress says {completed} clips, but selection has only {len(clips)}")

    config = _config(args)
    memory = SpatialVideoMemory(config)
    latest = progress / "latest"
    if completed:
        if not latest.is_dir():
            raise ValueError(f"progress marker exists but checkpoint is missing: {latest}")
        memory.load_memory(latest)
        print(json.dumps({"resumed_after_clip": completed}), flush=True)

    for clip in clips[completed:]:
        started = time.perf_counter()
        report = memory.ingest_video(clip.path)
        report = {
            "clip": clip.ordinal,
            "start_sec": clip.start_sec,
            "end_sec": clip.end_sec,
            **report,
            "wall_seconds": time.perf_counter() - started,
        }
        _atomic_save(memory, progress)
        completed += 1
        _write_marker(progress, completed=completed, report=report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    if output.exists():
        _remove_tree(output)
    shutil.copytree(progress / "latest", output)
    print(json.dumps({"saved": str(output), "completed_clips": completed}, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="final memory directory")
    p.add_argument("--progress", help="restartable checkpoint directory")
    p.add_argument("--reset-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--media-index", default=str(DEFAULT_MEDIA_INDEX))
    p.add_argument("--memory-id", default=DEFAULT_MEMORY_ID)
    p.add_argument("--media-root", default=str(DEFAULT_MEDIA_ROOT))
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--spatial-checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--chunk-seconds", type=float, default=4.0)
    p.add_argument("--frames-per-chunk", type=int, default=8)
    p.add_argument("--max-side", type=int, default=448)
    p.add_argument("--max-chunks", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=2648)
    p.add_argument("--window-size", type=int, default=2648)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-download", action="store_true")
    return p


if __name__ == "__main__":
    try:
        raise SystemExit(build(parser().parse_args()))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
