#!/usr/bin/env python3
"""Run a matched five-arm SVCBench evaluation on real video prefixes.

Each query point is an independent episode.  Frames are sampled from the
complete observable prefix, uniformly capped to one shared budget, and then
used exactly once by visual ICL and Spatial-TTT.  The five arms are:

* ``blind``: text-only Qwen base context at stream position zero;
* ``icl``: one native-video encoding of the matched frame bundle;
* ``spatial``: text-only read from the episode fast weights;
* ``spatial_base``: text-only read with the Spatial branch bypassed;
* ``spatial_slow_only``: text-only read from the official learned initial
  state after discarding the episode fast weights.

All arms share one :class:`SpatialVideoMemory` instance and therefore the
same base model, processor, and official Spatial-TTT checkpoint.  For each
query, Spatial frames are written in small chronological chunks, but every
selected real frame is presented only once.  No frame later than
``query_time - cutoff_margin`` is admitted.

Example::

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_svcbench_matched.py \
      --annotations /data/quzitsix/svcbench_real_v1/subset/svcbench_subset.json \
      --video-root /data/quzitsix/svcbench_real_v1 \
      --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
      --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
      --device cuda:0 --num-shards 1 --shard-index 0 \
      --output runs/svcbench_matched.jsonl

Use ``--dry-run`` to validate paths, grouping, query selection, and sharding
without loading the model.  Existing outputs are never overwritten unless
``--overwrite`` is explicit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_svcbench import (
    _load_subset,
    _normalize_subtype,
    _parse_integer,
    _question,
    _resolve_video,
    _trajectory_metrics,
)
from ttt_frame.video import iter_video_chunks


CONDITIONS = ("blind", "icl", "spatial", "spatial_slow_only", "spatial_base")
SCHEMA = "ttt_frame.svcbench_matched/1"
SUMMARY_SCHEMA = "ttt_frame.svcbench_matched_summary/1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    """Hash a local model file or directory without retaining file contents."""

    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "files": 1,
            "bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
    if not resolved.is_dir():
        raise FileNotFoundError(f"model artifact does not exist: {resolved}")
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory contains no files: {resolved}")
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(resolved).as_posix()
        size = item.stat().st_size
        file_hash = _sha256_file(item)
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return {
        "path": str(resolved),
        "kind": "directory_tree",
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _video_key(item: dict[str, Any]) -> str:
    return f"{item.get('source_dataset', '')}/{item.get('video_path', '')}"


def _select_points(item: dict[str, Any], *, stride: int, maximum: int) -> list[dict[str, Any]]:
    points = list(item["query_points"])[::stride]
    return points[:maximum] if maximum else points


def _decode_until(
    path: Path,
    *,
    stop_time: float,
    chunk_seconds: float,
    frames_per_chunk: int,
    max_side: int,
) -> tuple[list[Any], list[float]]:
    """Decode chronological samples only through the chunk containing stop_time.

    The iterator samples the start of equal subintervals.  We additionally
    filter by the exact timestamp so a boundary chunk cannot expose the future.
    """

    if stop_time < 0:
        return [], []
    images: list[Any] = []
    timestamps: list[float] = []
    chunks = iter_video_chunks(
        path, chunk_seconds=chunk_seconds, frames_per_chunk=frames_per_chunk,
        max_side=max_side,
    )
    with closing(chunks):
        for chunk in chunks:
            for image, timestamp in zip(chunk.images, chunk.timestamps):
                if float(timestamp) <= stop_time:
                    images.append(image)
                    timestamps.append(float(timestamp))
            # This chunk contains the requested boundary.  Closing here keeps
            # the decoder from walking through the remainder of a long video.
            if (chunk.index + 1) * chunk_seconds > stop_time + 1e-6:
                break
    return images, timestamps


def _uniform_prefix(
    images: list[Any], timestamps: list[float], cutoff: float, max_frames: int,
) -> tuple[list[Any], list[float], list[int], int]:
    eligible = [index for index, timestamp in enumerate(timestamps)
                if timestamp <= cutoff]
    eligible_count = len(eligible)
    if max_frames and len(eligible) > max_frames:
        offsets = np.rint(np.linspace(0, len(eligible) - 1, max_frames)).astype(int).tolist()
        eligible = [eligible[offset] for offset in offsets]
    return (
        [images[index] for index in eligible],
        [timestamps[index] for index in eligible],
        eligible,
        eligible_count,
    )


def _frame_bundle_hash(images: list[Any], timestamps: list[float]) -> str:
    if len(images) != len(timestamps):
        raise ValueError("frame images and timestamps do not match")
    digest = hashlib.sha256()
    digest.update(b"ttt_frame.svcbench.frame_bundle/1\0")
    for image, timestamp in zip(images, timestamps):
        rgb = image.convert("RGB")
        digest.update(f"{float(timestamp):.9f}\0{rgb.width}x{rgb.height}\0RGB\0".encode())
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    """Split a stream without leaving a one- or two-frame temporal tail.

    Qwen3-VL pads video inputs shorter than three frames.  Merging a short
    remainder into the preceding write keeps Spatial-TTT's real-frame exposure
    identical to ICL's whenever the episode itself contains at least three
    frames.  Episodes with fewer than three frames are padded identically by
    both paths inside their respective video preprocessors.
    """

    if size < 4 or size % 2:
        raise ValueError("Spatial write chunks must use an even size of at least four")
    start = 0
    while len(values) - start > size + 2:
        yield values[start:start + size]
        start += size
    if start < len(values):
        yield values[start:]


def _processor_frame_count(real_frames: int) -> int:
    """Return Qwen3-VL's encoded frame count after temporal padding."""

    if real_frames <= 0:
        return 0
    locally_padded = max(3, real_frames)
    return locally_padded + locally_padded % 2


def _timed_question(text: str, *, query_time: float, cutoff_time: float) -> str:
    """Give every arm the same temporal prior exposed by video metadata."""

    timing = (
        f"The count is requested at video time {query_time:.2f} seconds. "
        f"The visible prefix ends at {cutoff_time:.2f} seconds."
    )
    return _question(f"{timing}\n{text}")


def _result(answer: str, gold: int, elapsed: float) -> dict[str, Any]:
    prediction = _parse_integer(answer)
    return {
        "answer": answer,
        "prediction": prediction,
        "correct": prediction == gold,
        "status": "ok" if prediction is not None else "parse_error",
        "error": None,
        "elapsed_sec": round(elapsed, 4),
    }


def _error(exc: BaseException, elapsed: float) -> dict[str, Any]:
    return {
        "answer": "",
        "prediction": None,
        "correct": False,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "elapsed_sec": round(elapsed, 4),
    }


def _call(started: float, gold: int, function, *args, **kwargs) -> dict[str, Any]:
    try:
        answer = function(*args, **kwargs)
        return _result(answer, gold, time.perf_counter() - started)
    except Exception as exc:  # retain an auditable row for partial model failures
        return _error(exc, time.perf_counter() - started)


def _summarize(rows: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["question_id"]))].append(row)
    subtype = {
        str(item["id"]): item.get(
            "counting_subtype_normalized", _normalize_subtype(item.get("counting_subtype")),
        )
        for item in items
    }
    trajectories: dict[str, Any] = {}
    for (condition, question_id), question_rows in grouped.items():
        question_rows.sort(key=lambda row: (float(row["query_time"]), int(row["query_index"])))
        metric = _trajectory_metrics(
            [row.get("prediction") for row in question_rows],
            [int(row["gold"]) for row in question_rows],
        )
        metric["counting_subtype"] = subtype.get(question_id, "Unknown")
        trajectories[f"{condition}::{question_id}"] = metric

    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        values = [value for key, value in trajectories.items()
                  if key.startswith(condition + "::")]

        def mean(name: str) -> float | None:
            entries = [float(value[name]) for value in values if value.get(name) is not None]
            return sum(entries) / len(entries) if entries else None

        total = len(condition_rows)
        valid = sum(row.get("prediction") is not None for row in condition_rows)
        by_condition[condition] = {
            "n_questions": len(values),
            "n_queries": total,
            "n_valid": valid,
            "n_errors": sum(row.get("status") == "error" for row in condition_rows),
            "coverage_micro": valid / total if total else 0.0,
            "accuracy_micro_all_queries": (
                sum(bool(row.get("correct")) for row in condition_rows) / total if total else 0.0
            ),
            "accuracy_macro_valid": mean("accuracy"),
            "gpa_macro": mean("gpa"),
            "moc_macro": mean("moc"),
            "uda_macro": mean("uda"),
            "coverage_macro": mean("coverage"),
        }
    return {"by_condition": by_condition, "trajectories": trajectories}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-path", default="/data/quzitsix/models/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--spatial-checkpoint",
        default="/data/quzitsix/models/Spatial-TTT-nano/model.safetensors",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-side", type=int, default=224)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument(
        "--max-frames", type=int, default=512,
        help="uniform frame cap over the complete observable prefix",
    )
    parser.add_argument(
        "--write-chunk-frames", type=int, default=8,
        help="number of selected frames in each Spatial-TTT ingest call",
    )
    parser.add_argument(
        "--cutoff-margin", type=float, default=0.0,
        help="exclude frames later than query_time minus this many seconds",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--max-query-points", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/svcbench_matched.jsonl"))
    parser.add_argument(
        "--summary-output", type=Path,
        help="default: OUTPUT with _summary.json replacing the .jsonl suffix",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.annotations.is_file():
        raise SystemExit(f"annotations do not exist: {args.annotations}")
    if not args.video_root.is_dir():
        raise SystemExit(f"video root does not exist: {args.video_root}")
    if (args.max_side < 1 or args.chunk_seconds <= 0 or args.frames_per_chunk < 1
            or args.max_frames < 1 or args.write_chunk_frames < 4
            or args.write_chunk_frames % 2
            or args.cutoff_margin < 0 or args.max_new_tokens < 1
            or args.query_stride < 1 or args.max_query_points < 0
            or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards):
        raise SystemExit("invalid sampling, generation, or sharding argument")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    items = _load_subset(args.annotations)

    # Partition stable video keys, never questions.  A video with multiple
    # question trajectories is consequently owned by exactly one shard.
    video_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        video_groups[_video_key(item)].append(item)
    ordered_keys = sorted(video_groups)
    selected_keys = [key for index, key in enumerate(ordered_keys)
                     if index % args.num_shards == args.shard_index]
    selected_items = [item for key in selected_keys for item in video_groups[key]]

    resolved: dict[str, Path] = {}
    missing: list[dict[str, str]] = []
    for key in selected_keys:
        path = _resolve_video(args.video_root, video_groups[key][0])
        if path is None:
            missing.append({"video_key": key, "video_file": str(video_groups[key][0].get("video_file", ""))})
        else:
            resolved[key] = path.resolve()

    selected_point_count = sum(
        len(_select_points(item, stride=args.query_stride, maximum=args.max_query_points))
        for item in selected_items
    )
    dry_report = {
        "mode": "dry_run",
        "schema": SCHEMA,
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": _sha256_file(args.annotations),
        "video_root": str(args.video_root.resolve()),
        "conditions": list(CONDITIONS),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "all_videos": len(ordered_keys),
        "selected_videos": len(selected_keys),
        "selected_questions": len(selected_items),
        "selected_query_points": selected_point_count,
        "expected_rows": selected_point_count * len(CONDITIONS),
        "missing": missing,
        "sampling": {
            "chunk_seconds": args.chunk_seconds,
            "frames_per_chunk": args.frames_per_chunk,
            "max_frames": args.max_frames,
            "max_side": args.max_side,
            "cutoff_margin": args.cutoff_margin,
            "write_chunk_frames": args.write_chunk_frames,
        },
        "videos": [
            {
                "video_key": key,
                "path": str(resolved[key]) if key in resolved else None,
                "question_ids": [str(item["id"]) for item in video_groups[key]],
                "query_points": sum(
                    len(_select_points(item, stride=args.query_stride,
                                       maximum=args.max_query_points))
                    for item in video_groups[key]
                ),
            }
            for key in selected_keys
        ],
    }
    if args.dry_run:
        print(json.dumps(dry_report, ensure_ascii=False, indent=2))
        return 0 if selected_items and not missing and selected_point_count else 2
    if not selected_items or not selected_point_count:
        raise SystemExit("the selected shard contains no query points")
    if missing:
        raise SystemExit(f"{len(missing)} selected videos are missing")

    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    if args.output.resolve() == summary_path.resolve():
        raise SystemExit("--output and --summary-output must be different files")
    for destination in (args.output, summary_path):
        if destination.exists() and not args.overwrite:
            raise SystemExit(f"output already exists (pass --overwrite): {destination}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    checkpoint_path = Path(args.spatial_checkpoint)
    if not model_path.exists():
        raise SystemExit(f"local model path does not exist: {model_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"official Spatial-TTT checkpoint does not exist: {checkpoint_path}")

    # Import model code only after dry-run validation.
    import torch
    import transformers
    from scripts.evaluate_fho_sta import _answer_icl
    from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory

    model_artifact = _artifact_fingerprint(model_path)
    checkpoint_artifact = _artifact_fingerprint(checkpoint_path)
    engine = SpatialVideoMemory(SpatialVideoConfig(
        model_path=str(model_path), device=args.device, dtype=args.dtype,
        spatial_checkpoint=str(checkpoint_path), local_files_only=True,
        max_side=args.max_side, chunk_seconds=args.chunk_seconds,
        frames_per_chunk=args.frames_per_chunk, max_new_tokens=args.max_new_tokens,
    ))
    if engine.spatial_checkpoint_hash != checkpoint_artifact["sha256"]:
        raise RuntimeError("loaded Spatial checkpoint hash differs from the requested artifact")

    created_at = datetime.now(timezone.utc).isoformat()
    runner_hash = _sha256_file(Path(__file__))
    sampling = {
        "decoder": "ttt_frame.video.iter_video_chunks",
        "chunk_seconds": args.chunk_seconds,
        "frames_per_chunk": args.frames_per_chunk,
        "prefix_selection": "uniform_over_all_sampled_frames",
        "max_frames": args.max_frames,
        "max_side": args.max_side,
        "cutoff_margin": args.cutoff_margin,
        "write_chunk_frames": args.write_chunk_frames,
        "query_stride": args.query_stride,
        "max_query_points": args.max_query_points,
    }
    comparison_payload = {
        "schema": SCHEMA,
        "runner_sha256": runner_hash,
        "annotations_sha256": _sha256_file(args.annotations),
        "model_artifact_sha256": model_artifact["sha256"],
        "base_config_fingerprint": engine.base_fingerprint,
        "spatial_checkpoint_sha256": checkpoint_artifact["sha256"],
        "conditions": list(CONDITIONS),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "sampling": sampling,
        "num_shards": args.num_shards,
    }
    comparison_fingerprint = hashlib.sha256(
        json.dumps(comparison_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_protocol = {
        **comparison_payload,
        "comparison_fingerprint": comparison_fingerprint,
        "created_at_utc": created_at,
        "git_revision": _git_revision(),
        "annotations": str(args.annotations.resolve()),
        "video_root": str(args.video_root.resolve()),
        "model_artifact": model_artifact,
        "spatial_checkpoint_artifact": checkpoint_artifact,
        "model_path": str(model_path.resolve()),
        "spatial_checkpoint": str(checkpoint_path.resolve()),
        "device": args.device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "shard_index": args.shard_index,
    }

    rows: list[dict[str, Any]] = []
    video_hashes: dict[Path, str] = {}
    mode = "w" if args.overwrite else "x"
    with args.output.open(mode, encoding="utf-8") as output:
        for video_key in selected_keys:
            path = resolved[video_key]
            video_hash = video_hashes.setdefault(path, _sha256_file(path))
            group = video_groups[video_key]
            point_sets = {
                str(item["id"]): _select_points(
                    item, stride=args.query_stride, maximum=args.max_query_points,
                )
                for item in group
            }
            stop_time = max(
                float(point["query_time"]) - args.cutoff_margin
                for points in point_sets.values() for point in points
            )
            print(f"Decoding {video_key} through t={max(0.0, stop_time):.3f}s", flush=True)
            decode_error: BaseException | None = None
            decoded_images: list[Any] = []
            decoded_times: list[float] = []
            try:
                decoded_images, decoded_times = _decode_until(
                    path, stop_time=max(0.0, stop_time),
                    chunk_seconds=args.chunk_seconds,
                    frames_per_chunk=args.frames_per_chunk, max_side=args.max_side,
                )
                if not decoded_images:
                    raise RuntimeError(f"no sampled frame at or before {stop_time:.6f}s")
            except Exception as exc:
                decode_error = exc

            for item in group:
                for point in point_sets[str(item["id"])]:
                    episode_started = time.perf_counter()
                    target = float(point["query_time"])
                    cutoff = target - args.cutoff_margin
                    prompt = _timed_question(
                        str(item["question"]), query_time=target, cutoff_time=cutoff,
                    )
                    images: list[Any] = []
                    timestamps: list[float] = []
                    selected_indices: list[int] = []
                    eligible_count = 0
                    bundle_hash: str | None = None
                    prefix_error = decode_error
                    if prefix_error is None:
                        images, timestamps, selected_indices, eligible_count = _uniform_prefix(
                            decoded_images, decoded_times, cutoff, args.max_frames,
                        )
                        if images:
                            bundle_hash = _frame_bundle_hash(images, timestamps)
                        else:
                            prefix_error = RuntimeError(
                                f"no sampled frame at or before cutoff {cutoff:.6f}s"
                            )

                    gold = int(point["count"])
                    results: dict[str, dict[str, Any]] = {}
                    spatial_ingested_frames = 0
                    spatial_encoded_frames = 0
                    spatial_write_calls_completed = 0
                    spatial_write_calls_planned = (
                        len(list(_chunks(images, args.write_chunk_frames))) if images else 0
                    )

                    # Blind and ICL start at stream position zero.  ICL uses
                    # the same official-checkpoint engine in base context.
                    engine.reset()
                    started = time.perf_counter()
                    results["blind"] = _call(
                        started, gold, engine.answer, prompt, use_memory=False,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if prefix_error is None:
                        engine.reset()
                        started = time.perf_counter()
                        results["icl"] = _call(
                            started, gold, _answer_icl, engine, images, timestamps, prompt,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        results["icl"] = _error(prefix_error, 0.0)

                    spatial_conditions = ("spatial", "spatial_base", "spatial_slow_only")
                    if prefix_error is None:
                        engine.reset()
                        try:
                            image_chunks = list(_chunks(images, args.write_chunk_frames))
                            time_chunks = list(_chunks(timestamps, args.write_chunk_frames))
                            for frame_chunk, time_chunk in zip(image_chunks, time_chunks):
                                engine.ingest_frames(frame_chunk, time_chunk)
                                spatial_ingested_frames += len(frame_chunk)
                                spatial_encoded_frames += _processor_frame_count(len(frame_chunk))
                                spatial_write_calls_completed += 1
                            engine.finish_ingest()
                            started = time.perf_counter()
                            results["spatial"] = _call(
                                started, gold, engine.answer, prompt, use_memory=True,
                                max_new_tokens=args.max_new_tokens,
                            )
                            started = time.perf_counter()
                            results["spatial_base"] = _call(
                                started, gold, engine.answer, prompt, use_memory=False,
                                max_new_tokens=args.max_new_tokens,
                            )
                            try:
                                engine.discard_fast_state()
                                started = time.perf_counter()
                                results["spatial_slow_only"] = _call(
                                    started, gold, engine.answer, prompt, use_memory=True,
                                    max_new_tokens=args.max_new_tokens,
                                )
                            except Exception as exc:
                                results["spatial_slow_only"] = _error(exc, 0.0)
                        except Exception as exc:
                            for condition in spatial_conditions:
                                results.setdefault(condition, _error(exc, 0.0))
                    else:
                        for condition in spatial_conditions:
                            results[condition] = _error(prefix_error, 0.0)

                    episode_id = (
                        f"{video_key}::{item['id']}::{point.get('q_id', point.get('query_index'))}"
                    )
                    common = {
                        "schema": SCHEMA,
                        "episode_id": episode_id,
                        "query_episode_independent": True,
                        "question_id": str(item["id"]),
                        "q_id": str(point.get("q_id", "")),
                        "query_index": int(point.get("query_index", 0)),
                        "source_dataset": str(item.get("source_dataset", "")),
                        "video_path": str(item.get("video_path", "")),
                        "video_file": str(path),
                        "video_sha256": video_hash,
                        "question": str(item["question"]),
                        "prompt": prompt,
                        "counting_subtype": item.get(
                            "counting_subtype_normalized",
                            _normalize_subtype(item.get("counting_subtype")),
                        ),
                        "query_time": target,
                        "cutoff_time": cutoff,
                        "gold": gold,
                        "sampled_prefix_frame_count": eligible_count,
                        "selected_frame_count": len(images),
                        "selected_sample_indices": selected_indices,
                        "frame_timestamps": timestamps,
                        "frame_bundle_sha256": bundle_hash,
                        "future_frames_filtered": True,
                        "max_decoded_timestamp": max(decoded_times) if decoded_times else None,
                        "run_protocol": run_protocol,
                    }
                    episode_elapsed = round(time.perf_counter() - episode_started, 4)
                    for condition in CONDITIONS:
                        result = results[condition]
                        is_icl = condition == "icl"
                        is_spatial = condition.startswith("spatial")
                        visual_attempted = is_icl and prefix_error is None
                        visual_completed = visual_attempted and result["status"] != "error"
                        row = {
                            **common,
                            "condition": condition,
                            **result,
                            "visual_frames_provided_to_answer": (
                                len(images) if visual_attempted else 0
                            ),
                            "visual_frames_encoded": (
                                _processor_frame_count(len(images)) if visual_completed else None
                            ),
                            "visual_padding_frames": (
                                _processor_frame_count(len(images)) - len(images)
                                if visual_completed else None
                            ),
                            "spatial_frames_ingested": (
                                spatial_ingested_frames if is_spatial else 0
                            ),
                            "spatial_frames_encoded": (
                                spatial_encoded_frames if is_spatial else 0
                            ),
                            "spatial_padding_frames": (
                                spatial_encoded_frames - spatial_ingested_frames
                                if is_spatial else 0
                            ),
                            "visual_encode_passes": 1 if visual_completed else 0,
                            "visual_encode_attempted": visual_attempted,
                            "spatial_write_calls": (
                                spatial_write_calls_completed if is_spatial else 0
                            ),
                            "spatial_write_calls_planned": (
                                spatial_write_calls_planned if is_spatial else 0
                            ),
                            "shared_spatial_write": (
                                is_spatial and spatial_write_calls_completed > 0
                            ),
                            "episode_memory_state": {
                                "blind": "no_visual_input_at_position_zero",
                                "icl": "matched_visual_context",
                                "spatial": "episode_fast_state",
                                "spatial_slow_only": "official_learned_initial_state",
                                "spatial_base": "spatial_branch_bypassed_after_ingest",
                            }[condition],
                            "episode_elapsed_sec": episode_elapsed,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        rows.append(row)
                        print(json.dumps({
                            "episode_id": episode_id,
                            "condition": condition,
                            "gold": gold,
                            "prediction": result["prediction"],
                            "status": result["status"],
                        }, ensure_ascii=False), flush=True)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "run_protocol": run_protocol,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256_file(args.output),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "n_videos": len(selected_keys),
        "n_questions": len(selected_items),
        "n_query_points": selected_point_count,
        "n_rows": len(rows),
        "expected_rows": selected_point_count * len(CONDITIONS),
        "diagnostic_summary": _summarize(rows, selected_items),
        "diagnostic_summary_status": (
            "Non-authoritative per-shard progress summary. Use "
            "scripts/summarize_svcbench_matched.py for official subtype gates, "
            "failure-preserving trajectories, and cross-shard validation."
        ),
        "note": (
            "Matched real-video prefixes; every query point is reset independently. "
            "ICL and all Spatial ablations share the same selected frame bundle."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] != "error" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
