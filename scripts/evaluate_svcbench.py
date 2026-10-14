#!/usr/bin/env python3
"""Evaluate ICL versus Spatial-TTT on a small SVCBench trajectory subset.

The runner uses one source video and asks each published question at its
ordered query points.  At query time ``t`` the model receives only frames with
timestamps ``<= t``.  ICL sees a bounded recent window; Spatial-TTT receives
the same sampled frames incrementally and reads a text-only answer from its
fast weights.  This makes the test a state-maintenance probe rather than a
single final-video QA test.

The output is one JSONL row per condition/question/query point plus a summary
JSON containing exact count accuracy and the official SVCBench GPA, MoC, and
UDA-style trajectory metrics.  The official metric implementation remains the
reference; this copy also reports parse failures and coverage explicitly.

Example::

    conda run --no-capture-output -n meowbench python scripts/evaluate_svcbench.py \
      --annotations /data/quzitsix/svcbench_subset/subset/svcbench_subset.json \
      --video-root /data/quzitsix/svcbench_subset \
      --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
      --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
      --device cuda:2 --dtype bfloat16 \
      --output runs/svcbench_0030.jsonl

Use ``--dry-run`` to validate the grouped trajectories and local paths without
loading a model.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_frame.video import iter_video_chunks


CONDITIONS = ("icl", "spatial")
_INT_RE = re.compile(r"(?<![A-Za-z0-9_])[+-]?\d+(?:\.0+)?(?![A-Za-z0-9_])")


def _normalize_subtype(value: Any) -> str:
    aliases = {
        "O1-Snapshot": "O1-Snap",
        "E1-Act": "E1-Action",
        "E1-Trans": "E1-Transit",
        "E2-Episodic": "E2-Episode",
        "E2-Cyclic": "E2-Periodic",
    }
    value = str(value or "Unknown")
    return aliases.get(value, value)


def _load_subset(path: Path) -> list[dict[str, Any]]:
    """Load grouped subset JSON or reconstruct groups from flat eval JSONL."""

    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        groups: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
        for row in rows:
            source, video = str(row["source_dataset"]), str(row["video_path"])
            key = (f"{source}/{video}", str(row.get("id", row.get("q_id", ""))))
            if key not in groups:
                groups[key] = {
                    "id": key[1], "source_dataset": source, "video_path": video,
                    "question": row["question"], "counting_type": row.get("counting_type", ""),
                    "counting_subtype": row.get("counting_subtype", "Unknown"),
                    "counting_subtype_normalized": _normalize_subtype(row.get("counting_subtype")),
                    "query_points": [],
                    "video_file": f"videos/{source}/{video}",
                }
            groups[key]["query_points"].append({
                "q_id": str(row.get("q_id", "")),
                "query_index": int(row.get("query_index", 0)),
                "query_time": float(row["query_time"]),
                "count": int(row["count"]),
            })
        result = list(groups.values())
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("videos"), list):
            raise ValueError("expected svcbench_subset.json with a videos list")
        result = payload["videos"]

    if not result:
        raise ValueError(f"no questions found in {path}")
    for item in result:
        item["query_points"] = sorted(item["query_points"],
                                       key=lambda point: (float(point["query_time"]), int(point.get("query_index", 0))))
        if not item["query_points"]:
            raise ValueError(f"question {item.get('id')} has no query points")
        item.setdefault("counting_subtype_normalized", _normalize_subtype(item.get("counting_subtype")))
        item.setdefault("video_file", f"videos/{item['source_dataset']}/{item['video_path']}")
    return result


def _resolve_video(root: Path, item: dict[str, Any]) -> Path | None:
    rel = str(item.get("video_file", ""))
    source = str(item.get("source_dataset", ""))
    name = str(item.get("video_path", ""))
    candidates = [
        root / rel,
        root / "videos" / source / name,
        root / source / name,
        root / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _decode_samples(path: Path, *, chunk_seconds: float, frames_per_chunk: int) -> tuple[list[Any], list[float]]:
    images: list[Any] = []
    timestamps: list[float] = []
    for chunk in iter_video_chunks(path, chunk_seconds=chunk_seconds,
                                    frames_per_chunk=frames_per_chunk, max_side=10_000):
        images.extend(chunk.images)
        timestamps.extend(chunk.timestamps)
    if not images:
        raise RuntimeError(f"could not decode frames from {path}")
    return images, timestamps


def _resize(images: list[Any], max_side: int) -> list[Any]:
    result = []
    for image in images:
        image = image.convert("RGB").copy()
        image.thumbnail((max_side, max_side))
        result.append(image)
    return result


def _prefix_window(images: list[Any], timestamps: list[float], target: float,
                   *, window_seconds: float = 0.0, max_frames: int = 0) -> tuple[list[Any], list[float]]:
    eligible = [index for index, timestamp in enumerate(timestamps) if timestamp <= target + 1e-6]
    if not eligible:
        eligible = [0]
    if window_seconds > 0:
        start = target - window_seconds
        eligible = [index for index in eligible if timestamps[index] >= start - 1e-6] or [eligible[-1]]
    if max_frames > 0 and len(eligible) > max_frames:
        import numpy as np

        choices = np.rint(np.linspace(0, len(eligible) - 1, max_frames)).astype(int).tolist()
        eligible = [eligible[index] for index in choices]
    return [images[index] for index in eligible], [timestamps[index] for index in eligible]


def _parse_integer(text: str) -> int | None:
    if not text:
        return None
    if text.lstrip().startswith("__ERROR__"):
        return None
    match = _INT_RE.search(text)
    if not match:
        return None
    token = match.group(0)
    try:
        value = float(token)
        if not math.isfinite(value) or value != int(value):
            return None
        return int(value)
    except ValueError:
        return None


def _question(text: str) -> str:
    return f"{text}\nAnswer with exactly one integer number and no explanation."


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _trajectory_metrics(predictions: list[int | None], gts: list[int]) -> dict[str, Any]:
    valid = [(int(prediction), int(gold)) for prediction, gold in zip(predictions, gts)
             if prediction is not None]
    if not valid:
        return {"n_queries": len(gts), "n_valid": 0, "coverage": 0.0,
                "accuracy": 0.0, "gpa": None, "moc": None, "uda": None}
    pred = [item[0] for item in valid]
    gold = [item[1] for item in valid]
    gpa = sum(math.exp(-((p - g) ** 2) / (2 * (0.05 * max(g, 1)) ** 2))
               for p, g in zip(pred, gold)) / len(pred)
    moc = None
    if len(pred) >= 2:
        violation = next((index for index in range(len(pred) - 1) if pred[index + 1] < pred[index]), None)
        moc = 1.0 if violation is None else violation / (len(pred) - 1)
    uda = None
    if len(pred) >= 2:
        uda = sum(_sign(pred[index + 1] - pred[index]) == _sign(gold[index + 1] - gold[index])
                  for index in range(len(pred) - 1)) / (len(pred) - 1)
    return {
        "n_queries": len(gts), "n_valid": len(valid), "coverage": len(valid) / len(gts),
        "accuracy": sum(p == g for p, g in valid) / len(valid),
        "gpa": gpa, "moc": moc, "uda": uda,
    }


def _summarize(rows: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], str(row["question_id"]))].append(row)
    subtype = {str(item["id"]): item.get("counting_subtype_normalized", _normalize_subtype(item.get("counting_subtype")))
               for item in items}
    trajectories = {}
    for (condition, question_id), question_rows in grouped.items():
        question_rows.sort(key=lambda row: row["query_time"])
        metrics = _trajectory_metrics([row.get("prediction") for row in question_rows],
                                      [row["gold"] for row in question_rows])
        metrics["counting_subtype"] = subtype.get(question_id, "Unknown")
        trajectories[f"{condition}::{question_id}"] = metrics

    by_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        values = [metric for key, metric in trajectories.items() if key.startswith(condition + "::")]
        def mean(name: str):
            entries = [metric[name] for metric in values if metric.get(name) is not None]
            return sum(entries) / len(entries) if entries else None
        by_condition[condition] = {
            "n_questions": len(values),
            "accuracy": mean("accuracy"), "gpa": mean("gpa"),
            "moc": mean("moc"), "uda": mean("uda"), "coverage": mean("coverage"),
            "n_valid": sum(metric["n_valid"] for metric in values),
            "n_queries": sum(metric["n_queries"] for metric in values),
        }
    return {"by_condition": by_condition, "trajectories": trajectories}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-path", default="/data/quzitsix/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--spatial-checkpoint", default="/data/quzitsix/models/Spatial-TTT-nano/model.safetensors")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--frames-per-chunk", type=int, default=8)
    parser.add_argument("--icl-window-seconds", type=float, default=8.0,
                        help="bounded recent ICL context; 0 uses every observed sample")
    parser.add_argument("--icl-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--query-stride", type=int, default=1,
                        help="keep every k-th point in each trajectory (default: 1)")
    parser.add_argument("--max-query-points", type=int, default=0,
                        help="optional cap per trajectory after query-stride")
    parser.add_argument("--output", type=Path, default=Path("runs/svcbench_eval.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if (args.max_side < 1 or args.chunk_seconds <= 0 or args.frames_per_chunk < 1
            or args.icl_frames < 1 or args.query_stride < 1 or args.max_query_points < 0):
        raise SystemExit("invalid sampling or image-size budget")
    if not args.annotations.is_file():
        raise SystemExit(f"annotations do not exist: {args.annotations}")
    items = _load_subset(args.annotations)
    if args.limit_questions > 0:
        items = items[:args.limit_questions]
    resolved = []
    missing = []
    for item in items:
        path = _resolve_video(args.video_root, item)
        if path is None:
            missing.append({"question_id": item.get("id"), "video": item.get("video_file")})
        else:
            resolved.append((item, path))
    if args.dry_run:
        report = {
            "mode": "dry_run", "annotations": str(args.annotations), "video_root": str(args.video_root),
            "n_questions": len(items), "n_resolved": len(resolved), "missing": missing,
            "conditions": CONDITIONS, "icl_window_seconds": args.icl_window_seconds,
            "chunk_seconds": args.chunk_seconds, "frames_per_chunk": args.frames_per_chunk,
            "query_stride": args.query_stride, "max_query_points": args.max_query_points,
            "questions": [{"id": item["id"], "video": str(path),
                           "counting_subtype": item.get("counting_subtype_normalized"),
                           "n_query_points": len(item["query_points"]),
                           "first_query_time": item["query_points"][0]["query_time"],
                           "last_query_time": item["query_points"][-1]["query_time"]}
                          for item, path in resolved],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if resolved and not missing else 2
    if missing:
        raise SystemExit(f"{len(missing)} videos are missing; run download_svcbench_subset.sh first")

    from scripts.evaluate_fho_sta import _answer_icl
    from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory

    engine = SpatialVideoMemory(SpatialVideoConfig(
        model_path=args.model_path, device=args.device, dtype=args.dtype,
        spatial_checkpoint=args.spatial_checkpoint, local_files_only=True,
        max_side=args.max_side, chunk_seconds=args.chunk_seconds,
        frames_per_chunk=args.frames_per_chunk, max_new_tokens=args.max_new_tokens,
    ))
    rows: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    first_output = True

    for item, path in resolved:
        print(f"Decoding {path}", flush=True)
        images, timestamps = _decode_samples(path, chunk_seconds=args.chunk_seconds,
                                             frames_per_chunk=args.frames_per_chunk)
        images = _resize(images, args.max_side)
        selected_points = item["query_points"][::args.query_stride]
        if args.max_query_points:
            selected_points = selected_points[:args.max_query_points]
        query_times = sorted({float(point["query_time"]) for point in selected_points})
        print(f"Question {item['id']}: {len(selected_points)}/{len(item['query_points'])} query points", flush=True)
        # ICL and Spatial-TTT each get a fresh state.  The latter receives all
        # new sampled frames in chronological order and answers both questions
        # at each shared timestamp before any future frame is ingested.
        condition_answers: dict[str, dict[float, str]] = {"icl": {}, "spatial": {}}
        for target in query_times:
            prefix, prefix_times = _prefix_window(images, timestamps, target,
                                                  window_seconds=args.icl_window_seconds,
                                                  max_frames=args.icl_frames)
            prompt = _question(item["question"])
            try:
                condition_answers["icl"][target] = _answer_icl(
                    engine, prefix, prefix_times, prompt, max_new_tokens=args.max_new_tokens)
            except Exception as exc:
                condition_answers["icl"][target] = f"__ERROR__ {type(exc).__name__}: {exc}"

        engine.reset()
        spatial_seen = -1.0
        for target in query_times:
            new_indices = [index for index, timestamp in enumerate(timestamps)
                           if spatial_seen < timestamp <= target + 1e-6]
            if not new_indices and spatial_seen < 0:
                new_indices = [0]
            if new_indices:
                batch_images = [images[index] for index in new_indices]
                batch_times = [timestamps[index] for index in new_indices]
                engine.ingest_frames(batch_images, batch_times)
                spatial_seen = max(spatial_seen, batch_times[-1])
            try:
                condition_answers["spatial"][target] = engine.answer(
                    _question(item["question"]), use_memory=True, max_new_tokens=args.max_new_tokens)
            except Exception as exc:
                condition_answers["spatial"][target] = f"__ERROR__ {type(exc).__name__}: {exc}"

        with args.output.open("w" if first_output else "a", encoding="utf-8") as output:
            for point in selected_points:
                target = float(point["query_time"])
                for condition in CONDITIONS:
                    answer = condition_answers[condition][target]
                    prediction = _parse_integer(answer)
                    row = {
                        "schema": "ttt_frame.svcbench_eval/1",
                        "condition": condition, "question_id": str(item["id"]),
                        "q_id": point["q_id"], "source_dataset": item["source_dataset"],
                        "video_path": item["video_path"], "question": item["question"],
                        "counting_subtype": item.get("counting_subtype_normalized", item.get("counting_subtype")),
                        "query_index": point.get("query_index"), "query_time": target,
                        "gold": int(point["count"]), "answer": answer,
                        "prediction": prediction, "correct": prediction == int(point["count"]),
                        "max_side": args.max_side, "status": "ok" if prediction is not None else "parse_error",
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        first_output = False

    summary = {
        "schema": "ttt_frame.svcbench_summary/1", "annotations": str(args.annotations),
        "video_root": str(args.video_root), "model_path": args.model_path,
        "spatial_checkpoint": args.spatial_checkpoint, "n_questions": len(items),
        "conditions": CONDITIONS, "sampling": {
            "chunk_seconds": args.chunk_seconds, "frames_per_chunk": args.frames_per_chunk,
            "max_side": args.max_side, "icl_window_seconds": args.icl_window_seconds,
            "icl_frames": args.icl_frames,
            "query_stride": args.query_stride, "max_query_points": args.max_query_points,
        }, "summary": _summarize(rows, items),
        "note": "SVCBench trajectory smoke test; ICL uses a bounded recent window and Spatial-TTT receives the full sampled prefix.",
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
