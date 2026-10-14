#!/usr/bin/env python3
"""Run a small, leakage-safe FHO-STA prediction smoke test.

This is intentionally a boundary test rather than the official FHO detector
submission.  It turns one future object/verb annotation into a four-choice
prediction and compares the same Qwen3-VL-2B checkpoint under:

* ``blind``: the final observed frame only;
* ``icl_4f`` / ``icl_8f`` / ``icl_8s4f``: the observed prefix window remains in
  context;
* ``spatial_8s4f`` / ``spatial_4s8f``: the prefix is written into Spatial-TTT
  fast weights, then the video is discarded before a text-only query.

All decoded frames satisfy ``frame_index <= observed_frame``.  This makes the
runner useful before the official box/mAP pipeline is installed.  The output
must not be described as official STA mAP: it reports closed-set noun/verb
accuracy, with time-to-contact buckets for diagnostics.

Example::

    conda run --no-capture-output -n meowbench python scripts/evaluate_fho_sta.py \
      --annotations /data/quzitsix/ego4d_fho_subset/fho_sta_subset_16/fho_sta_val_subset.json \
      --video-root /data/quzitsix/ego4d_fho_subset/v2/clips \
      --frame-field clip_frame \
      --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
      --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
      --device cuda:2 --max-side 448 --limit 16 --output runs/fho_sta_smoke_16.jsonl

Use ``--dry-run`` to validate the annotation schema, selected horizons, and
video paths without loading a model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_fho_sta_subset import HORIZON_BINS, select_annotations


CONDITIONS = (
    "blind",
    "icl_4f",
    "icl_8f",
    "icl_8s4f",
    "spatial_8s4f",
    "spatial_4s8f",
)
TASKS = ("noun", "verb")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        raise ValueError("expected a FHO annotation object with an annotations list")
    return payload


def _video_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the per-video metadata mapping used by the annotation export.

    Official FHO-STA JSON stores this mapping in the top-level ``videos``
    field.  Some converted copies use ``info.video_metadata`` or a top-level
    ``video_metadata`` field, so retain those fallbacks.  The FPS is used to
    convert labelled frame indices to timestamps; silently falling back to 30
    FPS can move the sampled observation boundary on non-30-FPS videos.
    """
    metadata = payload.get("videos")
    if isinstance(metadata, dict):
        return metadata
    info = payload.get("info")
    if isinstance(info, dict):
        metadata = info.get("video_metadata")
        if isinstance(metadata, dict):
            return metadata
    metadata = payload.get("video_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _category_map(payload: dict[str, Any], kind: str) -> dict[int, str]:
    """Accept the list/dict category shapes used by Ego4D v1/v2 exports."""

    info = payload.get("info", {})
    if not isinstance(info, dict):
        info = {}
    taxonomy = info.get("taxonomy", {})
    if not isinstance(taxonomy, dict):
        taxonomy = {}
    candidates = (
        payload.get(f"{kind}_categories"),
        info.get(f"{kind}_categories"),
        taxonomy.get(f"{kind}_categories"),
    )
    raw = next((item for item in candidates if item is not None), None)
    if isinstance(raw, dict):
        result = {}
        for key, value in raw.items():
            try:
                result[int(key)] = str(value.get("name", value) if isinstance(value, dict) else value)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(raw, list):
        result = {}
        for index, value in enumerate(raw):
            if isinstance(value, dict):
                key = value.get("id", value.get("category_id", index))
                name = value.get("name", value.get("label", value.get("text", key)))
            else:
                key, name = index, value
            try:
                result[int(key)] = str(name)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def _objects(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    return [obj for obj in annotation.get("objects", []) if isinstance(obj, dict)
            and isinstance(obj.get("noun_category_id"), int)
            and isinstance(obj.get("verb_category_id"), int)]


def _metadata_for(annotation: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    uid = str(annotation.get("video_uid", annotation.get("video_id", "")))
    item = metadata.get(uid, {})
    return item if isinstance(item, dict) else {}


def _resolve_video(root: Path, annotation: dict[str, Any]) -> Path | None:
    names = []
    for key in ("video_uid", "video_id", "clip_uid"):
        value = annotation.get(key)
        if value:
            names.append(str(value))
    for name in dict.fromkeys(names):
        for candidate in (root / f"{name}.mp4", root / name / f"{name}.mp4", root / name):
            if candidate.is_file():
                return candidate
    # The selected subset is small; a recursive lookup is acceptable and
    # avoids assuming which CLI layout a dataset release uses.
    for name in dict.fromkeys(names):
        matches = list(root.rglob(f"{name}.mp4"))
        if matches:
            return matches[0]
    return None


def _frame_end(annotation: dict[str, Any], video_path: Path, *, frame_field: str) -> int:
    if frame_field == "frame":
        value = annotation.get("frame")
    elif frame_field == "clip_frame":
        value = annotation.get("clip_frame")
    elif video_path.stem == str(annotation.get("clip_uid", "")) and annotation.get("clip_frame") is not None:
        value = annotation["clip_frame"]
    else:
        value = annotation.get("frame", annotation.get("clip_frame"))
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"annotation {annotation.get('uid')} has no usable frame index")
    return max(0, int(value))


def decode_window(path: Path, end_frame: int, fps: float, *, seconds: float, count: int):
    """Decode an observed-only window and return PIL images plus timestamps.

    Spatial resizing is deliberately kept outside this decoder.  The same
    decoded prefix can then be resized once and shared by the ICL and
    parameter-memory conditions, which makes a resolution sweep comparable.
    """

    import av

    if fps <= 0 or seconds <= 0 or count <= 0:
        raise ValueError("fps, seconds, and count must be positive")
    start = max(0, int(np.floor(end_frame - seconds * fps)) + 1)
    targets = np.rint(np.linspace(start, end_frame, count)).astype(int).tolist()
    targets = sorted(set(targets))
    frames: list[Any] = []
    timestamps: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index > end_frame:
                break
            if index in targets:
                frames.append(frame.to_image().convert("RGB"))
                timestamps.append(index / fps)
            if frames and index >= targets[-1]:
                break
    if not frames:
        raise RuntimeError(f"could not decode an observed frame from {path} at {end_frame}")
    while len(frames) < count:
        frames.append(frames[-1].copy())
        timestamps.append(timestamps[-1] + 0.001)
    return frames, timestamps


def _resize_images(images, max_side: int):
    """Bound the spatial input presented to both ICL and Spatial-TTT.

    ``SpatialVideoMemory.ingest_video`` applies this bound through
    :func:`iter_video_chunks`, but the FHO runner uses ``ingest_frames`` so it
    can stop exactly at the labelled observation frame.  Applying the same
    PIL thumbnail here closes that otherwise easy-to-miss discrepancy.  PIL
    preserves the aspect ratio and never upsamples an already smaller frame.
    """

    if max_side < 1:
        raise ValueError("max_side must be positive")
    result = []
    for image in images:
        image = image.convert("RGB").copy()
        image.thumbnail((max_side, max_side))
        result.append(image)
    return result


def _options(gold: str, names: list[str], seed: int) -> tuple[list[str], int]:
    import random

    clean = [name for name in dict.fromkeys(names) if name]
    if gold not in clean:
        clean.insert(0, gold)
    rng = random.Random(seed)
    distractors = [name for name in clean if name != gold]
    rng.shuffle(distractors)
    choices = [gold] + distractors[:3]
    while len(choices) < 4:
        choices.append(f"class_{len(choices)}")
    rng.shuffle(choices)
    return choices, choices.index(gold)


def _question(task: str, choices: list[str]) -> str:
    label = "object noun" if task == "noun" else "action verb"
    lines = [
        "This is a short-term first-person anticipation task.",
        f"Which {label} is most likely to be involved next?",
        "Use only the observed frames. Output exactly one option letter.",
        "Options:",
    ]
    lines.extend(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    return "\n".join(lines)


def _parse_choice(text: str, choices: list[str]) -> int | None:
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return ord(match.group(1)) - ord("A")
    lowered = text.lower()
    for index, choice in enumerate(choices):
        if choice.lower() in lowered:
            return index
    return None


def _video_batch(engine, images, timestamps, question: str):
    import torch
    from transformers.video_utils import VideoMetadata

    # Qwen3-VL's video processor expects at least a short temporal sequence.
    # The blind condition intentionally exposes one observed frame only, so
    # repeat that frame for preprocessing while keeping all repeated timestamps
    # at or after the same observed endpoint.
    images = list(images)
    timestamps = list(timestamps)
    if not images:
        raise ValueError("at least one image is required")
    while len(images) < 3:
        images.append(images[-1].copy())
        timestamps.append(timestamps[-1] + 0.001)
    frames = np.stack([np.asarray(image.convert("RGB")) for image in images])
    indices = [int(round(timestamp * 1000)) for timestamp in timestamps]
    metadata = VideoMetadata(total_num_frames=max(indices[-1] + 1, len(images)),
                             fps=1000.0, frames_indices=indices)
    prompt = engine.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                           {"type": "text", "text": question}]}],
        tokenize=False, add_generation_prompt=True,
    )
    batch = engine.processor(text=[prompt], videos=[frames], video_metadata=[metadata],
                             do_sample_frames=False, return_tensors="pt")
    return engine._move(batch)


def _answer_icl(engine, images, timestamps, question: str, *, max_new_tokens: int) -> str:
    import torch

    batch = _video_batch(engine, images, timestamps, question)
    input_length = batch["input_ids"].shape[1]
    engine._clear_generation_state()
    try:
        with torch.no_grad(), engine.controller.context("base"):
            generated = engine.model.generate(**batch, max_new_tokens=max_new_tokens,
                                               do_sample=False, use_cache=True)
        return engine.processor.batch_decode(generated[:, input_length:],
                                             skip_special_tokens=True)[0].strip()
    finally:
        engine._clear_generation_state()


def _answer_spatial(engine, images, timestamps, question: str, *, max_new_tokens: int) -> str:
    engine.reset()
    engine.ingest_frames(images, timestamps)
    engine.finish_ingest()
    return engine.answer(question, use_memory=True, max_new_tokens=max_new_tokens)


def _condition_inputs(engine, path: Path, end_frame: int, fps: float, *, max_side: int):
    cache = {}
    cache["blind"] = decode_window(path, end_frame, fps, seconds=0.1, count=1)
    cache["icl_4f"] = decode_window(path, end_frame, fps, seconds=4.0, count=4)
    cache["icl_8f"] = decode_window(path, end_frame, fps, seconds=4.0, count=8)
    cache["icl_8s4f"] = decode_window(path, end_frame, fps, seconds=8.0, count=4)
    cache["spatial_8s4f"] = decode_window(path, end_frame, fps, seconds=8.0, count=4)
    cache["spatial_4s8f"] = decode_window(path, end_frame, fps, seconds=4.0, count=8)
    return {
        condition: (_resize_images(images, max_side), timestamps)
        for condition, (images, timestamps) in cache.items()
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['condition']}::{row['task']}"
        item = buckets.setdefault(key, {"n": 0, "correct": 0, "errors": 0,
                                        "horizons": {}})
        if row.get("status") != "ok":
            item["errors"] += 1
            continue
        item["n"] += 1
        item["correct"] += int(row["correct"])
        horizon = str(row["horizon_bin"])
        h = item["horizons"].setdefault(horizon, {"n": 0, "correct": 0})
        h["n"] += 1
        h["correct"] += int(row["correct"])
    for item in buckets.values():
        item["accuracy"] = round(item["correct"] / item["n"], 4) if item["n"] else 0.0
        for h in item["horizons"].values():
            h["accuracy"] = round(h["correct"] / h["n"], 4) if h["n"] else 0.0
    return buckets


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-path", default="/data/quzitsix/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--spatial-checkpoint", default="/data/quzitsix/models/Spatial-TTT-nano/model.safetensors")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-field", choices=("auto", "frame", "clip_frame"), default="auto")
    parser.add_argument("--max-side", type=int, default=448,
                        help="maximum PIL input edge for every ICL and Spatial condition")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("runs/fho_sta_smoke.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_side < 1:
        raise SystemExit("max-side must be positive")
    if not args.annotations.is_file():
        raise SystemExit(f"annotations do not exist: {args.annotations}")
    payload = _load(args.annotations)
    selected = select_annotations(payload, limit=args.limit, seed=args.seed)
    metadata = _video_metadata(payload)
    noun_map = _category_map(payload, "noun")
    verb_map = _category_map(payload, "verb")

    items = []
    missing = []
    for annotation in selected:
        path = _resolve_video(args.video_root, annotation)
        video = _metadata_for(annotation, metadata)
        fps = float(video.get("fps", 0) or 0)
        if fps <= 0:
            fps = 30.0
        end_frame = None
        if path is not None:
            try:
                end_frame = _frame_end(annotation, path, frame_field=args.frame_field)
            except ValueError as exc:
                missing.append({"uid": annotation.get("uid"), "reason": str(exc)})
                continue
        if path is None:
            missing.append({"uid": annotation.get("uid"), "reason": "video not found"})
            continue
        objects = _objects(annotation)
        if not objects:
            continue
        target = min(objects, key=lambda obj: float(obj.get("time_to_contact", 999999)))
        noun_id, verb_id = int(target["noun_category_id"]), int(target["verb_category_id"])
        noun = noun_map.get(noun_id, f"noun_{noun_id}")
        verb = verb_map.get(verb_id, f"verb_{verb_id}")
        ttc = float(target.get("time_to_contact", 0.0))
        horizon = min(HORIZON_BINS, key=lambda value: abs(value - ttc))
        items.append({"annotation": annotation, "path": path, "fps": fps,
                      "end_frame": end_frame, "noun": noun, "verb": verb,
                      "noun_id": noun_id, "verb_id": verb_id, "ttc": ttc,
                      "horizon_bin": horizon})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report = {"mode": "dry_run", "selected": len(selected), "usable": len(items),
                  "max_side": args.max_side,
                  "missing": missing, "conditions": CONDITIONS,
                  "items": [{"uid": item["annotation"].get("uid"),
                             "video_uid": item["annotation"].get("video_uid"),
                             "frame": item["end_frame"], "fps": item["fps"],
                             "ttc": item["ttc"], "horizon_bin": item["horizon_bin"],
                             "noun": item["noun"], "verb": item["verb"],
                             "video": str(item["path"])} for item in items]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        # A partial subset is useful for inspection, but should not look like
        # a ready-to-run experiment in shell pipelines.  Require every
        # selected item to resolve before reporting dry-run success.
        return 0 if items and not missing else 2

    if missing:
        raise SystemExit(f"{len(missing)} selected videos are missing; run the selective download first")
    if not items:
        raise SystemExit("no usable FHO-STA items")

    from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory

    engine = SpatialVideoMemory(SpatialVideoConfig(
        model_path=args.model_path, device=args.device, dtype=args.dtype,
        spatial_checkpoint=args.spatial_checkpoint, local_files_only=True,
        max_side=args.max_side, max_new_tokens=args.max_new_tokens,
    ))
    all_nouns = list(noun_map.values())
    all_verbs = list(verb_map.values())
    rows: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as output:
        for item_index, item in enumerate(items):
            started = time.perf_counter()
            inputs = _condition_inputs(engine, item["path"], item["end_frame"], item["fps"],
                                       max_side=args.max_side)
            for condition in CONDITIONS:
                for task, gold, names in (("noun", item["noun"], all_nouns),
                                           ("verb", item["verb"], all_verbs)):
                    choices, gold_index = _options(gold, names, args.seed + item_index * 31 + len(task))
                    question = _question(task, choices)
                    images, timestamps = inputs[condition]
                    try:
                        if condition == "blind":
                            answer = _answer_icl(engine, images[-1:], timestamps[-1:], question,
                                                 max_new_tokens=args.max_new_tokens)
                        elif condition.startswith("icl"):
                            answer = _answer_icl(engine, images, timestamps, question,
                                                 max_new_tokens=args.max_new_tokens)
                        else:
                            answer = _answer_spatial(engine, images, timestamps, question,
                                                     max_new_tokens=args.max_new_tokens)
                        prediction = _parse_choice(answer, choices)
                        row = {
                            "uid": item["annotation"].get("uid"),
                            "video_uid": item["annotation"].get("video_uid"),
                            "condition": condition, "task": task,
                            "max_side": args.max_side,
                            "observed_frame": item["end_frame"], "fps": item["fps"],
                            "ttc": item["ttc"], "horizon_bin": item["horizon_bin"],
                            "gold": gold, "gold_index": gold_index,
                            "choices": choices, "answer": answer,
                            "prediction_index": prediction,
                            "correct": prediction == gold_index,
                            "elapsed_sec": round(time.perf_counter() - started, 4),
                            "status": "ok",
                        }
                    except Exception as exc:
                        row = {
                            "uid": item["annotation"].get("uid"),
                            "video_uid": item["annotation"].get("video_uid"),
                            "condition": condition, "task": task,
                            "max_side": args.max_side,
                            "observed_frame": item["end_frame"], "fps": item["fps"],
                            "ttc": item["ttc"], "horizon_bin": item["horizon_bin"],
                            "gold": gold, "choices": choices, "answer": "",
                            "prediction_index": None, "correct": False,
                            "status": "error", "error": f"{type(exc).__name__}: {exc}",
                        }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {"schema": "ttt_frame.fho_sta_smoke/1", "annotations": str(args.annotations),
               "video_root": str(args.video_root), "model_path": args.model_path,
               "spatial_checkpoint": args.spatial_checkpoint, "n_items": len(items),
               "max_side": args.max_side,
               "n_rows": len(rows), "summary": _summary(rows),
               "note": "Closed-set noun/verb smoke test; not official FHO STA mAP."}
    args.output.with_name("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
