#!/usr/bin/env python3
"""Evaluate a visual in-context baseline on a HomeSentinel prefix.

The runner intentionally keeps the evaluation protocol separate from the
Spatial-TTT writer.  For each query it samples a fixed number of frames from
every HomeSentinel/asuka video up to (and including) that query's
``video_cutoff_idx`` and sends the resulting chronological frame sequence to
Qwen3-VL as one native ``video`` input.  No later video, caption, answer, or
evidence field is read by the model.

This is the B quick baseline from the predictive-TTT plan.  It is an
explicit-history ICL control, not a parameter-memory method.  The lexical
scores in the summary are diagnostics for the existing open-ended benchmark;
they are not human-validated semantic accuracy.

Example::

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_homesentinel_icl.py \
      --category all --max-cutoff 10 --frames-per-video 2 \
      --max-history-frames 32 --device cuda:2 \
      --output runs/homesentinel_b_icl_event_home10.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data/HomeSentinel/asuka")
DEFAULT_ORDER = Path("/data/HomeSentinel/video_order.json")
DEFAULT_MODEL = "/data/quzitsix/models/Qwen3-VL-2B-Instruct"


def _tokens(text: str) -> list[str]:
    text = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"[a-z0-9]+", text)


def score_answer(prediction: str, gold: str) -> dict[str, Any]:
    """Return the same transparent lexical diagnostics as the Spatial runner."""

    p = _tokens(prediction)
    g = _tokens(gold)
    overlap = sum((Counter(p) & Counter(g)).values())
    precision = overlap / len(p) if p else 0.0
    recall = overlap / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    p_text, g_text = " ".join(p), " ".join(g)
    return {
        "exact_normalized": p_text == g_text,
        "gold_substring": bool(g_text) and g_text in p_text,
        "token_f1": round(f1, 4),
    }


def load_order(order_path: Path, data_root: Path) -> list[str]:
    order = json.loads(order_path.read_text(encoding="utf-8"))
    try:
        ids = list(order["asuka"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{order_path} does not contain an asuka video list") from exc
    missing = [video_id for video_id in ids if not (data_root / video_id / "indoor_video.mp4").is_file()]
    if missing:
        raise ValueError(f"missing asuka indoor videos ({len(missing)}): {missing[:5]}")
    return ids


def load_queries(data_root: Path, category: str, max_cutoff: int) -> list[dict[str, Any]]:
    names = ("event", "home") if category == "all" else (category,)
    rows: list[dict[str, Any]] = []
    for name in names:
        path = data_root / "benchmark_queries" / f"{name}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"query file is not a list: {path}")
        for query in raw:
            if not isinstance(query, dict) or not query.get("query_id") or not query.get("question"):
                raise ValueError(f"invalid query in {path}: {query!r}")
            if query.get("answer_type") != "open_ended":
                raise ValueError(f"unsupported answer type for {query['query_id']}")
            cutoff = int(query.get("video_cutoff_idx", -1))
            # B is a prefix experiment.  Full-history (-1) items are excluded
            # unless the caller explicitly raises max-cutoff to the dataset size.
            if cutoff < 0 or cutoff > max_cutoff:
                continue
            row = dict(query)
            row["category_file"] = name
            row["cutoff"] = cutoff
            rows.append(row)
    rows.sort(key=lambda row: (row["cutoff"], row["category_file"], row["query_id"]))
    return rows


def _video_duration(container, stream) -> float:
    import av

    if container.duration is not None:
        # PyAV exposes container.duration in AV_TIME_BASE units (microseconds);
        # ``av.time_base`` is the number of those units per second.
        duration = float(container.duration / av.time_base)
        if duration > 0:
            return duration
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if duration > 0:
            return duration
    return 0.0


def sample_video_frames(path: Path, count: int, max_side: int, max_seconds: float = 0.0) -> list[Any]:
    """Decode one video once and retain ``count`` chronological RGB frames.

    HomeSentinel videos are long, so storing only the selected PIL images keeps
    the ICL context bounded.  We seek to the selected timestamps instead of
    decoding a 1,000-second file from beginning to end for every query.  If a
    codec refuses a seek, the helper falls back to one sequential pass.
    """

    if count <= 0:
        raise ValueError("frames-per-video must be positive")
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream: {path}")
        stream = container.streams.video[0]
        duration = _video_duration(container, stream)
        # Use segment starts, matching the project's chronological sampler.
        span = min(duration, max_seconds) if max_seconds > 0 and duration > 0 else duration
        targets = [span * i / count for i in range(count)] if span > 0 else [float(i) for i in range(count)]
        origin = (float(stream.start_time * stream.time_base)
                  if stream.start_time is not None and stream.time_base is not None else 0.0)
        selected: list[Any] = []
        seek_failed = False
        for target in targets:
            target_abs = origin + target
            try:
                # av.container.seek uses the stream time base when ``stream``
                # is supplied.  Decode only from the preceding keyframe.
                seek_ts = int(target_abs / float(stream.time_base)) if stream.time_base else int(target_abs * av.time_base)
                container.seek(seek_ts, stream=stream, any_frame=False, backward=True)
                chosen = None
                fallback = None
                for frame in container.decode(stream):
                    absolute = frame.time
                    if absolute is None:
                        continue
                    fallback = frame
                    if float(absolute) + 0.05 >= target_abs:
                        chosen = frame
                        break
                if chosen is None:
                    chosen = fallback
                if chosen is None:
                    raise ValueError("seek returned no frames")
                picture = chosen.to_image().convert("RGB")
                picture.thumbnail((max_side, max_side))
                selected.append(picture)
            except Exception:
                seek_failed = True
                break
        if not seek_failed and len(selected) == count:
            return selected

        # Sequential fallback for unusual containers/codecs.  This path is
        # still bounded in memory because it only retains requested frames.
        container.seek(0, any_frame=False, backward=True)
        selected = []
        target_index = 0
        first_time: float | None = None
        last_image = None
        for frame in container.decode(stream):
            absolute = frame.time
            if absolute is None:
                if stream.average_rate:
                    absolute = float(frame.pts or 0) / float(stream.average_rate)
                else:
                    absolute = float(len(selected))
            if first_time is None:
                first_time = float(absolute)
            timestamp = max(0.0, float(absolute) - first_time)
            picture = None
            while target_index < count and timestamp + 1e-6 >= targets[target_index]:
                if picture is None:
                    picture = frame.to_image().convert("RGB")
                    picture.thumbnail((max_side, max_side))
                selected.append(picture.copy())
                target_index += 1
            if picture is None:
                picture = frame.to_image().convert("RGB")
                picture.thumbnail((max_side, max_side))
            last_image = picture
            if target_index >= count:
                break
        if not selected:
            raise ValueError(f"no decodable frames: {path}")
        while len(selected) < count:
            selected.append((last_image or selected[-1]).copy())
        return selected[:count]


def select_history(all_frames: dict[int, list[Any]], cutoff: int, max_history: int) -> list[Any]:
    frames = [image for index in range(1, cutoff + 1) for image in all_frames[index]]
    if not frames:
        raise ValueError(f"no frames available through cutoff {cutoff}")
    if len(frames) <= max_history:
        return frames
    # Uniform temporal subsampling retains the beginning and end of the
    # observed prefix while enforcing an identical history budget per query.
    indices = [round(i * (len(frames) - 1) / (max_history - 1))
               for i in range(max_history)] if max_history > 1 else [len(frames) // 2]
    return [frames[index] for index in indices]


def _pad_temporal(images: list[Any], minimum_frames: int = 3) -> list[Any]:
    if not images:
        raise ValueError("at least one image is required")
    padded = list(images)
    while len(padded) < minimum_frames:
        padded.append(padded[-1].copy())
    return padded


class NativeVideoICL:
    """Thin native Qwen3-VL video processor/generation wrapper."""

    def __init__(self, model_path: str, device: str, dtype: str, local_files_only: bool):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=local_files_only, dtype=self.dtype,
            device_map={"": str(self.device)}, attn_implementation="sdpa",
        )
        self.model.eval()

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device, dtype=self.dtype if value.is_floating_point() else value.dtype)
            else:
                moved[key] = value
        return moved

    @torch.inference_mode()
    def answer(self, question: str, images: list[Any], max_new_tokens: int) -> str:
        import numpy as np
        from transformers.video_utils import VideoMetadata

        images = _pad_temporal(images)
        frames = np.stack([np.asarray(image.convert("RGB")) for image in images])
        timestamps = list(range(len(images)))
        metadata = VideoMetadata(total_num_frames=len(images), fps=1.0,
                                 frames_indices=timestamps)
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": (
                    "The video frames are in chronological order, oldest first. "
                    "Answer only from the observed frames and do not assume unseen future events.\n\n"
                    + question.strip()
                )},
            ]}], tokenize=False, add_generation_prompt=True,
        )
        batch = self.processor(text=[prompt], videos=[frames], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt")
        batch = self._move(batch)
        if "pixel_values_videos" not in batch or "video_grid_thw" not in batch:
            raise RuntimeError("native video processor produced no video pixels/grid")
        output = self.model.generate(**batch, max_new_tokens=max_new_tokens,
                                     do_sample=False, use_cache=True)
        continuation = output[:, batch["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
        return answer


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [row for row in rows if row["status"] == "ok"]
    result: dict[str, Any] = {
        "n_rows": len(rows), "n_ok": len(good),
        "mean_token_f1": round(sum(row["score"]["token_f1"] for row in good) / len(good), 4) if good else 0.0,
        "exact_normalized": sum(bool(row["score"]["exact_normalized"]) for row in good),
        "gold_substring": sum(bool(row["score"]["gold_substring"]) for row in good),
    }
    for field, key in (("category", "by_category"), ("cutoff", "by_cutoff")):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in good:
            groups.setdefault(str(row[field]), []).append(row)
        result[key] = {
            group: {
                "n": len(items),
                "mean_token_f1": round(sum(item["score"]["token_f1"] for item in items) / len(items), 4),
                "exact_normalized": sum(bool(item["score"]["exact_normalized"]) for item in items),
                "gold_substring": sum(bool(item["score"]["gold_substring"]) for item in items),
            }
            for group, items in sorted(groups.items())
        }
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--order-path", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--category", choices=("event", "home", "all"), default="all")
    parser.add_argument("--max-cutoff", type=int, default=10,
                        help="only queries whose finite cutoff is <= this video index")
    parser.add_argument("--limit", type=int, default=0, help="limit selected queries (0 means all)")
    parser.add_argument("--frames-per-video", type=int, default=2)
    parser.add_argument("--max-history-frames", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--sample-seconds", type=float, default=60.0,
                        help="sample only the first N seconds of each long video (0 means full duration)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/homesentinel_b_icl_event_home10.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate selection/sampling without loading Qwen")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if (args.max_cutoff < 1 or args.limit < 0 or args.frames_per_video < 1
            or args.max_history_frames < 1 or args.sample_seconds < 0):
        raise SystemExit("max-cutoff, frames-per-video and max-history-frames must be positive; sample-seconds must be nonnegative")
    data_root = args.data_root.resolve()
    order = load_order(args.order_path.resolve(), data_root)
    queries = load_queries(data_root, args.category, args.max_cutoff)
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        raise SystemExit("no finite event/home queries matched the cutoff")
    needed = sorted({index for query in queries for index in range(1, query["cutoff"] + 1)})
    if needed and max(needed) > len(order):
        raise SystemExit(f"query cutoff exceeds video order ({max(needed)} > {len(order)})")
    print(json.dumps({"queries": len(queries), "categories": sorted({q["category_file"] for q in queries}),
                      "cutoffs": sorted({q["cutoff"] for q in queries}),
                      "videos_needed": len(needed), "frames_per_video": args.frames_per_video,
                      "max_history_frames": args.max_history_frames,
                      "sample_seconds": args.sample_seconds}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}; use --overwrite or choose another path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_frames: dict[int, list[Any]] = {}
    started_all = time.perf_counter()
    for index in needed:
        video_id = order[index - 1]
        path = data_root / video_id / "indoor_video.mp4"
        started = time.perf_counter()
        all_frames[index] = sample_video_frames(path, args.frames_per_video, args.max_side,
                                                args.sample_seconds)
        print(json.dumps({"sampled_video_index": index, "video_id": video_id,
                          "frames": len(all_frames[index]),
                          "elapsed_sec": round(time.perf_counter() - started, 3)}, ensure_ascii=False), flush=True)
    model = NativeVideoICL(args.model_path, args.device, args.dtype, args.local_files_only)
    rows: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as out:
        for query in queries:
            started = time.perf_counter()
            history = select_history(all_frames, query["cutoff"], args.max_history_frames)
            try:
                answer = model.answer(query["question"], history, args.max_new_tokens)
                row = {
                    "query_id": query["query_id"], "category": query["category_file"],
                    "cutoff": query["cutoff"], "question": query["question"],
                    "ground_truth": query.get("ground_truth", ""), "arm": "icl_visual",
                    "status": "ok", "answer": answer,
                    "history_frames": len(history),
                    "score": score_answer(answer, query.get("ground_truth", "")),
                    "elapsed_sec": round(time.perf_counter() - started, 4),
                }
            except Exception as exc:  # preserve independent query diagnostics
                row = {
                    "query_id": query["query_id"], "category": query["category_file"],
                    "cutoff": query["cutoff"], "question": query["question"],
                    "ground_truth": query.get("ground_truth", ""), "arm": "icl_visual",
                    "status": "error", "answer": "", "history_frames": len(history),
                    "score": {"exact_normalized": False, "gold_substring": False, "token_f1": 0.0},
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": round(time.perf_counter() - started, 4),
                }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            print(f"{row['query_id']} {row['status']}: {row['answer'][:160]}", flush=True)
    payload = {
        "schema": "ttt_frame.homesentinel_icl_eval/1",
        "data_root": str(data_root), "order_path": str(args.order_path.resolve()),
        "model_path": args.model_path, "device": args.device, "dtype": args.dtype,
        "category": args.category, "max_cutoff": args.max_cutoff,
        "frames_per_video": args.frames_per_video, "max_history_frames": args.max_history_frames,
        "sample_seconds": args.sample_seconds,
        "max_side": args.max_side, "max_new_tokens": args.max_new_tokens,
        "n_queries": len(queries), "n_videos_sampled": len(needed),
        "elapsed_sec": round(time.perf_counter() - started_all, 3),
        "summary": summarize(rows), "predictions_jsonl": str(args.output),
        "note": "Visual ICL receives only sampled frames from videos <= each query cutoff; lexical scores are diagnostics.",
    }
    # Keep summaries next to their prediction file without clobbering an
    # unrelated run's generic ``summary.json``.
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
