#!/usr/bin/env python3
"""Evaluate Spatial-TTT on the complete local HomeSentinel/asuka history.

The 52 ``asuka`` videos are ingested in ``video_order.json`` order.  Queries
with a finite ``video_cutoff_idx`` are answered immediately after that video;
queries with cutoff ``-1`` are answered after the complete history.  This
prevents later videos from leaking into episodic questions.  Every query is
answered twice: with Spatial fast weights and with the paired ``--without-
memory`` read control at the same cutoff.

Example::

    conda run --no-capture-output -n meowbench python \
      scripts/evaluate_homesentinel_spatial.py \
      --device cuda:4 --chunk-seconds 60 --frames-per-chunk 4 \
      --output runs/homesentinel_asuka_spatial_60s4f/predictions.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import string
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data/HomeSentinel/asuka")
DEFAULT_ORDER = Path("/data/HomeSentinel/video_order.json")
DEFAULT_MODEL = "/data/quzitsix/models/Qwen3-VL-2B-Instruct"
DEFAULT_CHECKPOINT = "/data/quzitsix/models/Spatial-TTT-nano/model.safetensors"


def _tokens(text: str) -> list[str]:
    text = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"[a-z0-9]+", text)


def score_answer(prediction: str, gold: str) -> dict[str, Any]:
    """Return transparent lexical diagnostics for an open-ended answer."""

    p = _tokens(prediction)
    g = _tokens(gold)
    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
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


def load_queries(data_root: Path, category: str | None = None) -> list[dict[str, Any]]:
    names = (category,) if category else ("owner", "home", "event")
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
            row = dict(query)
            row["category_file"] = name
            row["cutoff"] = int(query.get("video_cutoff_idx", -1))
            rows.append(row)
    return rows


def validate_cutoffs(queries: list[dict[str, Any]], n_videos: int) -> None:
    bad = [q["query_id"] for q in queries if q["cutoff"] < -1 or q["cutoff"] > n_videos]
    if bad:
        raise ValueError(f"query cutoff outside 0..{n_videos} or -1: {bad[:8]}")


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n_rows": len(rows)}
    for arm in ("memory", "without_memory"):
        selected = [row for row in rows if row["arm"] == arm and row["status"] == "ok"]
        by_category: dict[str, dict[str, Any]] = {}
        by_cutoff_type: dict[str, dict[str, Any]] = {}
        for row in selected:
            category = row["category"]
            bucket = by_category.setdefault(category, {"n": 0, "token_f1_sum": 0.0,
                                                        "exact_normalized": 0,
                                                        "gold_substring": 0})
            cutoff_type = "full_history" if row["cutoff"] == -1 else "episodic_cutoff"
            other = by_cutoff_type.setdefault(cutoff_type, {"n": 0, "token_f1_sum": 0.0,
                                                            "exact_normalized": 0,
                                                            "gold_substring": 0})
            for bucket2 in (bucket, other):
                bucket2["n"] += 1
                bucket2["token_f1_sum"] += float(row["score"]["token_f1"])
                bucket2["exact_normalized"] += int(row["score"]["exact_normalized"])
                bucket2["gold_substring"] += int(row["score"]["gold_substring"])
        for groups in (by_category, by_cutoff_type):
            for bucket in groups.values():
                n = bucket.pop("token_f1_sum")
                bucket["mean_token_f1"] = round(n / bucket["n"], 4) if bucket["n"] else 0.0
        result[arm] = {"n": len(selected), "by_category": by_category,
                       "by_cutoff_type": by_cutoff_type,
                       "mean_token_f1": round(
                           sum(float(row["score"]["token_f1"]) for row in selected) / len(selected), 4
                       ) if selected else 0.0,
                       "exact_normalized": sum(bool(row["score"]["exact_normalized"]) for row in selected),
                       "gold_substring": sum(bool(row["score"]["gold_substring"]) for row in selected)}
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--order-path", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--category", choices=("owner", "home", "event"))
    parser.add_argument("--limit", type=int, default=0,
                        help="limit queries after loading (0 means all; use category for a focused pilot)")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="ingest only a prefix for a smoke test; 0 means all 52")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--spatial-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/homesentinel_asuka_spatial_60s4f/predictions.jsonl"))
    return parser.parse_args(argv)


def _write_row(out, query: dict[str, Any], arm: str, answer: str, started: float,
               video_index: int, status: str = "ok", error: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "query_id": query["query_id"],
        "category": query["category_file"],
        "difficulty": query.get("difficulty"),
        "question": query["question"],
        "ground_truth": query.get("ground_truth", ""),
        "cutoff": query["cutoff"],
        "video_index_answered": video_index,
        "arm": arm,
        "status": status,
        "answer": answer,
        "elapsed_sec": round(time.perf_counter() - started, 4),
    }
    if status == "ok":
        row["score"] = score_answer(answer, row["ground_truth"])
    else:
        row["score"] = {"exact_normalized": False, "gold_substring": False, "token_f1": 0.0}
        row["error"] = error or "unknown error"
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    return row


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.limit < 0 or args.max_videos < 0 or args.chunk_seconds <= 0 or args.frames_per_chunk <= 0:
        raise SystemExit("limit/max-videos must be nonnegative; chunk-seconds and frames-per-chunk must be positive")
    data_root = args.data_root.resolve()
    order = load_order(args.order_path.resolve(), data_root)
    queries = load_queries(data_root, args.category)
    validate_cutoffs(queries, len(order))
    if args.limit:
        queries = queries[:args.limit]
    video_count = min(args.max_videos or len(order), len(order))
    if video_count < len(order):
        # A full-history query has no valid answer until all 52 videos have
        # been ingested; silently treating a prefix as its full history would
        # make a smoke run look like a valid benchmark result.
        queries = [q for q in queries if q["cutoff"] != -1 and q["cutoff"] <= video_count]
    by_cutoff: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        by_cutoff[video_count if query["cutoff"] == -1 else query["cutoff"]].append(query)
    expected = {(q["query_id"], arm) for q in queries for arm in ("memory", "without_memory")}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Delayed import keeps --help and data validation usable without CUDA.
    from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory

    config = SpatialVideoConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        spatial_checkpoint=args.spatial_checkpoint,
        chunk_seconds=args.chunk_seconds,
        frames_per_chunk=args.frames_per_chunk,
        max_side=args.max_side,
        max_new_tokens=args.max_new_tokens,
    )
    model = SpatialVideoMemory(config)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as out:
        for video_index, video_id in enumerate(order[:video_count], start=1):
            video_path = data_root / video_id / "indoor_video.mp4"
            started = time.perf_counter()
            ingest_report = model.ingest_video(video_path)
            model.finish_ingest()
            print(json.dumps({"video_index": video_index, "video_id": video_id,
                              "ingest": ingest_report,
                              "elapsed_sec": round(time.perf_counter() - started, 2)},
                             ensure_ascii=False), flush=True)
            for query in by_cutoff.get(video_index, []):
                for arm, use_memory in (("memory", True), ("without_memory", False)):
                    q_started = time.perf_counter()
                    try:
                        answer = model.answer(query["question"], use_memory=use_memory,
                                              max_new_tokens=args.max_new_tokens)
                        row = _write_row(out, query, arm, answer, q_started, video_index)
                    except Exception as exc:  # preserve other query results
                        row = _write_row(out, query, arm, "", q_started, video_index,
                                         status="error", error=f"{type(exc).__name__}: {exc}")
                    rows.append(row)
                    print(f"  {query['query_id']} {arm}: {row['answer'][:160]}", flush=True)
        # The full-history queries are mapped to video_count above.  This branch
        # is only reached for a deliberately short smoke run with no full query.
    payload = {
        "schema": "ttt_frame.homesentinel_spatial_eval/1",
        "data_root": str(data_root),
        "order_path": str(args.order_path.resolve()),
        "model_path": args.model_path,
        "spatial_checkpoint": args.spatial_checkpoint,
        "device": args.device,
        "dtype": args.dtype,
        "chunk_seconds": args.chunk_seconds,
        "frames_per_chunk": args.frames_per_chunk,
        "max_side": args.max_side,
        "n_videos_ingested": video_count,
        "n_queries_selected": len(queries),
        "n_prediction_rows": len(rows),
        "expected_rows": len(expected),
        "elapsed_sec": round(time.perf_counter() - started_all, 3),
        "summary": summary(rows),
        "predictions_jsonl": str(args.output),
        "note": "Open-ended lexical scores are diagnostics; gold/evidence never enter model prompts.",
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(rows) == len(expected) and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
