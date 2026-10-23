#!/usr/bin/env python3
"""Fit a frozen recent predictor and test Idea 01 on causal past-only caches.

This is a latent preflight, not a six-choice action evaluator. No future.mp4,
answer key or private benchmark metadata is opened. By default every cache in
the explicit research day split is required; --train-ids / --eval-ids make a
small pilot an explicit subset rather than silently ignoring missing data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from ttt_frame.forecast_residual import (
    ARMS,
    PredictiveRunConfig,
    RidgeRecentPredictor,
    run_prequential,
)


def read_split(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema") != "forecast_bench_research_day_split_v1"
        or data.get("official_split") is not False
        or data.get("grouping") != "recording_day_all_participants"
    ):
        raise ValueError("expected an explicit, unofficial recording-day research split")
    seen_ids, seen_days = set(), set()
    for role in ("train", "validation", "test"):
        ids, days = data["splits"][role], data["days"][role]
        if not ids or not days or len(set(ids)) != len(ids):
            raise ValueError(f"empty or duplicated {role} split")
        if seen_ids.intersection(ids) or seen_days.intersection(days):
            raise ValueError("research split overlaps IDs or recording days")
        for sample_id in ids:
            if data["id_to_day"][sample_id] not in days:
                raise ValueError(f"day assignment inconsistent for {sample_id}")
        seen_ids.update(ids)
        seen_days.update(days)
    return data


def select_ids(split: dict, role: str, requested: list[str] | None) -> list[str]:
    selected = requested or list(split["splits"][role])
    if len(set(selected)) != len(selected) or not set(selected) <= set(split["splits"][role]):
        raise ValueError(f"requested IDs must be distinct members of {role}")
    return selected


def load_cache(path: Path, sample_id: str, split: dict, allow_partial: bool) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if (
        cache.get("kind") != "bench_v0_causal_visual_features"
        or cache.get("format_version") != 1
        or cache.get("sample_id") != sample_id
        or cache.get("input_manifest_sha256") != split["inputs_manifest_sha256"]
    ):
        raise ValueError(f"cache identity or source manifest mismatch: {path}")
    if not allow_partial and (cache.get("cropped") or not cache.get("complete_prefix")):
        raise ValueError("partial cache requires --allow-partial-prefix; it is a smoke test")
    features = cache["features"]
    requested, actual = cache["requested_prefix_seconds"], cache["actual_prefix_seconds"]
    fps = float(cache["sample_fps"])
    if (
        features.ndim != 3 or not features.is_floating_point()
        or tuple(features.shape[1:]) != (8, 256)
        or features.shape[0] < 2 or not torch.isfinite(features).all()
        or not 0 < fps <= 2
        or requested.shape != (features.shape[0],) or actual.shape != requested.shape
        or not requested.is_floating_point() or not actual.is_floating_point()
        or not torch.isfinite(requested).all() or not torch.isfinite(actual).all()
        or bool((actual < 0).any()) or bool((actual > requested + 1e-6).any())
        or bool((actual[1:] < actual[:-1]).any())
        or bool((requested - actual >= 1 / fps + 1e-6).any())
        or bool((requested >= 1830).any()) or bool((requested < 0).any())
        or not torch.allclose(
            requested[1:] - requested[:-1], torch.full_like(requested[1:], 1 / fps),
            atol=1e-6, rtol=0,
        )
    ):
        raise ValueError(f"invalid features or noncausal/irregular time grid: {path}")
    if cache.get("complete_prefix") and (
        abs(float(requested[0])) > 1e-6
        or abs(float(requested[-1]) - (1830 - 1 / fps)) > 1e-6
    ):
        raise ValueError("complete prefix must cover the entire 1830-second past")
    if cache.get("encoder") != "qwen3_vl_per_image" or cache.get("audio_used") is not False:
        raise ValueError("expected independent RGB image features without audio")
    return cache


def cache_signature(cache: dict) -> dict:
    config = cache["config"]
    # Cropping settings are episode-level; representation settings must match.
    return {
        "encoder": cache["encoder"],
        "model_fingerprint": cache["model_fingerprint"],
        "sample_fps": cache["sample_fps"],
        "shape": list(cache["features"].shape[1:]),
        "representation": {k: v for k, v in config.items() if k not in (
            "context_tail_seconds", "limit_frames", "batch_size", "device",
        )},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ids", nargs="+")
    parser.add_argument("--eval-ids", nargs="+")
    parser.add_argument("--eval-role", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-partial-prefix", action="store_true")
    parser.add_argument("--horizons-steps", type=int, nargs="+", default=[1, 2, 4, 10])
    parser.add_argument("--window-steps", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bank-capacity", type=int, default=16,
                        help="cue/residual pairs per horizon; 16 matches 1 MiB fast state")
    parser.add_argument("--score-last-seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args(argv)
    if args.threads < 1 or not 0 < args.score_last_seconds <= 1830:
        parser.error("positive --threads and --score-last-seconds in (0,1830] required")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("--ridge must be finite and positive")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not contain duplicates")
    if args.output_dir.exists():
        parser.error("output directory exists; choose a new run name")
    torch.set_num_threads(args.threads)
    split = read_split(args.split_manifest)
    if args.output_dir.resolve().is_relative_to(Path(split["root"]).resolve()):
        parser.error("output directory must be outside the source benchmark")
    train_ids = select_ids(split, "train", args.train_ids)
    eval_ids = select_ids(split, args.eval_role, args.eval_ids)
    train = [load_cache(args.cache_dir / f"{i}.pt", i, split, args.allow_partial_prefix)
             for i in train_ids]
    signature = cache_signature(train[0])
    if any(cache_signature(cache) != signature for cache in train[1:]):
        raise ValueError("train feature representations differ")
    # Validate every evaluation artifact before fitting or creating outputs.
    for sample_id in eval_ids:
        cache = load_cache(
            args.cache_dir / f"{sample_id}.pt", sample_id, split, args.allow_partial_prefix
        )
        if cache_signature(cache) != signature:
            raise ValueError(f"train/evaluation feature representations differ: {sample_id}")
    started = time.perf_counter()
    baseline = RidgeRecentPredictor.fit(
        [cache["features"].float() for cache in train],
        horizons=tuple(args.horizons_steps), window_steps=args.window_steps,
        ridge=args.ridge, standardize=True,
    )
    fit_seconds = time.perf_counter() - started
    del train
    args.output_dir.mkdir(parents=True)
    split_sha = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    torch.save({
        "kind": "idea01_frozen_recent_ridge", "baseline": baseline.to_dict(),
        "train_ids": train_ids, "split_sha256": split_sha, "feature_signature": signature,
    }, args.output_dir / "recent_predictor.pt")
    results, totals = [], {arm: {"sum_squared_error": 0.0, "count": 0} for arm in args.arms}
    for sample_id in eval_ids:
        cache = load_cache(
            args.cache_dir / f"{sample_id}.pt", sample_id, split, args.allow_partial_prefix
        )
        times = cache["requested_prefix_seconds"]
        score_start = int(torch.searchsorted(times, times[-1] + 1 / cache["sample_fps"]
                                            - args.score_last_seconds))
        config = PredictiveRunConfig(
            hidden_dim=args.hidden_dim, learning_rate=args.learning_rate,
            bank_capacity=args.bank_capacity, seed=args.seed, score_start_step=score_start,
        )
        case_start = time.perf_counter()
        result = run_prequential(cache["features"].float(), baseline, config, arms=args.arms)
        forecasts = {}
        for arm, record in result["arms"].items():
            forecasts[arm] = record.pop("last_forecasts")
            record.pop("predictions", None)
            if record["mse"] is not None:
                totals[arm]["sum_squared_error"] += record["mse"] * record["count"]
                totals[arm]["count"] += record["count"]
        result.update({
            "sample_id": sample_id, "elapsed_seconds": time.perf_counter() - case_start,
            "cropped": cache["cropped"], "complete_prefix": cache["complete_prefix"],
            "source_media": cache.get("source_media", {}),
            "sample_fps": cache["sample_fps"],
            "observed_requested_range_seconds": [float(times[0]), float(times[-1])],
            "last_observed_seconds_before_cutoff": float(cache["actual_prefix_seconds"][-1])
                - 1830,
            "last_prediction_target_seconds_after_cutoff": [
                float(times[-1]) + h / cache["sample_fps"] - 1830
                for h in args.horizons_steps
            ],
        })
        torch.save(forecasts, args.output_dir / f"{sample_id}_forecasts.pt")
        results.append(result)
        print(json.dumps({"sample_id": sample_id,
                          "mse": {a: r["mse"] for a, r in result["arms"].items()}},
                         ensure_ascii=False), flush=True)
    summary = {
        "schema": "idea01_bench_v0_latent_preflight_v1",
        "task": "past_prefix_prequential_latent_prediction",
        "is_mcq_accuracy": False, "future_media_used": False,
        "explicit_bank_is_vlm_icl": False, "outer_training_performed": False,
        "metric_space": "training-set-standardized visual features",
        "metric_count_unit": "matured origin-horizon pair; mean over pooled tokens and features",
        "score_range_definition": "prediction origins in last requested seconds of observed prefix",
        "train_ids": train_ids, "eval_ids": eval_ids, "eval_role": args.eval_role,
        "split_sha256": split_sha, "fit_seconds": fit_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_signature": signature, "arguments": {k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()},
        "aggregate": {arm: {"mse": value["sum_squared_error"] / value["count"]
                            if value["count"] else None, "count": value["count"]}
                      for arm, value in totals.items()},
        "cases": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved latent preflight to {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
