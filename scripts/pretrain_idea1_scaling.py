#!/usr/bin/env python3
"""Learn Idea 01 fast-weight initializations from declared training prefixes.

This trains ordinary future-feature regression on within-train observed pairs;
it does not differentiate through online updates and does not open MCQ answers
or benchmark future clips. Checkpoints at different steps share one trajectory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.run_forecast_residual import read_split
from scripts.run_idea1_mcq_forecasts import (
    atomic_json, atomic_save, canonical_sha256, read_feature_protocol, sha256_file, validated_cache,
)
from ttt_frame.forecast_residual import RidgeRecentPredictor
from ttt_frame.idea1_scaling_pretrain import (
    FrozenPairDataset, OfflineForecastMLPs, OfflinePretrainConfig, train_step, validation_losses,
)


def load_inputs(cache_dir: Path, split_path: Path, predictor_path: Path,
                *, device: str) -> tuple[dict, RidgeRecentPredictor, FrozenPairDataset, FrozenPairDataset, dict]:
    split = read_split(split_path)
    if sha256_file(Path(split["root"]) / "inputs_mcq.jsonl") != split["inputs_manifest_sha256"]:
        raise ValueError("public manifest changed")
    predictor_record_path = predictor_path.parent / "protocol.json"
    record = json.loads(predictor_record_path.read_text())
    if (record["protocol_sha256"] != canonical_sha256(record["protocol"])
            or record["protocol"]["split_sha256"] != sha256_file(split_path)
            or record["protocol"]["splits"] != split["splits"]
            or record["predictor_sha256"] != sha256_file(predictor_path)):
        raise ValueError("predictor provenance differs from the declared research split")
    predictor = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if (predictor.get("kind") != "idea01_frozen_recent_ridge"
            or predictor.get("train_ids") != split["splits"]["train"]
            or predictor.get("protocol_sha256") != record["protocol_sha256"]):
        raise ValueError("recent predictor must have been fitted on declared training IDs only")
    for name, expected in record["protocol"]["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"frozen predictor implementation changed: {name}")
    baseline = RidgeRecentPredictor.from_dict(predictor["baseline"])
    feature_record = read_feature_protocol(cache_dir, split)
    if record["protocol"]["feature_protocol"] != feature_record:
        raise ValueError("feature cache does not match frozen recent predictor")
    sources = {}
    datasets = {}
    # Test caches are deliberately not opened by this entry point.
    for role in ("train", "validation"):
        streams = []
        for sample_id in split["splits"][role]:
            cache, identity = validated_cache(cache_dir / f"{sample_id}.pt", sample_id, split, feature_record)
            if identity != record["protocol"]["cache_sources"][sample_id]:
                raise ValueError(f"cache changed since recent predictor fit: {sample_id}")
            sources[sample_id] = identity
            streams.append(cache["features"])
        datasets[role] = FrozenPairDataset(streams, baseline, device=device)
        del streams
    provenance = {
        "split_sha256": sha256_file(split_path), "input_manifest_sha256": split["inputs_manifest_sha256"],
        "feature_protocol_sha256": feature_record["protocol_sha256"],
        "recent_predictor_sha256": record["predictor_sha256"],
        "recent_predictor_protocol_sha256": record["protocol_sha256"],
        "cache_sources": sources,
    }
    return split, baseline, datasets["train"], datasets["validation"], provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("runs/forecast_bench_v0/idea01_mcq_v1/features"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("runs/forecast_bench_v0/research_day_split.json"))
    parser.add_argument("--recent-predictor", type=Path,
                        default=Path("runs/forecast_bench_v0/idea01_mcq_v1/forecasts/recent_predictor.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--steps", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--validation-origins", type=int, default=512)
    args = parser.parse_args(argv)
    if (args.output_dir.exists() or args.threads < 1 or args.validation_origins < 1
            or not args.steps or any(type(step) is not int or step < 1 for step in args.steps)
            or len(set(args.steps)) != len(args.steps)):
        parser.error("new output directory, positive threads/origins and distinct positive snapshot steps required")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    config = OfflinePretrainConfig(
        hidden_dim=args.hidden_dim, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, seed=args.seed,
    )
    started = time.perf_counter()
    split, baseline, train, validation, provenance = load_inputs(
        args.cache_dir, args.split_manifest, args.recent_predictor, device=args.device)
    if args.output_dir.resolve().is_relative_to(Path(split["root"]).resolve()):
        parser.error("output must be outside the benchmark source")
    val_batch = validation.sample(args.validation_origins, torch.Generator().manual_seed(187))
    # Only this sampled validation batch is kept on the accelerator.
    del validation
    model = OfflineForecastMLPs(len(baseline.horizons), baseline.dim, config, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    protocol = {
        "schema": "idea01_offline_initialization_protocol_v1", "provenance": provenance,
        "config": asdict(config), "snapshot_steps": sorted(args.steps),
        "train_ids": split["splits"]["train"], "validation_ids": split["splits"]["validation"],
        "horizons": list(baseline.horizons), "window_steps": baseline.window_steps,
        "device": args.device, "dtype": "float32", "matmul_precision": "highest",
        "optimizer": "AdamW", "loss": "sum_arm_horizon_mean_batch_token_dim_MSE",
        "width_output_gain": "sqrt(128/hidden_dim)",
        "pair_sampling": "uniform_case_then_uniform_valid_origin_with_replacement; four horizons share cue",
        "validation": {"origins": args.validation_origins, "seed": 187,
                       "used_for": "fixed snapshot diagnostic only, no early stopping or selection"},
        "test_ids_or_labels_used": False, "mcq_answers_used": False,
        "future_media_used": False, "meta_outer_training": False,
        "training_type": "offline within-train past-prefix prediction initialization",
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/pretrain_idea1_scaling.py", "ttt_frame/idea1_scaling_pretrain.py")},
    }
    protocol_sha = canonical_sha256(protocol)
    args.output_dir.mkdir(parents=True)
    atomic_json({"protocol_sha256": protocol_sha, "protocol": protocol}, args.output_dir / "protocol.json")
    generator = torch.Generator().manual_seed(config.seed)
    seen = torch.zeros(len(train.streams), max(train.origins_per_case), dtype=torch.bool)
    snapshot_metrics, loss_records = {}, []
    steps = {0, *args.steps}
    train_started = time.perf_counter()
    for step in range(max(args.steps) + 1):
        if step:
            batch = train.sample(config.batch_size, generator)
            seen[batch["case_indices"], batch["origins"]] = True
            record = train_step(model, optimizer, batch, config)
            loss_records.append({"step": step, **record})
            if step % 25 == 0 or step in steps:
                print(json.dumps({"step": step, "hidden_dim": config.hidden_dim,
                                  "train": record, "training_seconds": time.perf_counter() - train_started}), flush=True)
        if step not in steps:
            continue
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        checkpoint = {
            "schema": "idea01_offline_initialization_checkpoint_v1", "protocol_sha256": protocol_sha,
            "step": step, "steps": step, "train_ids": split["splits"]["train"],
            "predictor_sha256": provenance["recent_predictor_sha256"],
            "hidden_dim": config.hidden_dim, "width_output_gain": math_gain(config.hidden_dim),
            "initial_weights": model.initial_weights(), "horizons": list(baseline.horizons),
            "training_origins_drawn": step * config.batch_size,
            "unique_training_case_origins": int(seen.sum()),
            "possible_training_case_origins": sum(train.origins_per_case),
            "training_horizon_token_pairs_per_target": step * config.batch_size * len(baseline.horizons) * baseline.tokens,
            "optimizer_steps_per_target": step,
            "validation": validation_losses(model, val_batch),
            "training_seconds": time.perf_counter() - train_started,
            "test_ids_or_labels_used": False,
        }
        path = args.output_dir / f"step_{step:06d}.pt"
        atomic_save(checkpoint, path)
        snapshot_metrics[str(step)] = {key: value for key, value in checkpoint.items() if key != "initial_weights"}
        snapshot_metrics[str(step)]["checkpoint_sha256"] = sha256_file(path)
        print(json.dumps({"snapshot": step, "validation": checkpoint["validation"],
                          "unique_training_case_origins": checkpoint["unique_training_case_origins"]}), flush=True)
    atomic_json({
        "schema": "idea01_offline_initialization_summary_v1", "complete": True,
        "protocol_sha256": protocol_sha, "elapsed_seconds": time.perf_counter() - started,
        "training_seconds": time.perf_counter() - train_started, "snapshots": snapshot_metrics,
        "independent_train_cases": len(train.streams), "supervised_mcq_training_examples_added": 0,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device))
            if args.device.startswith("cuda") else 0,
    }, args.output_dir / "summary.json")
    atomic_json(loss_records, args.output_dir / "training_curve.json")
    print(json.dumps({"complete": True, "output": str(args.output_dir),
                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return 0


def math_gain(hidden_dim: int) -> float:
    return (128 / hidden_dim) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())
