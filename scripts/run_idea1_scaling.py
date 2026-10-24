#!/usr/bin/env python3
"""Predeclare and execute Idea 01 capacity/update/pretraining scaling trials.

Uses immutable dense visual caches and the original train-only recent model.
No MCQ answers or future media are read. All test comparisons are exploratory,
because the original 57-question test was already inspected in the pilot.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.run_forecast_residual import read_split
from scripts.run_idea1_mcq_forecasts import canonical_sha256, sha256_file
from ttt_frame.forecast_residual import RidgeRecentPredictor


WIDTHS = (128, 512, 2048)
PRETRAIN_STEPS = (200, 800)
ARMS = ("persistence", "recent_only", "plain_delayed", "residual", "reset_residual",
        "recent_tail_residual", "plain_no_update", "residual_no_update")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy()).hexdigest()


def make_plan(base_run: Path, output: Path, split_path: Path) -> dict:
    base_run, output, split_path = (path.resolve() for path in (base_run, output, split_path))
    split = read_split(split_path)
    prior = json.loads((base_run / "forecasts/protocol.json").read_text())
    if (prior["protocol"]["split_sha256"] != sha256_file(split_path)
            or prior["protocol"]["splits"] != split["splits"]):
        raise ValueError("source predictor and declared split differ")
    trials = []
    for hidden in WIDTHS:
        for updates in (1, 4):
            name = f"h{hidden}_random_u{updates}"
            trials.append({"name": name, "forecast_dir": str(output / "forecasts" / name),
                           "arms": list(ARMS), "configuration": {
                               "hidden_dim": hidden, "inner_steps": updates,
                               "pretraining_steps": 0, "initialization_checkpoint": None,
                               "learning_rate": 0.05, "gain_mode": "normalized", "seed": 0,
                           }})
        for steps in PRETRAIN_STEPS:
            name = f"h{hidden}_pre{steps}_u1"
            trials.append({"name": name, "forecast_dir": str(output / "forecasts" / name),
                           "arms": list(ARMS), "configuration": {
                               "hidden_dim": hidden, "inner_steps": 1,
                               "pretraining_steps": steps,
                               "initialization_checkpoint": str(output / "pretraining" / f"h{hidden}" /
                                                                 f"step_{steps:06d}.pt"),
                               "learning_rate": 0.05, "gain_mode": "normalized", "seed": 0,
                           }})
    return {
        "schema": "idea01_scaling_experiment_plan_v1", "base_run": str(base_run),
        "split_manifest": str(split_path), "text_cache": str(base_run / "text_features.pt"),
        "baseline_readout": str(base_run / "mcq/shared_readout.pt"),
        "feature_cache": str(base_run / "features"),
        "predictor": str(base_run / "forecasts/recent_predictor.pt"),
        "source_forecast_protocol": str(base_run / "forecasts/protocol.json"),
        "input_manifest_sha256": split["inputs_manifest_sha256"],
        "split_sha256": sha256_file(split_path), "predictor_sha256": prior["predictor_sha256"],
        "widths": list(WIDTHS), "pretraining_steps": list(PRETRAIN_STEPS),
        "head_training_cases": [32, 64, len(split["splits"]["train"])],
        "primary_readout": "one train111-fitted semantic ridge across balanced trial/arm views",
        "secondary_readout": "nested train32/train64 and frozen pilot head; no test selection",
        "pretraining": {"batch_origins": 128, "horizons_steps": [1, 2, 4, 10],
                        "loss": "offline past-pair visual prediction; plain and residual initializers",
                        "optimizer": "AdamW", "learning_rate": 0.001,
                        "weight_decay": 0.01, "seed": 0,
                        "is_meta_outer_training": False, "test_data_used": False},
        "controls": {"same_visual_caches": True, "same_recent_predictor": True,
                     "per_case_reset": True, "no_cross_case_writes": True,
                     "width_gain": "sqrt(128/hidden) on hidden activation, also in gradients",
                     "width_gain_reason": "keep initial output-gradient scale comparable across widths"},
        "test_interpretation": "exploratory reuse of pilot test57, not new confirmatory evidence",
        "unique_labeled_training_cases_increase": False,
        "trials": trials,
    }


def load_trial_inputs(plan: dict, trial: dict, device: str):
    from ttt_frame.idea1_scaling_memory import ScalingMemoryConfig
    source = json.loads(Path(plan["source_forecast_protocol"]).read_text())
    split = read_split(Path(plan["split_manifest"]))
    predictor_path = Path(plan["predictor"])
    if (sha256_file(predictor_path) != plan["predictor_sha256"]
            or sha256_file(Path(plan["split_manifest"])) != plan["split_sha256"]
            or source["protocol"]["splits"] != split["splits"]
            or source["protocol"]["input_manifest_sha256"] != plan["input_manifest_sha256"]):
        raise ValueError("plan/source predictor/split mismatch")
    baseline = RidgeRecentPredictor.from_dict(torch.load(
        predictor_path, map_location="cpu", weights_only=True)["baseline"])
    options = trial["configuration"]
    config = ScalingMemoryConfig(hidden_dim=options["hidden_dim"], inner_steps=options["inner_steps"],
                                 learning_rate=options["learning_rate"], seed=options["seed"],
                                 gain_mode=options["gain_mode"])
    initial = None
    checkpoint_sha = None
    if options["initialization_checkpoint"] is not None:
        path = Path(options["initialization_checkpoint"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_sha = sha256_file(path)
        # Pretraining emits an explicit train-only provenance record.
        if (checkpoint["train_ids"] != split["splits"]["train"]
                or checkpoint["predictor_sha256"] != plan["predictor_sha256"]
                or checkpoint["steps"] != options["pretraining_steps"]):
            raise ValueError("initializer checkpoint does not match training plan")
        initial = checkpoint["initial_weights"]
    protocol = {
        "schema": "idea01_scaling_forecasts_protocol_v1", "trial_name": trial["name"],
        "splits": split["splits"], "split_sha256": plan["split_sha256"],
        "input_manifest_sha256": plan["input_manifest_sha256"], "arms": trial["arms"],
        "configuration": options, "online_config": asdict(config),
        "predictor_sha256": plan["predictor_sha256"],
        "initialization_checkpoint_sha256": checkpoint_sha,
        "source_forecast_protocol_sha256": source["protocol_sha256"],
        "plan_sha256": canonical_sha256(plan), "feature_cache": plan["feature_cache"],
        "answer_keys_used": False, "future_media_used": False,
        "test_interpretation": plan["test_interpretation"],
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/run_idea1_scaling.py", "ttt_frame/idea1_scaling_memory.py")},
        "torch_version": str(torch.__version__), "device_type": torch.device(device).type,
    }
    return split, source, baseline, config, initial, protocol


def validate_saved(path: Path, sample_id: str, digest: str) -> dict:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (record.get("schema") != "idea01_scaling_case_forecasts_v1"
            or record.get("sample_id") != sample_id or record.get("protocol_sha256") != digest
            or set(record.get("last_forecasts", {})) != set(ARMS)):
        raise ValueError(f"conflicting saved result: {sample_id}")
    for arm, value in record["last_forecasts"].items():
        if (value.shape != (4, 8, 256) or not bool(torch.isfinite(value).all())
                or tensor_sha(value) != record["forecast_sha256"][arm]):
            raise ValueError(f"corrupt saved result: {sample_id}/{arm}")
    return {"sample_id": sample_id, "artifact_sha256": sha256_file(path)}


def execute_trial(plan: dict, trial: dict, *, device: str, batch_size: int, resume: bool,
                  shard_index: int = 0, num_shards: int = 1) -> None:
    from ttt_frame.idea1_scaling_memory import run_scaling_forecasts_batch
    from scripts.run_idea1_mcq_forecasts import validated_cache
    split, source, baseline, config, initial, protocol = load_trial_inputs(plan, trial, device)
    protocol["execution_batch_size"] = batch_size
    protocol["case_shards"] = num_shards
    digest = canonical_sha256(protocol)
    output = Path(trial["forecast_dir"])
    output.mkdir(parents=True, exist_ok=True)
    installed = output / "protocol.json"
    if installed.exists():
        if not resume or json.loads(installed.read_text()) != {
                "protocol_sha256": digest, "protocol": protocol}:
            raise ValueError(f"existing incompatible trial protocol {trial['name']}")
    else:
        if any(output.glob("*.pt")):
            raise ValueError("unbound outputs exist without protocol")
        atomic_json(installed, {"protocol_sha256": digest, "protocol": protocol})
    all_ids = [sid for role in ("train", "validation", "test") for sid in split["splits"][role]]
    ids = [sid for index, sid in enumerate(all_ids) if index % num_shards == shard_index]
    pending, done = [], []
    for sample_id in ids:
        path = output / f"{sample_id}_forecasts.pt"
        if path.exists():
            if not resume:
                raise ValueError(f"existing result requires --resume: {sample_id}")
            done.append(validate_saved(path, sample_id, digest))
        else:
            pending.append(sample_id)
    started = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        batch_ids = pending[offset:offset + batch_size]
        sequences, feature_hashes = [], {}
        for sample_id in batch_ids:
            cache_path = Path(plan["feature_cache"]) / f"{sample_id}.pt"
            cache, identity = validated_cache(cache_path, sample_id, split,
                                              source["protocol"]["feature_protocol"])
            if identity != source["protocol"]["cache_sources"][sample_id]:
                raise ValueError(f"feature cache changed: {sample_id}")
            sequences.append(cache["features"])
            feature_hashes[sample_id] = identity["feature_sha256"]
        batch_start = time.perf_counter()
        results = run_scaling_forecasts_batch(torch.stack(sequences), baseline, config,
                                              arms=tuple(trial["arms"]), initial_weights=initial,
                                              device=device)
        elapsed = time.perf_counter() - batch_start
        for sample_id, result in zip(batch_ids, results, strict=True):
            per_arm = result.pop("arms")
            forecasts = {arm: row.pop("last_forecasts") for arm, row in per_arm.items()}
            payload = {"schema": "idea01_scaling_case_forecasts_v1", "sample_id": sample_id,
                       "protocol_sha256": digest, "trial": trial["name"],
                       "last_forecasts": forecasts, "stats": per_arm,
                       "feature_sha256": feature_hashes[sample_id],
                       "forecast_sha256": {arm: tensor_sha(value) for arm, value in forecasts.items()},
                       "result_metadata": result, "batch_seconds": elapsed,
                       "batch_cases": len(batch_ids), "execution_device": device}
            payload["metadata_sha256"] = canonical_sha256({
                key: value for key, value in payload.items() if key != "last_forecasts"})
            destination = output / f"{sample_id}_forecasts.pt"
            temporary = destination.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)
            done.append(validate_saved(destination, sample_id, digest))
        print(json.dumps({"trial": trial["name"], "done": len(done), "total": len(ids),
                          "batch_seconds": elapsed, "batch_cases": len(batch_ids)}), flush=True)
    summary_name = "summary.json" if num_shards == 1 else f"summary_shard_{shard_index:02d}.json"
    atomic_json(output / summary_name, {
        "complete": True, "cases": done, "protocol_sha256": digest,
        "elapsed_seconds": time.perf_counter() - started, "device": device,
        "batch_size": batch_size, "reused": len(ids) - len(pending),
        "shard_index": shard_index, "num_shards": num_shards,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/forecast_bench_v0/idea01_scaling_v1"))
    parser.add_argument("--base-run", type=Path, default=Path("runs/forecast_bench_v0/idea01_mcq_v1"))
    parser.add_argument("--split-manifest", type=Path, default=Path("runs/forecast_bench_v0/research_day_split.json"))
    parser.add_argument("--create-plan", action="store_true")
    parser.add_argument("--trials", nargs="+")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.threads < 1 or args.num_shards < 1
            or not 0 <= args.shard_index < args.num_shards):
        parser.error("positive threads and batch size required")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if args.create_plan:
        plan = make_plan(args.base_run, args.output_dir, args.split_manifest)
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("a different plan already exists")
        atomic_json(plan_path, plan)
        print(f"Predeclared {len(plan['trials'])} trials: {plan_path}", flush=True)
        return 0
    plan = json.loads(plan_path.read_text())
    selected = args.trials or [trial["name"] for trial in plan["trials"]]
    if len(set(selected)) != len(selected) or set(selected) - {trial["name"] for trial in plan["trials"]}:
        raise ValueError("trials must be unique members of the installed plan")
    for trial in plan["trials"]:
        if trial["name"] in selected:
            execute_trial(plan, trial, device=args.device, batch_size=args.batch_size, resume=args.resume,
                          shard_index=args.shard_index, num_shards=args.num_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
