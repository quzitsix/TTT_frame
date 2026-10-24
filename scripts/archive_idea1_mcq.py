#!/usr/bin/env python3
"""Validate and archive completed Idea 01 results without copying answer keys.

This entry point reads saved aggregates, public protocols and past-only tensors.
It does not open benchmark answers, private metadata or future media. Diagnostics
compare forecasts to one another; they are not future-target prediction errors.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.cache_idea1_text import canonical_sha256 as text_protocol_hash, tensor_sha256 as text_tensor_hash
from scripts.evaluate_idea1_mcq import digest_json
from scripts.run_forecast_residual import read_split
from scripts.run_idea1_mcq_forecasts import (
    canonical_sha256, read_feature_protocol, sha256_file, validated_cache, validate_forecast,
)
from ttt_frame.forecast_residual import RidgeRecentPredictor
from ttt_frame.idea1_mcq_memory import ARMS, FinalForecastConfig, run_final_forecasts


def summary(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("diagnostic values must be finite and nonempty")
    return {"n": len(values), "min": min(values), "max": max(values),
            "mean": statistics.mean(values)}


def write_json(path: Path, value: object) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def frozen_parameter_counts(model_path: Path) -> dict:
    """Count checkpoint shapes without loading tensor values or using a GPU."""
    from safetensors import safe_open
    counts = {"visual": 0, "text_decoder": 0}
    tensors = {"visual": 0, "text_decoder": 0}
    for path in sorted(model_path.glob("model*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for name in source.keys():
                role = ("visual" if name.startswith("model.visual.") else
                        "text_decoder" if name.startswith("model.language_model.") else None)
                if role:
                    counts[role] += math.prod(source.get_slice(name).get_shape())
                    tensors[role] += 1
    if not all(counts.values()):
        raise ValueError("cannot identify frozen visual/text checkpoint components")
    return {"parameter_values": counts, "tensor_arrays": tensors,
            "scope": "checkpoint tensors in separately loaded frozen components; lm_head excluded"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/forecast_bench_v0/idea01_mcq_v1"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("runs/forecast_bench_v0/research_day_split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bench_v0_idea01_v1"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    run = args.run_dir
    split = read_split(args.split_manifest)
    ids = sorted(sample_id for values in split["splits"].values() for sample_id in values)
    test_ids = set(split["splits"]["test"])
    feature_record = read_feature_protocol(run / "features", split)
    record = json.loads((run / "forecasts/protocol.json").read_text())
    if record["protocol_sha256"] != canonical_sha256(record["protocol"]):
        raise ValueError("forecast protocol checksum mismatch")
    if (record["protocol"]["split_sha256"] != sha256_file(args.split_manifest)
            or record["protocol"]["splits"] != split["splits"]
            or record["protocol"]["feature_protocol"] != feature_record):
        raise ValueError("forecast/feature/split mismatch")
    predictor_path = run / "forecasts/recent_predictor.pt"
    if sha256_file(predictor_path) != record["predictor_sha256"]:
        raise ValueError("predictor checksum mismatch")
    baseline = RidgeRecentPredictor.from_dict(torch.load(
        predictor_path, map_location="cpu", weights_only=True)["baseline"])
    forecasts_summary = json.loads((run / "forecasts/summary.json").read_text())
    if forecasts_summary.get("complete") is not True:
        raise ValueError("forecast run incomplete")
    summary_cases = {item["sample_id"]: item for item in forecasts_summary["cases"]}
    if set(summary_cases) != set(ids):
        raise ValueError("forecast summary does not cover every split ID")
    text_cache = torch.load(run / "text_features.pt", map_location="cpu", weights_only=True)
    if (text_cache["protocol_sha256"] != text_protocol_hash(text_cache["protocol"])
            or text_cache["input_manifest_sha256"] != split["inputs_manifest_sha256"]
            or set(text_cache["records"]) != set(ids)):
        raise ValueError("text protocol identity mismatch")
    for item in text_cache["records"].values():
        if item["embedding_sha256"] != text_tensor_hash(item["embeddings"]):
            raise ValueError("text embedding checksum mismatch")
    readout = json.loads((run / "mcq/protocol.json").read_text())
    completion = json.loads((run / "mcq/completion.json").read_text())
    if (readout["protocol_fingerprint"] != digest_json(readout["protocol"])
            or readout["protocol"]["forecast_protocol_sha256"] != record["protocol_sha256"]
            or readout["protocol"]["text_cache_sha256"] != sha256_file(run / "text_features.pt")
            or readout["shared_readout_sha256"] != sha256_file(run / "mcq/shared_readout.pt")
            or completion.get("complete") is not True
            or completion["protocol_fingerprint"] != readout["protocol_fingerprint"]
            or completion["predictions_sha256"] != sha256_file(run / "mcq/predictions.jsonl")):
        raise ValueError("readout/completion identity mismatch")
    aggregate = json.loads((run / "report/aggregate_metrics.json").read_text())
    if (aggregate["protocol_fingerprint"] != readout["protocol_fingerprint"]
            or aggregate["inputs_sha256"] != split["inputs_manifest_sha256"]
            or aggregate.get("complete") is not True):
        raise ValueError("report protocol mismatch")

    feature_times, cache_bytes, case_times = [], 0, []
    test_stats = {arm: {} for arm in ARMS}
    deltas = {arm: {"normalized_mse": [], "normalized_l2": [], "relative_l2": []} for arm in ARMS}
    first_cache = first_forecast = None
    artifact_checks = {}
    for sample_id in ids:
        cache_path = run / "features" / f"{sample_id}.pt"
        cache, identity = validated_cache(cache_path, sample_id, split, feature_record)
        if identity != record["protocol"]["cache_sources"][sample_id]:
            raise ValueError(f"cache changed: {sample_id}")
        forecast_path = run / "forecasts" / f"{sample_id}_forecasts.pt"
        checked = validate_forecast(forecast_path, sample_id, record)
        if checked["artifact_sha256"] != summary_cases[sample_id]["artifact_sha256"]:
            raise ValueError(f"forecast changed after run completion: {sample_id}")
        if checked["artifact_sha256"] != readout["protocol"]["forecast_sha256"][sample_id]:
            raise ValueError(f"forecast changed after readout: {sample_id}")
        forecast = torch.load(forecast_path, map_location="cpu", weights_only=True)
        feature_times.append(float(cache["extraction_seconds"]))
        cache_bytes += cache_path.stat().st_size
        case_times.append(float(forecast["elapsed_seconds"]))
        if first_cache is None:
            first_cache, first_forecast = cache, forecast
        artifact_checks[sample_id] = {"cache_sha256": identity["cache_sha256"],
                                     "forecast_sha256": checked["artifact_sha256"]}
        if sample_id not in test_ids:
            continue
        recent = forecast["last_forecasts"]["recent_only"].double()
        recent_normalized = (recent - baseline.mean.double()) / baseline.std.double()
        for arm in ARMS:
            for name, value in forecast["stats"][arm].items():
                test_stats[arm].setdefault(name, []).append(float(value))
            difference = (forecast["last_forecasts"][arm].double() - recent) / baseline.std.double()
            l2 = float(torch.linalg.vector_norm(difference))
            deltas[arm]["normalized_mse"].append(float(difference.square().mean()))
            deltas[arm]["normalized_l2"].append(l2)
            deltas[arm]["relative_l2"].append(l2 / max(float(recent_normalized.norm()), 1e-12))
    # One past-only causal replay confirms that an actual update affects the
    # saved forecasts and that disabling updates removes the residual exactly.
    config = FinalForecastConfig(**record["protocol"]["online_config"])
    replay = run_final_forecasts(first_cache["features"], baseline, config)
    no_update = run_final_forecasts(first_cache["features"], baseline,
                                    replace(config, learning_rate=0), arms=("recent_only", "residual"))
    replay_diffs = {arm: float((replay["arms"][arm]["last_forecasts"]
                               - first_forecast["last_forecasts"][arm]).abs().max()) for arm in ARMS}
    no_update_delta = float((no_update["arms"]["residual"]["last_forecasts"]
                             - no_update["arms"]["recent_only"]["last_forecasts"]).abs().max())
    if any(value != 0 for value in replay_diffs.values()) or no_update_delta != 0:
        raise ValueError("replay or no-update control failed")

    workers = json.loads((run / "feature_workers.json").read_text())
    logs = [Path(worker["log"]) for worker in workers["workers"]]
    approximate_wall = max(path.stat().st_mtime for path in logs) - workers["launched_unix"]
    readout_checkpoint = torch.load(run / "mcq/shared_readout.pt", map_location="cpu", weights_only=True)
    diagnostics = {
        "schema": "idea01_execution_diagnostics_v1", "sample_count": len(ids),
        "test_count": len(test_ids), "private_answers_or_future_opened": False,
        "feature_extraction": {
            "frames_per_case": 3660, "total_frames": len(ids) * 3660,
            "case_seconds": summary(feature_times), "sum_case_seconds": sum(feature_times),
            "cache_total_file_bytes": cache_bytes, "workers": len(workers["workers"]),
            "devices": sorted({worker["device"] for worker in workers["workers"]}),
            "approximate_parallel_wall_seconds": approximate_wall,
            "wall_measurement": "last worker log mtime minus recorded 12-worker launch; includes process startup; one preflight cache predates launch and is reused",
            "worker_plan_sha256": sha256_file(run / "feature_workers.json"),
        },
        "forecasting": {"wall_seconds": forecasts_summary["elapsed_seconds"],
                        "recent_predictor_fit_seconds": forecasts_summary["fit_seconds"],
                        "case_seconds": summary(case_times), "sum_case_seconds": sum(case_times),
                        "workers": forecasts_summary["workers"], "threads_per_worker": forecasts_summary["threads"]},
        "readout": {"fit_select_and_predict_seconds": completion["elapsed_seconds"],
                    "text_cache_seconds": text_cache["elapsed_seconds"],
                    "selected_lambda": readout["protocol"]["selected_lambda"]},
        "test_state_statistics": {arm: {key: summary(values) for key, values in fields.items()}
                                  for arm, fields in test_stats.items()},
        "test_forecast_difference_from_recent_only": {
            arm: {key: summary(values) for key, values in fields.items()} for arm, fields in deltas.items()},
        "difference_scope": "forecast-to-forecast diagnostics in train-standardized feature space; not error against actual future targets",
        "past_only_replay": {"sample_id": first_forecast["sample_id"],
                            "max_abs_saved_forecast_delta_by_arm": replay_diffs,
                            "zero_lr_residual_vs_recent_max_abs_delta": no_update_delta},
        "parameter_counts": {
            "frozen_components": frozen_parameter_counts(Path(text_cache["protocol"]["model_path"])),
            "fast_memory_values_per_arm": 4 * 2 * 256 * config.hidden_dim,
            "fast_memory_bytes_per_arm": 4 * 2 * 256 * config.hidden_dim * 4,
            "recent_predictor_coefficient_values": baseline.coefficients.numel(),
            "recent_predictor_statistics_values": baseline.mean.numel() + baseline.std.numel(),
            "readout_coefficient_values": readout_checkpoint["coefficients"].numel(),
            "readout_coefficient_bytes": readout_checkpoint["coefficients"].numel() * readout_checkpoint["coefficients"].element_size(),
        },
        "artifact_validation": artifact_checks,
        "source_sha256": {"scripts/archive_idea1_mcq.py": sha256_file(Path(__file__))},
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    copies = {
        "aggregate_metrics.json": run / "report/aggregate_metrics.json",
        "readout_protocol.json": run / "mcq/protocol.json",
        "validation_selection.json": run / "mcq/validation_selection.json",
        "forecast_protocol.json": run / "forecasts/protocol.json",
        "feature_protocol.json": run / "features/cache_protocol.json",
        "research_day_split.json": args.split_manifest,
    }
    for arm in list(ARMS) + ["text_prior"]:
        path = run / "report" / f"{arm}_predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if (len(rows) != len(ids) or {row["id"] for row in rows} != set(ids)
                or any(set(row) != {"id", "prediction"} for row in rows)):
            raise ValueError("prediction export must contain only complete ID/prediction pairs")
        copies[path.name] = path
    for name, source in copies.items():
        (output / name).write_bytes(source.read_bytes())
    write_json(output / "text_protocol.json", {
        "protocol": text_cache["protocol"], "protocol_sha256": text_cache["protocol_sha256"],
        "cache_sha256": sha256_file(run / "text_features.pt"),
    })
    write_json(output / "execution_diagnostics.json", diagnostics)
    manifest = {
        "schema": "idea01_mcq_archive_v1", "protocol_fingerprint": readout["protocol_fingerprint"],
        "sample_count": len(ids), "prediction_rows": len(ids) * 8,
        "primary_test_count": len(test_ids),
        "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                  for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"},
    }
    write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps({"archived_files": len(manifest["files"]), "output": str(output),
                      "feature_extraction": diagnostics["feature_extraction"],
                      "forecasting": diagnostics["forecasting"], "readout": diagnostics["readout"],
                      "past_only_replay": diagnostics["past_only_replay"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
