#!/usr/bin/env python3
"""Produce all seven Idea 01 forecasts from complete, past-only visual caches.

Every split member must have a complete 2fps cache before fitting starts. The
recent predictor is fitted on the declared training split only; all cases then
start independent online memories. This runner never opens answers or future
media. It emits latent forecasts for a separately trained, shared MCQ readout.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import uuid

import torch

# Support the documented ``python scripts/...py`` invocation as well as -m.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_forecast_residual import cache_signature, load_cache, read_split
from ttt_frame.forecast_residual import RidgeRecentPredictor
from ttt_frame.idea1_mcq_memory import ARMS, FinalForecastConfig, run_final_forecasts


SOURCE_FILES = (
    "scripts/run_idea1_mcq_forecasts.py", "scripts/run_forecast_residual.py",
    "ttt_frame/idea1_mcq_memory.py", "ttt_frame/forecast_residual.py",
    "ttt_frame/predictive_memory.py",
)
HORIZONS = (1, 2, 4, 10)
ROLES = ("train", "validation", "test")
_WORKER: dict = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy()).hexdigest()


def json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_feature_protocol(cache_dir: Path, split: dict) -> dict:
    path = cache_dir / "cache_protocol.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    protocol = record["protocol"]
    if record.get("protocol_sha256") != canonical_sha256(protocol):
        raise ValueError("corrupt feature protocol checksum")
    if (
        protocol.get("kind") != "idea01_full_prefix_visual_cache_protocol"
        or protocol.get("format_version") != 1
        or protocol.get("input_manifest_sha256") != split["inputs_manifest_sha256"]
        or protocol.get("feature_config", {}).get("sample_fps") != 2
        or protocol["feature_config"].get("context_tail_seconds") is not None
        or protocol["feature_config"].get("limit_frames") not in (None, 0)
        or protocol.get("storage_dtype") != "float32"
        or protocol.get("audio_used") is not False
        or protocol.get("future_media_used") is not False
        or protocol.get("questions_or_choices_encoded") is not False
        or protocol.get("execution", {}).get("images_attend_independently") is not True
    ):
        raise ValueError("feature protocol must describe full 2fps independent past-only images")
    return record


def validated_cache(path: Path, sample_id: str, split: dict, feature_record: dict) -> tuple[dict, dict]:
    """Validate metadata, actual tensor bytes, causality, and immutable source identity."""
    before = path.stat()
    cache = load_cache(path, sample_id, split, allow_partial=False)
    protocol = feature_record["protocol"]
    for key, expected in {
        "protocol_sha256": feature_record["protocol_sha256"],
        "model_fingerprint": protocol["model_fingerprint"],
        "execution": protocol["execution"],
        "config": protocol["feature_config"],
    }.items():
        if json_value(cache.get(key)) != json_value(expected):
            raise ValueError(f"cache feature protocol mismatch for {sample_id}: {key}")
    expected_times = torch.arange(3660, dtype=torch.float64) / 2
    if (cache["features"].shape != (3660, 8, 256)
            or cache["features"].dtype != torch.float32
            or cache["requested_prefix_seconds"].dtype != torch.float64
            or cache["actual_prefix_seconds"].dtype != torch.float64
            or cache["sample_fps"] != 2
            or not torch.equal(cache["requested_prefix_seconds"], expected_times)):
        raise ValueError(f"expected complete 3660-step float32 features / float64 2fps grid: {sample_id}")
    feature_sha = tensor_sha256(cache["features"])
    if cache.get("feature_sha256") != feature_sha:
        raise ValueError(f"corrupt feature checksum for {sample_id}")
    source_media = cache.get("source_media")
    if not isinstance(source_media, dict) or set(source_media) != {"context", "recent"}:
        raise ValueError(f"expected context/recent source media fingerprints: {sample_id}")
    for role, record in source_media.items():
        if (not isinstance(record, dict) or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64 or not isinstance(record.get("path"), str)
                or type(record.get("size_bytes")) is not int or record["size_bytes"] < 1
                or type(record.get("mtime_ns")) is not int):
            raise ValueError(f"invalid {role} source media fingerprint: {sample_id}")
        current = Path(record["path"]).stat()
        if (current.st_size, current.st_mtime_ns) != (record["size_bytes"], record["mtime_ns"]):
            raise ValueError(f"{role} source media changed since feature extraction: {sample_id}")
    file_sha = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"cache changed while validating: {path}")
    identity = {
        "cache_sha256": file_sha, "cache_size_bytes": after.st_size,
        "feature_sha256": feature_sha, "source_media": source_media,
        "requested_time_sha256": tensor_sha256(cache["requested_prefix_seconds"]),
        "actual_time_sha256": tensor_sha256(cache["actual_prefix_seconds"]),
    }
    return cache, identity


def prepare_inputs(cache_dir: Path, split_path: Path) -> tuple[dict, dict, list[torch.Tensor]]:
    """Validate *all* declared cases before fitting, retaining train tensors only."""
    split = read_split(split_path)
    manifest_path = Path(split["root"]) / "inputs_mcq.jsonl"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != split["inputs_manifest_sha256"]:
        raise ValueError("public input manifest checksum differs from research split")
    public_ids = [json.loads(line)["id"] for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [sample_id for role in ROLES for sample_id in split["splits"][role]]
    if len(set(public_ids)) != len(public_ids) or set(ids) != set(public_ids):
        raise ValueError("split must cover every public input ID exactly once")
    if any(Path(sample_id).name != sample_id or sample_id in (".", "..") for sample_id in ids):
        raise ValueError("sample IDs must be safe basenames")
    missing = [sample_id for sample_id in ids if not (cache_dir / f"{sample_id}.pt").is_file()]
    if missing:
        raise ValueError(f"feature cache is not ready: missing {len(missing)}/{len(ids)} cases; first IDs: {missing[:8]}")
    feature_record = read_feature_protocol(cache_dir, split)
    signatures, source_records, train = [], {}, []
    train_ids = set(split["splits"]["train"])
    for sample_id in ids:
        cache, identity = validated_cache(cache_dir / f"{sample_id}.pt", sample_id, split, feature_record)
        signatures.append(cache_signature(cache))
        source_records[sample_id] = identity
        if sample_id in train_ids:
            train.append(cache["features"])
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("feature representation differs between cases or splits")
    config = FinalForecastConfig()
    protocol = {
        "schema": "idea01_mcq_forecasts_protocol_v1",
        "split_sha256": sha256_file(split_path),
        "input_manifest_sha256": manifest_sha,
        "splits": split["splits"], "counts": {role: len(split["splits"][role]) for role in ROLES},
        "feature_protocol": feature_record, "feature_signature": signatures[0],
        "cache_sources": source_records,
        "recent_predictor": {"horizons": list(HORIZONS), "window_steps": 4,
                             "ridge": 0.01, "standardize": True, "max_fit_rows": 20000},
        "online_config": asdict(config), "arms": list(ARMS),
        "sample_fps": 2, "prefix_seconds": 1830, "sequence_steps": 3660,
        "outer_training_performed": False, "future_media_used": False,
        "answer_keys_used": False, "device": "cpu", "torch_version": str(torch.__version__),
        "source_sha256": {name: sha256_file(REPO_ROOT / name) for name in SOURCE_FILES},
    }
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError("public input manifest changed during cache validation")
    return split, json_value(protocol), train


def _worker_init(cache_dir: str, output_dir: str, split: dict, protocol_record: dict, threads: int) -> None:
    torch.set_num_threads(threads)
    predictor_path = Path(output_dir) / "recent_predictor.pt"
    if sha256_file(predictor_path) != protocol_record["predictor_sha256"]:
        raise ValueError("frozen recent predictor changed")
    payload = torch.load(predictor_path, map_location="cpu", weights_only=True)
    if payload.get("protocol_sha256") != protocol_record["protocol_sha256"]:
        raise ValueError("frozen recent predictor protocol mismatch")
    _WORKER.clear()
    _WORKER.update(cache_dir=Path(cache_dir), output_dir=Path(output_dir), split=split,
                   record=protocol_record, baseline=RidgeRecentPredictor.from_dict(payload["baseline"]))


def _run_case(sample_id: str) -> dict:
    started = time.perf_counter()
    record, split = _WORKER["record"], _WORKER["split"]
    protocol = record["protocol"]
    cache, identity = validated_cache(_WORKER["cache_dir"] / f"{sample_id}.pt", sample_id,
                                      split, protocol["feature_protocol"])
    if identity != protocol["cache_sources"][sample_id]:
        raise ValueError(f"cache changed since preflight: {sample_id}")
    result = run_final_forecasts(cache["features"], _WORKER["baseline"],
                                 FinalForecastConfig(**protocol["online_config"]), device="cpu")
    per_arm = result.pop("arms")
    forecasts = {arm: row.pop("last_forecasts") for arm, row in per_arm.items()}
    payload = {
        "schema": "idea01_mcq_case_forecasts_v1", "sample_id": sample_id,
        "split_role": next(role for role in ROLES if sample_id in split["splits"][role]),
        "protocol_sha256": record["protocol_sha256"], "predictor_sha256": record["predictor_sha256"],
        "cache_sha256": identity["cache_sha256"], "feature_sha256": identity["feature_sha256"],
        "last_forecasts": forecasts, "stats": per_arm, "result_metadata": result,
        "forecast_sha256": {arm: tensor_sha256(value) for arm, value in forecasts.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "last_actual_prefix_seconds": float(cache["actual_prefix_seconds"][-1]),
        "forecast_target_seconds_after_cutoff": [(h - 1) / 2 for h in HORIZONS],
    }
    payload["metadata_sha256"] = canonical_sha256({
        key: value for key, value in payload.items() if key != "last_forecasts"
    })
    destination = _WORKER["output_dir"] / f"{sample_id}_forecasts.pt"
    atomic_save(payload, destination)
    return {"sample_id": sample_id, "split_role": payload["split_role"],
            "elapsed_seconds": payload["elapsed_seconds"], "artifact_sha256": sha256_file(destination)}


def validate_forecast(path: Path, sample_id: str, record: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("metadata_sha256") != canonical_sha256({
        key: value for key, value in payload.items()
        if key not in ("last_forecasts", "metadata_sha256")
    }):
        raise ValueError(f"corrupt forecast metadata checksum: {sample_id}")
    protocol = record["protocol"]
    expected_role = next(role for role in ROLES if sample_id in protocol["splits"][role])
    expected = {
        "schema": "idea01_mcq_case_forecasts_v1", "sample_id": sample_id,
        "split_role": expected_role, "protocol_sha256": record["protocol_sha256"],
        "predictor_sha256": record["predictor_sha256"],
        "cache_sha256": protocol["cache_sources"][sample_id]["cache_sha256"],
        "feature_sha256": protocol["cache_sources"][sample_id]["feature_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"forecast identity/protocol mismatch: {sample_id}")
    if set(payload.get("last_forecasts", {})) != set(ARMS) or set(payload.get("stats", {})) != set(ARMS):
        raise ValueError(f"forecast arm set mismatch: {sample_id}")
    for arm, value in payload["last_forecasts"].items():
        if (not isinstance(value, torch.Tensor) or value.shape != (4, 8, 256)
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                or tensor_sha256(value) != payload.get("forecast_sha256", {}).get(arm)):
            raise ValueError(f"corrupt forecast for {sample_id}/{arm}")
    return {"sample_id": sample_id, "split_role": expected_role,
            "elapsed_seconds": payload["elapsed_seconds"], "artifact_sha256": sha256_file(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    split, protocol, train = prepare_inputs(args.cache_dir, args.split_manifest)
    protocol["threads_per_worker"] = args.threads
    if args.output_dir.resolve().is_relative_to(Path(split["root"]).resolve()):
        parser.error("output directory must be outside the source benchmark")
    if args.output_dir.resolve() == args.cache_dir.resolve():
        parser.error("output directory must differ from feature cache directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / ".runner.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f"active/stale runner lock: {lock}; overlapping runs are forbidden") from None
    try:
        protocol_path = args.output_dir / "protocol.json"
        digest = canonical_sha256(protocol)
        fit_seconds = None
        if protocol_path.exists():
            if not args.resume:
                raise ValueError("output exists; pass --resume to validate and reuse it")
            record = json.loads(protocol_path.read_text(encoding="utf-8"))
            if record.get("protocol_sha256") != digest or record.get("protocol") != protocol:
                raise ValueError("resume rejected: protocol, model, feature source, split, or implementation changed")
            predictor_path = args.output_dir / "recent_predictor.pt"
            if sha256_file(predictor_path) != record.get("predictor_sha256"):
                raise ValueError("resume rejected: recent predictor checksum changed")
            del train
        else:
            # An interrupted fit can leave an orphan predictor. Do not silently
            # overwrite it or mix unbound outputs with the new protocol.
            if any(args.output_dir.glob("*.pt")):
                raise ValueError("unbound artifacts exist without protocol.json; choose a new output directory")
            fit_started = time.perf_counter()
            baseline = RidgeRecentPredictor.fit(train, horizons=HORIZONS, window_steps=4,
                                                 ridge=0.01, standardize=True, max_fit_rows=20000)
            fit_seconds = time.perf_counter() - fit_started
            del train
            predictor_path = args.output_dir / "recent_predictor.pt"
            atomic_save({"kind": "idea01_frozen_recent_ridge", "baseline": baseline.to_dict(),
                         "protocol_sha256": digest, "train_ids": split["splits"]["train"]}, predictor_path)
            record = {"protocol_sha256": digest, "protocol": protocol,
                      "predictor_sha256": sha256_file(predictor_path), "fit_seconds": fit_seconds}
            atomic_json(record, protocol_path)
            del baseline
        ids = [sample_id for role in ROLES for sample_id in split["splits"][role]]
        pending, records = [], {}
        for sample_id in ids:
            destination = args.output_dir / f"{sample_id}_forecasts.pt"
            if destination.exists():
                if not args.resume:
                    raise ValueError(f"forecast already exists: {sample_id}")
                records[sample_id] = validate_forecast(destination, sample_id, record)
            else:
                pending.append(sample_id)
        print(json.dumps({"validated_cases": len(ids), "train_cases": len(split["splits"]["train"]),
                          "pending_cases": len(pending), "reused_cases": len(records),
                          "protocol_sha256": digest}), flush=True)
        initializer_args = (str(args.cache_dir.resolve()), str(args.output_dir.resolve()),
                            split, record, args.threads)
        if args.workers == 1:
            _worker_init(*initializer_args)
            for sample_id in pending:
                row = _run_case(sample_id)
                records[sample_id] = row
                print(json.dumps(row), flush=True)
        elif pending:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init, initargs=initializer_args) as executor:
                futures = {executor.submit(_run_case, sample_id): sample_id for sample_id in pending}
                for future in as_completed(futures):
                    row = future.result()
                    records[row["sample_id"]] = row
                    print(json.dumps(row), flush=True)
        # Manifest changes during a long run invalidate completion, even if
        # the already-computed artifacts remain available for inspection.
        if sha256_file(Path(split["root"]) / "inputs_mcq.jsonl") != protocol["input_manifest_sha256"]:
            raise ValueError("public input manifest changed during forecasting")
        atomic_json({
            "schema": "idea01_mcq_forecasts_summary_v1", "complete": True,
            "protocol_sha256": digest, "predictor_sha256": record["predictor_sha256"],
            "counts": protocol["counts"], "arms": list(ARMS),
            "elapsed_seconds": time.perf_counter() - started,
            "fit_seconds": record["fit_seconds"], "workers": args.workers, "threads": args.threads,
            "reused_cases": len(ids) - len(pending), "cases": [records[sample_id] for sample_id in ids],
            "future_media_used": False, "answer_keys_used": False,
        }, args.output_dir / "summary.json")
    finally:
        lock.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
