#!/usr/bin/env python3
"""Train one shared semantic readout and freeze all Idea 01 MCQ predictions.

Only train/validation answer fields are exposed to this stage. Test answers
are read by the separate summarize_forecast_mcq.py stage after predictions.
This small supervised prototype is not an outer-trained Spatial-TTT model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_forecast_residual import read_split
from scripts.cache_idea1_text import canonical_sha256 as text_protocol_sha256, case_text_sha256, tensor_sha256
from scripts.run_idea1_mcq_forecasts import canonical_sha256 as forecast_protocol_sha256, validate_forecast
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.idea1_mcq_memory import ARMS
from ttt_frame.idea1_mcq_readout import SharedRidgeReadout, pool_forecasts


LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load_fit_labels(path: Path, split: dict, role: str) -> dict[str, str]:
    """Filter by public ID before accessing answer fields; test role is forbidden."""
    if role not in ("train", "validation"):
        raise ValueError("test labels are forbidden before frozen predictions")
    allowed = set(split["splits"][role])
    labels = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") not in allowed:
                continue
            sample_id, label = row["id"], row.get("correct_answer")
            if sample_id in labels or label not in tuple("ABCDEF"):
                raise ValueError(f"invalid/duplicate {role} answer for {sample_id}")
            labels[sample_id] = label
    if set(labels) != allowed:
        raise ValueError(f"missing {role} labels")
    return labels


def choose_lambda(records: list[dict]) -> float:
    """Validation total across the same seven arms; ties favor more regularization."""
    if not records:
        raise ValueError("empty validation selection")
    return float(max(records, key=lambda row: (row["correct_sum"], row["ridge_lambda"]))[
        "ridge_lambda"])


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.output_dir.exists():
        parser.error("positive --threads and a new --output-dir are required")
    if args.output_dir.resolve().is_relative_to(args.root.resolve()):
        parser.error("output directory must be outside source benchmark")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    split = read_split(args.split_manifest)
    cases = load_forecast_cases(args.root)
    ids = [case.id for case in cases]
    case_by_id = {case.id: case for case in cases}
    if set(ids) != set().union(*(set(values) for values in split["splits"].values())):
        raise ValueError("split must cover all public cases")
    public_sha = sha256_file(args.root / "inputs_mcq.jsonl")
    if public_sha != split["inputs_manifest_sha256"]:
        raise ValueError("split/public input hash mismatch")
    text_cache = torch.load(args.text_cache, map_location="cpu", weights_only=True)
    if (text_cache.get("schema") != "idea01_candidate_text_features_v1"
            or text_cache.get("input_manifest_sha256") != public_sha
            or set(text_cache["records"]) != set(ids)):
        raise ValueError("text cache/public input mismatch")
    if text_cache["protocol_sha256"] != text_protocol_sha256(text_cache["protocol"]):
        raise ValueError("corrupt text cache protocol")
    installed_protocol = json.loads((args.forecast_dir / "protocol.json").read_text())
    expected_protocol = installed_protocol["protocol_sha256"]
    forecast_protocol = installed_protocol["protocol"]
    if (expected_protocol != forecast_protocol_sha256(forecast_protocol)
            or forecast_protocol["split_sha256"] != sha256_file(args.split_manifest)
            or forecast_protocol["splits"] != split["splits"]
            or forecast_protocol["input_manifest_sha256"] != public_sha):
        raise ValueError("forecast protocol/split/public input mismatch")
    if installed_protocol["predictor_sha256"] != sha256_file(args.forecast_dir / "recent_predictor.pt"):
        raise ValueError("forecast predictor checksum mismatch")
    features, candidates, forecast_hashes = {}, {}, {}
    forecast_protocols, predictor_hashes = set(), set()
    for sample_id in ids:
        path = args.forecast_dir / f"{sample_id}_forecasts.pt"
        validate_forecast(path, sample_id, installed_protocol)
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("sample_id") != sample_id or set(artifact["last_forecasts"]) != set(ARMS):
            raise ValueError(f"invalid forecast identity/arms for {sample_id}")
        forecast_protocols.add(artifact["protocol_sha256"])
        predictor_hashes.add(artifact["predictor_sha256"])
        forecast_hashes[sample_id] = sha256_file(path)
        features[sample_id] = pool_forecasts(torch.stack([
            artifact["last_forecasts"][arm] for arm in ARMS]))
        record = text_cache["records"][sample_id]
        if record["labels"] != [option.label for option in case_by_id[sample_id].options]:
            raise ValueError(f"candidate label order mismatch for {sample_id}")
        if (record["text_sha256"] != case_text_sha256(case_by_id[sample_id])
                or record["embedding_sha256"] != tensor_sha256(record["embeddings"])
                or record["embeddings"].shape != (6, 2048)
                or not bool(torch.isfinite(record["embeddings"]).all())):
            raise ValueError(f"candidate content/embedding checksum mismatch for {sample_id}")
        candidates[sample_id] = record["embeddings"].float()
    if len(forecast_protocols) != 1 or len(predictor_hashes) != 1:
        raise ValueError("mixed forecast protocols or predictors")
    # Ensure the forecast directory's installed protocol agrees with every case.
    if forecast_protocols != {expected_protocol}:
        raise ValueError("forecast protocol manifest mismatch")
    train_ids, val_ids = split["splits"]["train"], split["splits"]["validation"]
    train_labels = load_fit_labels(args.root / "answers.jsonl", split, "train")
    val_labels = load_fit_labels(args.root / "answers.jsonl", split, "validation")

    def batch(selected: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.stack([features[sample_id] for sample_id in selected]),
                torch.stack([candidates[sample_id] for sample_id in selected]))

    def label_indices(selected: list[str], labels: dict) -> torch.Tensor:
        return torch.tensor([text_cache["records"][sample_id]["labels"].index(labels[sample_id])
                             for sample_id in selected], dtype=torch.long)

    train_x, train_options = batch(train_ids)
    val_x, val_options = batch(val_ids)
    train_y, val_y = label_indices(train_ids, train_labels), label_indices(val_ids, val_labels)
    selection, models = [], {}
    for ridge_lambda in LAMBDAS:
        model = SharedRidgeReadout.fit(
            train_x, train_options, train_y, ridge_lambda=ridge_lambda,
            train_ids=train_ids, arm_names=ARMS,
        )
        scores = model.predict_scores(val_x, val_options)
        counts = (scores.argmax(dim=-1) == val_y[:, None]).sum(dim=0)
        row = {"ridge_lambda": ridge_lambda, "correct_sum": int(counts.sum()),
               "independent_validation_cases": len(val_ids),
               "arm_predictions": len(val_ids) * len(ARMS),
               "correct_by_arm": dict(zip(ARMS, counts.tolist()))}
        selection.append(row)
        models[ridge_lambda] = model
        print(json.dumps(row), flush=True)
    selected_lambda = choose_lambda(selection)
    model = models[selected_lambda]
    del models
    protocol = {
        "schema": "idea01_mcq_readout_protocol_v1", "input_manifest_sha256": public_sha,
        "split_sha256": sha256_file(args.split_manifest), "train_ids": train_ids,
        "validation_ids": val_ids, "test_ids": split["splits"]["test"],
        "arms": list(ARMS) + ["text_prior"], "forecast_protocol_sha256": expected_protocol,
        "predictor_sha256": next(iter(predictor_hashes)), "forecast_sha256": forecast_hashes,
        "text_cache_sha256": sha256_file(args.text_cache),
        "text_protocol_sha256": text_cache["protocol_sha256"],
        "train_labels_sha256": digest_json(train_labels),
        "validation_labels_sha256": digest_json(val_labels),
        "ridge_lambda_grid": list(LAMBDAS), "selected_lambda": selected_lambda,
        "selection": "highest total validation correct over shared seven arms; ties larger lambda",
        "refit_validation": False, "test_labels_used": False,
        "outer_memory_training_performed": False, "seed": 0,
        "readout_training": f"{len(train_ids)} independent train cases, seven arm views each, one shared map",
        "input_budget": {"prefix_seconds": 1830, "frames_per_case": 3660,
                         "fps": 2, "max_side": 224},
        "comparability": "dense-video supervised prototype; 90-frame zero-shot VLM ICL is contextual reference only",
        "sources": {name: sha256_file(REPO / name) for name in (
            "scripts/evaluate_idea1_mcq.py", "ttt_frame/idea1_mcq_readout.py")},
        "torch": str(torch.__version__),
    }
    fingerprint = digest_json(protocol)
    args.output_dir.mkdir(parents=True)
    torch.save(model.to_dict(), args.output_dir / "shared_readout.pt")
    atomic_json(args.output_dir / "protocol.json", {
        "protocol_fingerprint": fingerprint, "protocol": protocol,
        "shared_readout_sha256": sha256_file(args.output_dir / "shared_readout.pt"),
    })
    atomic_json(args.output_dir / "validation_selection.json", selection)
    all_x, all_options = batch(ids)
    scores = model.predict_scores(all_x, all_options)
    blind_scores = model.blind_scores(all_options)
    rows = []
    for index, sample_id in enumerate(ids):
        labels = text_cache["records"][sample_id]["labels"]
        role = next(role for role, members in split["splits"].items() if sample_id in members)
        for arm_index, arm in enumerate(list(ARMS) + ["text_prior"]):
            option_scores = scores[index, arm_index] if arm in ARMS else blind_scores[index]
            if option_scores.shape != (6,) or not bool(torch.isfinite(option_scores).all()):
                raise ValueError(f"invalid readout scores for {sample_id}/{arm}")
            prediction = labels[int(option_scores.argmax())]
            rows.append({
                "schema": "forecast_mcq_prediction_v1", "id": sample_id, "arm": arm,
                "prediction": prediction, "raw_answer": prediction, "status": "ok",
                "protocol_fingerprint": fingerprint, "elapsed_seconds": 0.0,
                "candidate_scores": dict(zip(labels, option_scores.tolist())),
                "research_split": role,
                "training_case": role == "train", "validation_selection_case": role == "validation",
                "tie_count": int((option_scores == option_scores.max()).sum()),
            })
    temporary = args.output_dir / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    temporary.replace(args.output_dir / "predictions.jsonl")
    atomic_json(args.output_dir / "completion.json", {
        "complete": True, "rows": len(rows), "expected_rows": len(ids) * (len(ARMS) + 1),
        "elapsed_seconds": time.perf_counter() - started,
        "predictions_sha256": sha256_file(args.output_dir / "predictions.jsonl"),
        "test_answers_loaded": False, "protocol_fingerprint": fingerprint,
    })
    print(f"Frozen {len(rows)} predictions; selected lambda={selected_lambda}. "
          "Run separate scorer to inspect test results.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
