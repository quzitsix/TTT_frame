#!/usr/bin/env python3
"""Select an inverse reader using validation, freeze predictions, then score separately.

The predict stage accesses correct_answer only for validation IDs. Association
scores and the common frozen prior are standardized within each question; this
uses neither gold labels nor cross-question/test statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROLES = ("train", "validation", "test")
BETAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
RADII = (("r01", 0.1), ("r05", 0.5), ("r10", 1.0))
REQUIRED_SCORES = ("direct", "wrong_direct", "reset_direct", "reset_search") + tuple(
    f"{prefix}{radius}" for prefix in ("search_", "wrong_search_") for radius, _ in RADII
)
ARMS = ("prior", "direct", "direct_matched", "search", "reset", "frozen_init",
        "wrong_direct", "wrong_direct_matched", "wrong_search", "direct_raw", "search_raw",
        "wrong_raw")
DEFAULT_BASELINE = REPO / "data/bench_v0_baseline_v1/qwen_full_icl_predictions.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in rows), encoding="utf-8")


def indexed(rows: list[dict], expected: set[str], name: str) -> dict[str, dict]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows) or set(result) != expected:
        raise ValueError(f"{name}: duplicate, missing or unexpected IDs")
    return result


def load_split(path: Path) -> dict:
    # Reuse the frozen day-split validator rather than accepting arbitrary ID folds.
    from scripts.run_forecast_residual import read_split
    split = read_split(path)
    if set(split["splits"]) != set(ROLES) or set(split["days"]) != set(ROLES):
        raise ValueError("only train/validation/test split roles are supported")
    return split


def outside_benchmark(output: Path, root: Path) -> None:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("output must be outside source benchmark")


def validate_inference_protocol(path: Path, split_path: Path, split: dict, root: Path,
                                *, current_sources: bool = True) -> dict:
    record = read_json(path)
    payload = record["payload"]
    if (payload.get("schema") != "idea03_native_spatial_probe_v1"
            or canonical_sha256(payload) != record["protocol_fingerprint"]):
        raise ValueError("inference protocol schema or fingerprint mismatch")
    baseline = payload["baseline_protocol"]
    base = baseline["payload"]
    if (base.get("schema") != "forecast_mcq_protocol_v1"
            or canonical_sha256(base) != baseline["protocol_fingerprint"]):
        raise ValueError("baseline protocol schema or fingerprint mismatch")
    public_sha = sha256_file(root / "inputs_mcq.jsonl")
    ids = set().union(*(set(split["splits"][r]) for r in ROLES))
    if (payload["split"] != split or payload["split_sha256"] != sha256_file(split_path)
            or split["inputs_manifest_sha256"] != public_sha or base["inputs_sha256"] != public_sha
            or set(base["case_ids"]) != ids or len(base["case_ids"]) != len(ids)
            or Path(base["root"]).resolve() != root.resolve()
            or Path(split["root"]).resolve() != root.resolve()):
        raise ValueError("inference/public input/day split disagreement")
    if (payload.get("answer_keys_used") is not False or payload.get("future_media_used") is not False
            or payload.get("candidate_letters_or_positions_encoded") is not False):
        raise ValueError("inference protocol violates causal candidate boundary")
    if payload["searches"] != {f"search_{name}": radius for name, radius in RADII}:
        raise ValueError("unsupported declared search radius grid")
    if (not isinstance(payload["steps"], int) or payload["steps"] < 0
            or not math.isfinite(payload["step_size"]) or payload["step_size"] <= 0
            or not math.isfinite(payload["regularization"]) or payload["regularization"] < 0):
        raise ValueError("invalid declared search configuration")
    expected_donors = {}
    for sid in ids:
        candidates = sorted(donor for donor in split["splits"]["train"]
                            if split["id_to_day"][donor] != split["id_to_day"][sid])
        if not candidates:
            raise ValueError("no independent-day training donor")
        expected_donors[sid] = candidates[0]
    if payload["donors"] != expected_donors:
        raise ValueError("donor mapping does not follow fixed train-day rule")
    if current_sources:
        for name, hashes in (("inference", payload["sources"]), ("baseline", base["source_hashes"])):
            if not hashes:
                raise ValueError(f"missing {name} source hashes")
            for file, expected in hashes.items():
                source = (REPO / file).resolve()
                if not source.is_relative_to(REPO) or not source.is_file() or sha256_file(source) != expected:
                    raise ValueError(f"{name} source changed: {file}")
    return record


def validate_sampling(row: dict, public: dict, sampling: dict, root: Path) -> None:
    audit = row["sampling"]
    declared = audit["sampling"]
    if (audit.get("schema") != "bench_v0_mcq_past_frame_bundle_v1" or audit.get("sample_id") != row["id"]
            or audit.get("decoded") is not True or audit.get("audio_used") is not False
            or any(declared[key] != sampling[key] for key in ("context_frames", "recent_frames", "max_side"))):
        raise ValueError("row frame sampling disagrees with inference protocol")
    context_seconds = float(public["input"]["context_duration_seconds"])
    recent_seconds = float(public["input"]["input_duration_seconds"])
    cutoff = context_seconds + recent_seconds
    if declared["cutoff_prefix_seconds"] != cutoff:
        raise ValueError("row cutoff disagrees with public input")
    for role, offset, duration, public_key in (
            ("context", 0., context_seconds, "context_video"),
            ("recent", context_seconds, recent_seconds, "video")):
        n = sampling[f"{role}_frames"]
        requested, actual = audit[f"{role}_requested_seconds"], audit[f"{role}_actual_seconds"]
        expected = [offset + duration * i / n for i in range(n)]
        if (len(requested) != n or len(actual) != n
                or any(not math.isfinite(t) or abs(t-e) > 1e-8 for t, e in zip(requested, expected))
                or any(not math.isfinite(t) or t < offset or t >= cutoff or t > requested[i] + 1e-9
                       or (i and t < actual[i-1]) for i, t in enumerate(actual))):
            raise ValueError("invalid or noncausal actual frame timestamps")
        media = audit["source_media"][role]
        if (Path(media["path"]).resolve() != (root / public["input"][public_key]).resolve()
                or type(media["size_bytes"]) is not int or media["size_bytes"] <= 0
                or type(media["mtime_ns"]) is not int):
            raise ValueError("source media identity disagrees with public input")
    for key in ("frame_plan_sha256", "frame_bundle_sha256", "context_frame_bundle_sha256", "recent_frame_bundle_sha256"):
        digest = audit[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing decoded pixel/time hash")


def validate_search_diagnostics(row: dict, payload: dict) -> None:
    for prefix in ("", "wrong_"):
        for key, radius in payload["searches"].items():
            result = row["diagnostics"][prefix + key]
            expected = {"radius": radius, **{name: payload[name] for name in ("steps", "step_size", "regularization")}}
            if (result["search"] != expected or result["layers"] != len(result["per_layer"])
                    or result["layers"] <= 0 or not math.isfinite(result["max_displacement"])
                    or not 0 <= result["max_displacement"] <= radius + 1e-5):
                raise ValueError("search diagnostic configuration or radius mismatch")
            for field in ("direct_scores", "search_scores", "direct_objectives", "search_objectives"):
                normalize(result[field])
            if (result["direct_scores"] != row["scores"][prefix + "direct"]
                    or result["direct_objectives"] != result["direct_scores"]
                    or result["search_objectives"] != row["scores"][prefix + key]):
                raise ValueError("candidate scores disagree with search diagnostics")
            if any(a + 1e-6 < b for a, b in zip(result["search_objectives"], result["direct_objectives"])):
                raise ValueError("best-iterate search regressed below initial objective")


def load_labels(path: Path, allowed: set[str]) -> dict[str, str]:
    """Filter ID before accessing answer fields; test answer fields are not used in predict."""
    labels = {}
    for row in read_rows(path):
        if row["id"] not in allowed:
            continue
        if row["id"] in labels or row["correct_answer"] not in "ABCDEF" or len(row["correct_answer"]) != 1:
            raise ValueError("duplicate or invalid requested label")
        labels[row["id"]] = row["correct_answer"]
    if set(labels) != allowed:
        raise ValueError("missing requested labels")
    return labels


def normalize(values: list[float]) -> list[float]:
    values = [float(v) for v in values]
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("six finite candidate scores required")
    # fsum avoids candidate-order-dependent summation; no statistics cross questions.
    mean = math.fsum(values) / len(values)
    deviations = [v - mean for v in values]
    scale = math.sqrt(math.fsum(v * v for v in deviations) / len(values))
    if scale <= 1e-12:
        return [0.0] * len(values)
    return [v / scale for v in deviations]


def select_label(row: dict, scores: list[float]) -> str:
    best = max(scores)
    candidates = [i for i, value in enumerate(scores) if value == best]
    # Content, rather than the A-F position, resolves exact score ties.
    winner = min(candidates, key=lambda i: (row["option_texts"][i], row["labels"][i]))
    return row["labels"][winner]


def mixed_scores(row: dict, memory_key: str, beta: float) -> list[float]:
    prior = normalize(row["prior_scores"])
    memory = normalize(row["scores"][memory_key])
    return [p + beta * m for p, m in zip(prior, memory)]


def validate_scores(path: Path, split: dict, root: Path, inference: dict) -> dict[str, dict]:
    expected = set().union(*(set(split["splits"][r]) for r in ROLES))
    records = indexed(read_rows(path), expected, "scores")
    public = indexed(read_rows(root / "inputs_mcq.jsonl"), expected, "public inputs")
    roles = {sid: role for role in ROLES for sid in split["splits"][role]}
    payload = inference["payload"]
    for sid, row in records.items():
        if (row.get("schema") != "idea03_native_spatial_scores_v1" or row.get("status") != "ok"
                or row.get("answer_keys_used") is not False
                or row.get("protocol_fingerprint") != inference["protocol_fingerprint"]
                or row.get("donor_id") != payload["donors"][sid]):
            raise ValueError("row inference provenance, status or donor mismatch")
        validate_sampling(row, public[sid], payload["baseline_protocol"]["payload"]["sampling"], root)
        validate_search_diagnostics(row, payload)
        if row.get("research_split") != roles[sid] or len(row["labels"]) != 6 or set(row["labels"]) != set("ABCDEF"):
            raise ValueError("score role or labels disagree with protocol")
        options = public[sid]["input"]["options"]
        expected_text = {option["label"]: option["text"] for option in options}
        if len(row["option_texts"]) != 6 or dict(zip(row["labels"], row["option_texts"])) != expected_text:
            raise ValueError("candidate text does not match public inputs")
        normalize(row["prior_scores"])
        for key in REQUIRED_SCORES:
            normalize(row["scores"][key])
        for key in ("reset_direct", "reset_search"):
            if any(v != 0 for v in row["scores"][key]):
                raise ValueError("reset association delta must be exactly zero")
    return records


def select_configuration(records: dict[str, dict], labels: dict[str, str]) -> dict:
    def trial(key: str, beta: float, radius: float | None) -> dict:
        correct = sum(select_label(records[sid], mixed_scores(records[sid], key, beta)) == answer
                      for sid, answer in labels.items())
        return {"memory_key": key, "beta": beta, "radius": radius, "correct": correct,
                "n": len(labels)}
    direct = [trial("direct", beta, None) for beta in BETAS]
    search = [trial(f"search_{name}", beta, radius) for name, radius in RADII for beta in BETAS]
    return {"direct": min(direct, key=lambda x: (-x["correct"], x["beta"])),
            "search": min(search, key=lambda x: (-x["correct"], x["beta"], x["radius"])),
            "direct_grid": direct, "search_grid": search,
            "selection_rule": "validation correct descending, beta ascending, radius ascending",
            "labels_used": "validation only", "validation_labels_sha256": canonical_sha256(labels)}


def predict(scores: Path, split_manifest: Path, root: Path, output: Path, inference_protocol: Path) -> dict:
    outside_benchmark(output, root)
    if output.exists():
        raise FileExistsError("prediction output must be new")
    split = load_split(split_manifest)
    inference = validate_inference_protocol(inference_protocol, split_manifest, split, root)
    records = validate_scores(scores, split, root, inference)
    native = inference["payload"]
    sampling = native["baseline_protocol"]["payload"]["sampling"]
    public_rows = read_rows(root / "inputs_mcq.jsonl")
    durations = {(row["input"]["context_duration_seconds"], row["input"]["input_duration_seconds"])
                 for row in public_rows}
    if len(durations) != 1:
        raise ValueError("budget summary requires shared context/recent durations")
    context_seconds, recent_seconds = next(iter(durations))
    validation = load_labels(root / "answers.jsonl", set(split["splits"]["validation"]))
    selection = select_configuration(records, validation)
    selected_search = selection["search"]
    search_key, beta = selected_search["memory_key"], selected_search["beta"]
    direct_beta = selection["direct"]["beta"]
    protocol = {
        "schema": "idea03_spatial_evaluation_protocol_v1", "scores_file": str(scores.resolve()),
        "scores_sha256": sha256_file(scores), "split_sha256": sha256_file(split_manifest),
        "inference_protocol_sha256": sha256_file(inference_protocol),
        "inference_protocol_fingerprint": inference["protocol_fingerprint"],
        "benchmark_root": str(root.resolve()),
        "public_inputs_sha256": sha256_file(root / "inputs_mcq.jsonl"), "splits": split["splits"],
        "id_to_day": split.get("id_to_day", {}), "arms": list(ARMS), "selection": selection,
        "normalization": "population z-score within question and stream; std<=1e-12 gives zero",
        "tie_break": "lexicographic candidate text, then label only for identical text",
        "search": {**{name: native[name] for name in ("steps", "step_size", "regularization")},
                   "searches": native["searches"]},
        "source_sha256": {name: sha256_file(REPO / name) for name in (
            "scripts/evaluate_idea3_spatial.py", "scripts/run_forecast_residual.py")},
        "test_status": "exploratory reuse; silver labels; overlapping windows; only two test days",
        "input_budget": {**sampling, "context_seconds": context_seconds,
                         "recent_seconds": recent_seconds, "benchmark_training": False,
                         "validation_selection": True, "candidate_searches_per_radius": 6,
                         "comparison": "same declared raw-frame sampling as full ICL; different checkpoint/readout/compute"},
    }
    fingerprint = canonical_sha256(protocol)
    predictions = []
    for role in ROLES:
        for sid in split["splits"][role]:
            row = records[sid]
            prior = normalize(row["prior_scores"])
            arm_scores = {
                "prior": prior, "direct": mixed_scores(row, "direct", direct_beta),
                "direct_matched": mixed_scores(row, "direct", beta),
                "search": mixed_scores(row, search_key, beta),
                "reset": mixed_scores(row, "reset_search", beta), "frozen_init": prior,
                "wrong_direct": mixed_scores(row, "wrong_direct", direct_beta),
                "wrong_direct_matched": mixed_scores(row, "wrong_direct", beta),
                "wrong_search": mixed_scores(row, "wrong_" + search_key, beta),
                "direct_raw": normalize(row["scores"]["direct"]),
                "search_raw": normalize(row["scores"][search_key]),
                "wrong_raw": normalize(row["scores"]["wrong_" + search_key]),
            }
            predictions.append({"id": sid, "research_split": role,
                "protocol_fingerprint": fingerprint, "labels": row["labels"],
                "predictions": {arm: select_label(row, value) for arm, value in arm_scores.items()},
                "candidate_scores": arm_scores})
    output.mkdir(parents=True)
    shutil.copy2(inference_protocol, output / "inference_protocol.json")
    shutil.copy2(split_manifest, output / "split_manifest.json")
    write_json(output / "protocol.json", {"protocol": protocol, "protocol_fingerprint": fingerprint})
    write_json(output / "validation_selection.json", selection)
    write_rows(output / "predictions.jsonl", predictions)
    completion = {"schema": "idea03_spatial_prediction_completion_v1", "complete": True,
                  "protocol_fingerprint": fingerprint, "test_answer_fields_used": False,
                  "prediction_rows": len(predictions),
                  "predictions_sha256": sha256_file(output / "predictions.jsonl")}
    write_json(output / "completion.json", completion)
    return selection


def verify_predictions(output: Path) -> tuple[dict, dict[str, dict]]:
    installed = read_json(output / "protocol.json")
    protocol, fingerprint = installed["protocol"], installed["protocol_fingerprint"]
    completion = read_json(output / "completion.json")
    if (canonical_sha256(protocol) != fingerprint or completion.get("complete") is not True
            or completion.get("test_answer_fields_used") is not False
            or completion.get("protocol_fingerprint") != fingerprint
            or sha256_file(output / "predictions.jsonl") != completion["predictions_sha256"]):
        raise ValueError("prediction protocol or completion checksum mismatch")
    if (sha256_file(output / "inference_protocol.json") != protocol["inference_protocol_sha256"]
            or sha256_file(output / "split_manifest.json") != protocol["split_sha256"]):
        raise ValueError("saved inference protocol or split checksum mismatch")
    inference = read_json(output / "inference_protocol.json")
    if (canonical_sha256(inference["payload"]) != inference["protocol_fingerprint"]
            or inference["protocol_fingerprint"] != protocol["inference_protocol_fingerprint"]):
        raise ValueError("saved inference fingerprint mismatch")
    ids = set().union(*(set(protocol["splits"][r]) for r in ROLES))
    rows = indexed(read_rows(output / "predictions.jsonl"), ids, "frozen predictions")
    for role in ROLES:
        for sid in protocol["splits"][role]:
            row = rows[sid]
            if (row["protocol_fingerprint"] != fingerprint or row["research_split"] != role
                    or set(row["predictions"]) != set(ARMS)
                    or any(value not in set("ABCDEF") for value in row["predictions"].values())):
                raise ValueError("invalid frozen prediction")
    return protocol, rows


def paired(left: dict[str, str], right: dict[str, str], gold: dict[str, str], ids: list[str]) -> dict:
    gain = [sid for sid in ids if left[sid] == gold[sid] and right[sid] != gold[sid]]
    loss = [sid for sid in ids if left[sid] != gold[sid] and right[sid] == gold[sid]]
    return {"n": len(ids), "gains": len(gain), "losses": len(loss), "net": len(gain) - len(loss),
            "disagreements": sum(left[sid] != right[sid] for sid in ids),
            "gain_ids": gain, "loss_ids": loss}


def score(scores: Path, split_manifest: Path, root: Path, output: Path,
          baseline: Path = DEFAULT_BASELINE, *, inference_protocol: Path) -> dict:
    outside_benchmark(output, root)
    if (output / "report").exists():
        raise FileExistsError("score report must be new")
    protocol, rows = verify_predictions(output)
    for path, expected in ((inference_protocol, protocol["inference_protocol_sha256"]),
                           (scores, protocol["scores_sha256"]),
                           (split_manifest, protocol["split_sha256"]),
                           (root / "inputs_mcq.jsonl", protocol["public_inputs_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError("scoring input checksum mismatch")
    for file, expected in protocol["source_sha256"].items():
        if sha256_file(REPO / file) != expected:
            raise ValueError("evaluation source changed since predictions")
    ids = set(rows)
    baseline_rows = indexed(read_rows(baseline), ids, "full ICL baseline")
    baseline_pred = {sid: row["prediction"] for sid, row in baseline_rows.items()}
    if any(value not in set("ABCDEF") for value in baseline_pred.values()):
        raise ValueError("invalid baseline prediction")
    # This is the first stage that requests test labels. Predictions are already fixed.
    gold = load_labels(root / "answers.jsonl", ids)
    arms = {arm: {sid: row["predictions"][arm] for sid, row in rows.items()} for arm in ARMS}
    arms["full_icl"] = baseline_pred
    metrics = {}
    for arm, pred in arms.items():
        metrics[arm] = {}
        for role, role_ids in protocol["splits"].items():
            count = sum(pred[sid] == gold[sid] for sid in role_ids)
            days = sorted({protocol["id_to_day"].get(sid) for sid in role_ids} - {None})
            metrics[arm][role] = {"n": len(role_ids), "correct": count, "accuracy": count / len(role_ids),
                "by_day": {str(day): {"n": sum(protocol["id_to_day"].get(sid) == day for sid in role_ids),
                    "correct": sum(pred[sid] == gold[sid] for sid in role_ids
                                   if protocol["id_to_day"].get(sid) == day)} for day in days}}
    pairs = {}
    for arm, pred in arms.items():
        if arm == "full_icl":
            continue
        pairs[arm] = {control: {role: paired(pred, arms[control], gold, role_ids)
                              for role, role_ids in protocol["splits"].items()}
                      for control in ("full_icl", "direct", "direct_matched", "prior")}
    raw = read_rows(scores)
    diagnostics = {}
    for radius, _ in RADII:
        key = f"search_{radius}"
        for role in ROLES:
            changes = [a-b for row in raw if row["research_split"] == role
                       for a, b in zip(row["scores"][key], row["scores"]["direct"])]
            diagnostics[f"{key}/{role}"] = {"candidate_count": len(changes),
                "mean_regularized_objective_improvement": statistics.fmean(changes),
                "negative_objective_changes": sum(v < -1e-6 for v in changes),
                "interpretation": "searched regularized objective minus direct objective; not task correctness"}
    summary = {"schema": "idea03_spatial_results_v1", "protocol_fingerprint": canonical_sha256(protocol),
               "selection": protocol["selection"], "metrics": metrics, "paired": pairs,
               "score_diagnostics": diagnostics, "input_budget": protocol["input_budget"],
               "baseline_file": str(baseline.resolve()), "baseline_sha256": sha256_file(baseline),
               "predictions_sha256": sha256_file(output / "predictions.jsonl"),
               "answers_sha256": sha256_file(root / "answers.jsonl"),
               "test_status": protocol["test_status"],
               "inference": "No iid significance interval: test has only two recording-day clusters."}
    report = output / "report"
    report.mkdir()
    write_json(report / "summary.json", summary)
    cases = [{"id": sid, "research_split": rows[sid]["research_split"],
              "day": protocol["id_to_day"].get(sid),
              "predictions": {arm: pred[sid] for arm, pred in arms.items()},
              "correct": {arm: pred[sid] == gold[sid] for arm, pred in arms.items()}} for sid in rows]
    write_rows(report / "per_case.jsonl", cases)
    with (report / "per_case.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "research_split", "day"] +
            [f"{arm}_{suffix}" for arm in arms for suffix in ("prediction", "correct")])
        writer.writeheader()
        for case in cases:
            writer.writerow({"id": case["id"], "research_split": case["research_split"], "day": case["day"],
                **{f"{arm}_{suffix}": case["predictions" if suffix == "prediction" else "correct"][arm]
                   for arm in arms for suffix in ("prediction", "correct")}})
    write_json(report / "manifest.json", {"schema": "idea03_spatial_scoring_completion_v1",
        "complete": True, "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                                     for p in sorted(report.iterdir()) if p.is_file()}})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--inference-protocol", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-icl", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    if args.stage == "predict":
        result = predict(args.scores, args.split_manifest, args.root, args.output_dir, args.inference_protocol)
        print(json.dumps({k: result[k] for k in ("direct", "search")}, indent=2))
    else:
        result = score(args.scores, args.split_manifest, args.root, args.output_dir, args.full_icl,
                       inference_protocol=args.inference_protocol)
        print(json.dumps({arm: metrics["test"] for arm, metrics in result["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
