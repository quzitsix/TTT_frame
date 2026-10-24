#!/usr/bin/env python3
"""Shared train-only readouts for Idea 01 scaling, followed by separate scoring.

``--stage predict`` reads train/validation answer fields, never test labels.
``--stage score`` verifies frozen predictions before opening private references.
The reused 57-case test split is exploratory after the first Idea 01 experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import Tensor
from torch.nn import functional as F

from scripts.cache_idea1_text import canonical_sha256, case_text_sha256, tensor_sha256
from scripts.evaluate_idea1_mcq import load_fit_labels, sha256_file
from scripts.run_forecast_residual import read_split
from scripts.summarize_forecast_mcq import load_predictions, summarize, _paired_scores
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.idea1_mcq_readout import RIDGE_GRID, SharedRidgeReadout, pool_forecasts


IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")
ROLES = ("train", "validation", "test")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def nested_training_ids(split: dict, count: int) -> list[str]:
    """Nested, label-independent day-stratified ordering, fixed before scoring."""
    train = split["splits"]["train"]
    if type(count) is not int or not 2 <= count <= len(train):
        raise ValueError("readout training size must be between 2 and the available train cases")
    groups = defaultdict(list)
    for sample_id in train:
        groups[str(split["id_to_day"][sample_id])].append(sample_id)
    queues = {day: deque(sorted(values, key=lambda sample_id: hashlib.sha256(
        f"idea01_scaling_train_order_v1:{sample_id}".encode()).hexdigest()))
        for day, values in groups.items()}
    order = []
    while any(queues.values()):
        for day in sorted(queues):
            if queues[day]:
                order.append(queues[day].popleft())
    return order[:count]


@dataclass
class RidgeStatistics:
    feature_mean: Tensor
    feature_scale: Tensor
    text_mean: Tensor
    text_scale: Tensor
    target_mean: Tensor
    gram: Tensor
    cross: Tensor
    train_ids: tuple[str, ...]
    arm_names: tuple[str, ...]
    train_rows: int

    def solve(self, ridge_lambda: float) -> SharedRidgeReadout:
        if (isinstance(ridge_lambda, bool) or not math.isfinite(ridge_lambda)
                or ridge_lambda <= 0):
            raise ValueError("ridge_lambda must be finite and positive")
        regularized = self.gram.clone()
        regularized.diagonal().add_(ridge_lambda)
        coefficients = torch.linalg.solve(regularized, self.cross)
        return SharedRidgeReadout(
            self.feature_mean.clone(), self.feature_scale.clone(), self.text_mean.clone(),
            self.text_scale.clone(), coefficients, self.target_mean.clone(),
            float(ridge_lambda), len(self.train_ids), self.train_rows, self.train_ids,
            self.arm_names,
        )


def prepare_ridge_statistics(
    features: Tensor, options: Tensor, correct_indices: Tensor, *,
    view_weights: Tensor | None = None, train_ids: list[str], view_names: list[str],
) -> RidgeStatistics:
    """Primal sufficient statistics; equal-view case matches the original dual fit.

    Each independent case has total weight one. View weights sum to one within
    the case; the caller assigns equal trial weight, then equal arm weight.
    No extra supervision is implied by repeated method views of the same case.
    """
    x = features.detach().cpu().double()
    candidates = options.detach().cpu().double()
    if (x.ndim != 3 or candidates.ndim != 3 or x.shape[0] != candidates.shape[0]
            or x.shape[0] < 2 or min(x.shape[1:]) < 1 or candidates.shape[1] != 6
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(candidates).all())):
        raise ValueError("finite features[case,view,dim] and options[case,6,text_dim] required")
    n, views, width = x.shape
    if (len(train_ids) != n or len(set(train_ids)) != n or len(view_names) != views
            or len(set(view_names)) != views):
        raise ValueError("unique training IDs and view names must match tensor dimensions")
    if (correct_indices.shape != (n,) or correct_indices.dtype not in (torch.int32, torch.int64)
            or not bool(((correct_indices >= 0) & (correct_indices < 6)).all())):
        raise ValueError("correct indices must be a valid per-case integer vector")
    weights = torch.ones(views, dtype=torch.float64) if view_weights is None else view_weights.double().cpu()
    if (weights.shape != (views,) or not bool(torch.isfinite(weights).all())
            or not bool((weights > 0).all())):
        raise ValueError("view weights must be finite and strictly positive")
    weights = weights / weights.sum()
    feature_mean = (x * weights[None, :, None]).sum(dim=(0, 1)) / n
    variance = ((x - feature_mean).square() * weights[None, :, None]).sum(dim=(0, 1)) / n
    feature_scale = variance.sqrt().clamp_min(1e-6)
    text_rows = candidates.flatten(0, 1)
    text_mean = text_rows.mean(dim=0)
    text_scale = text_rows.std(dim=0, correction=0).clamp_min(1e-6)
    transformed = F.normalize((candidates - text_mean) / text_scale, dim=-1)
    target = transformed[torch.arange(n), correct_indices.detach().cpu().long()]
    target_mean = target.mean(dim=0)
    z = ((x - feature_mean) / feature_scale).reshape(n * views, width)
    row_weights = weights.repeat(n) / n
    gram = z.T @ (z * row_weights[:, None])
    centered_target = (target - target_mean).repeat_interleave(views, dim=0)
    cross = z.T @ (centered_target * row_weights[:, None])
    return RidgeStatistics(feature_mean, feature_scale, text_mean, text_scale, target_mean,
                           gram, cross, tuple(train_ids), tuple(view_names), n * views)


def load_plan(path: Path) -> tuple[dict, Path, Path, list[dict]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "idea01_scaling_experiment_plan_v1":
        raise ValueError("unsupported scaling experiment plan")
    trials = plan.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("plan requires a nonempty trial list")
    seen = set()
    for trial in trials:
        name, arms = trial.get("name"), trial.get("arms")
        if (not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or "__" in name
                or name in seen or not isinstance(arms, list) or not arms
                or len(set(arms)) != len(arms)
                or any(not isinstance(arm, str) or not IDENTIFIER.fullmatch(arm)
                       or "__" in arm for arm in arms)):
            raise ValueError("trial names and arms must be unique lowercase identifiers without __")
        seen.add(name)
    return plan, Path(plan["split_manifest"]), Path(plan["text_cache"]), trials


def load_inputs(plan_path: Path) -> dict:
    plan, split_path, text_path, trials = load_plan(plan_path)
    split = read_split(split_path)
    root = Path(split["root"])
    cases = load_forecast_cases(root)
    ids = [case.id for case in cases]
    if set(ids) != set().union(*(set(values) for values in split["splits"].values())):
        raise ValueError("split must cover every public case")
    manifest_sha = sha256_file(root / "inputs_mcq.jsonl")
    if manifest_sha != split["inputs_manifest_sha256"]:
        raise ValueError("public manifest/split mismatch")
    text = torch.load(text_path, map_location="cpu", weights_only=True)
    if (text.get("schema") != "idea01_candidate_text_features_v1"
            or text.get("input_manifest_sha256") != manifest_sha
            or set(text["records"]) != set(ids)
            or text["protocol_sha256"] != canonical_sha256(text["protocol"])):
        raise ValueError("text cache protocol/public input mismatch")
    options = {}
    for case in cases:
        row = text["records"][case.id]
        embeddings = row["embeddings"]
        if (row["labels"] != [option.label for option in case.options]
                or row["text_sha256"] != case_text_sha256(case)
                or row["embedding_sha256"] != tensor_sha256(embeddings)
                or embeddings.shape != (6, 2048) or not embeddings.is_floating_point()
                or not bool(torch.isfinite(embeddings).all())):
            raise ValueError(f"candidate text/embedding checksum mismatch for {case.id}")
        options[case.id] = embeddings.double()
    views, weights, trial_protocols, trial_files = [], [], {}, {}
    features = {sample_id: [] for sample_id in ids}
    for trial in trials:
        name, arms = trial["name"], trial["arms"]
        directory = Path(trial["forecast_dir"])
        installed = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        protocol = installed["protocol"]
        if (installed["protocol_sha256"] != canonical_sha256(protocol)
                or protocol.get("split_sha256") != sha256_file(split_path)
                or protocol.get("splits") != split["splits"]
                or protocol.get("input_manifest_sha256") != manifest_sha
                or protocol.get("arms") != arms):
            raise ValueError(f"trial protocol/split mismatch: {name}")
        if "configuration" in trial and (
                protocol.get("configuration") != trial["configuration"]
                or protocol.get("trial_name") != name
                or protocol.get("plan_sha256") != canonical_sha256(plan)):
            raise ValueError(f"trial configuration or installed plan mismatch: {name}")
        if "predictor_sha256" in plan and protocol.get("predictor_sha256") != plan["predictor_sha256"]:
            raise ValueError(f"trial recent predictor mismatch: {name}")
        trial_protocols[name] = installed
        trial_files[name] = {}
        for sample_id in ids:
            path = directory / f"{sample_id}_forecasts.pt"
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            if (artifact.get("schema") != "idea01_scaling_case_forecasts_v1"
                    or artifact.get("sample_id") != sample_id
                    or artifact.get("protocol_sha256") != installed["protocol_sha256"]
                    or set(artifact.get("last_forecasts", {})) != set(arms)
                    or set(artifact.get("stats", {})) != set(arms)):
                raise ValueError(f"trial forecast identity mismatch: {name}/{sample_id}")
            if "metadata_sha256" in artifact:
                metadata = {key: value for key, value in artifact.items()
                            if key not in ("last_forecasts", "metadata_sha256")}
                if artifact["metadata_sha256"] != canonical_sha256(metadata):
                    raise ValueError(f"trial forecast metadata checksum mismatch: {name}/{sample_id}")
            for arm in arms:
                value = artifact["last_forecasts"][arm]
                if (not isinstance(value, Tensor) or value.shape != (4, 8, 256)
                        or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                        or ("forecast_sha256" in artifact
                            and artifact["forecast_sha256"].get(arm) != tensor_sha256(value))):
                    raise ValueError(f"trial forecast tensor invalid: {name}/{sample_id}/{arm}")
            features[sample_id].append(pool_forecasts(torch.stack([
                artifact["last_forecasts"][arm] for arm in arms])))
            trial_files[name][sample_id] = sha256_file(path)
        views += [f"{name}__{arm}" for arm in arms]
        weights += [1.0 / (len(trials) * len(arms))] * len(arms)
    return {
        "plan": plan, "split": split, "root": root, "ids": ids,
        "split_path": split_path, "text_path": text_path, "text": text,
        "features": {sample_id: torch.cat(parts) for sample_id, parts in features.items()},
        "options": options, "views": views, "weights": torch.tensor(weights, dtype=torch.float64),
        "trial_protocols": trial_protocols, "trial_files": trial_files,
    }


def choose_regularization(records: list[dict]) -> float:
    if not records:
        raise ValueError("empty regularization selection")
    return float(max(records, key=lambda row: (round(row["weighted_validation_accuracy"], 12),
                                               row["ridge_lambda"]))["ridge_lambda"])


def predict(plan_path: Path, output: Path, threads: int) -> int:
    if output.exists():
        raise ValueError("predict stage requires a new output directory")
    started = time.perf_counter()
    inputs = load_inputs(plan_path)
    split, ids, views = inputs["split"], inputs["ids"], inputs["views"]
    if output.resolve().is_relative_to(inputs["root"].resolve()):
        raise ValueError("output must be outside benchmark root")
    train_labels = load_fit_labels(inputs["root"] / "answers.jsonl", split, "train")
    val_labels = load_fit_labels(inputs["root"] / "answers.jsonl", split, "validation")

    def batch(selected: list[str]) -> tuple[Tensor, Tensor]:
        return (torch.stack([inputs["features"][sample_id] for sample_id in selected]),
                torch.stack([inputs["options"][sample_id] for sample_id in selected]))

    def targets(selected: list[str], labels: dict) -> Tensor:
        return torch.tensor([inputs["text"]["records"][sid]["labels"].index(labels[sid])
                             for sid in selected], dtype=torch.long)

    val_ids = split["splits"]["validation"]
    val_x, val_options = batch(val_ids)
    val_y = targets(val_ids, val_labels)
    if ("readout_train_sizes" in inputs["plan"] and "head_training_cases" in inputs["plan"]
            and inputs["plan"]["readout_train_sizes"] != inputs["plan"]["head_training_cases"]):
        raise ValueError("conflicting readout training-size declarations")
    requested_sizes = inputs["plan"].get("readout_train_sizes", inputs["plan"].get(
        "head_training_cases", [32, 64, len(split["splits"]["train"])]))
    if (not isinstance(requested_sizes, list) or not requested_sizes
            or len(set(requested_sizes)) != len(requested_sizes)
            or any(type(size) is not int for size in requested_sizes)):
        raise ValueError("readout_train_sizes must be distinct integers")
    full_count = len(split["splits"]["train"])
    if full_count not in requested_sizes:
        raise ValueError("full train size must be included as primary shared readout")
    models, selections, selected_ids = {}, {}, {}
    for count in sorted(requested_sizes):
        selected = nested_training_ids(split, count)
        train_x, train_options = batch(selected)
        stats = prepare_ridge_statistics(
            train_x, train_options, targets(selected, train_labels),
            view_weights=inputs["weights"], train_ids=selected, view_names=views,
        )
        records, fitted = [], {}
        for ridge_lambda in RIDGE_GRID:
            model = stats.solve(ridge_lambda)
            counts = (model.predict_scores(val_x, val_options).argmax(-1) == val_y[:, None]).sum(0)
            weighted_accuracy = float((counts.double() * inputs["weights"]).sum() / len(val_ids))
            records.append({"ridge_lambda": ridge_lambda,
                            "weighted_validation_accuracy": weighted_accuracy,
                            "independent_validation_cases": len(val_ids),
                            "correct_by_view": dict(zip(views, counts.tolist()))})
            fitted[ridge_lambda] = model
        selected_lambda = choose_regularization(records)
        head = f"shared_n{count}"
        models[head], selections[head], selected_ids[head] = fitted[selected_lambda], records, selected
        print(json.dumps({"head": head, "selected_lambda": selected_lambda,
                          "validation_weighted_accuracy": next(row["weighted_validation_accuracy"]
                            for row in records if row["ridge_lambda"] == selected_lambda)}), flush=True)
    baseline_path = inputs["plan"].get("baseline_readout")
    if baseline_path:
        baseline_path = Path(baseline_path)
        frozen = SharedRidgeReadout.from_dict(torch.load(baseline_path, map_location="cpu", weights_only=True))
        if (set(frozen.train_ids) != set(split["splits"]["train"])
                or frozen.coefficients.shape != (1024, 2048)):
            raise ValueError("frozen reference readout has different train IDs or dimensions")
        models["baseline_frozen"] = frozen
    primary = f"shared_n{full_count}"
    protocol = {
        "schema": "idea01_scaling_readout_protocol_v1", "plan_sha256": sha256_file(plan_path),
        "plan": inputs["plan"], "split_sha256": sha256_file(inputs["split_path"]),
        "input_manifest_sha256": split["inputs_manifest_sha256"], "splits": split["splits"],
        "text_cache_sha256": sha256_file(inputs["text_path"]),
        "text_protocol_sha256": inputs["text"]["protocol_sha256"],
        "trial_protocols": inputs["trial_protocols"], "trial_forecast_sha256": inputs["trial_files"],
        "views": views, "view_weights": inputs["weights"].tolist(), "primary_head": primary,
        "heads": list(models), "head_train_ids": selected_ids,
        "head_train_labels_sha256": {head: canonical_sha256({sid: train_labels[sid] for sid in selected})
                                     for head, selected in selected_ids.items()},
        "validation_labels_sha256": canonical_sha256(val_labels),
        "selected_lambdas": {head: model.ridge_lambda for head, model in models.items()},
        "lambda_grid": list(RIDGE_GRID), "baseline_readout_sha256": sha256_file(baseline_path) if baseline_path else None,
        "training_policy": "one shared map per train size; equal case, trial and within-trial arm weights",
        "selection_policy": "highest weighted validation accuracy; ties to larger lambda",
        "validation_refit": False, "test_labels_used": False,
        "test_status": "exploratory comparison; the fixed test was already inspected in the earlier experiment",
        "independent_supervised_train_cases": {head: len(values) for head, values in selected_ids.items()},
        "sources": {name: sha256_file(ROOT / name) for name in (
            "scripts/evaluate_idea1_scaling.py", "ttt_frame/idea1_mcq_readout.py")},
        "torch": str(torch.__version__), "threads": threads,
    }
    fingerprint = canonical_sha256(protocol)
    output.mkdir(parents=True)
    model_hashes = {}
    for head, model in models.items():
        path = output / f"{head}.pt"
        torch.save(model.to_dict(), path)
        model_hashes[head] = sha256_file(path)
    write_json(output / "protocol.json", {"protocol_fingerprint": fingerprint, "protocol": protocol,
                                          "readout_sha256": model_hashes})
    write_json(output / "validation_selection.json", selections)
    all_x, all_options = batch(ids)
    role_by_id = {sid: role for role in ROLES for sid in split["splits"][role]}
    completion_files, train_val_metrics = {}, {}
    for head, model in models.items():
        scores = model.predict_scores(all_x, all_options)
        blind = model.blind_scores(all_options)
        rows = []
        metrics = {role: {view: {"n": 0, "correct": 0} for view in views + ["text_prior"]}
                   for role in ("train", "validation")}
        for index, sample_id in enumerate(ids):
            labels = inputs["text"]["records"][sample_id]["labels"]
            role = role_by_id[sample_id]
            for view_index, view in enumerate(views + ["text_prior"]):
                values = scores[index, view_index] if view != "text_prior" else blind[index]
                if values.shape != (6,) or not bool(torch.isfinite(values).all()):
                    raise ValueError("nonfinite or invalid candidate scores")
                prediction = labels[int(values.argmax())]
                rows.append({
                    "schema": "forecast_mcq_prediction_v1", "id": sample_id, "arm": view,
                    "prediction": prediction, "raw_answer": prediction, "status": "ok",
                    "protocol_fingerprint": fingerprint, "elapsed_seconds": 0.0, "readout": head,
                    "candidate_scores": dict(zip(labels, values.tolist())), "research_split": role,
                    "tie_count": int((values == values.max()).sum()),
                })
                if role in metrics:
                    gold = train_labels[sample_id] if role == "train" else val_labels[sample_id]
                    metrics[role][view]["n"] += 1
                    metrics[role][view]["correct"] += int(prediction == gold)
        filename = "predictions.jsonl" if head == primary else f"{head}_predictions.jsonl"
        path = output / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(path)
        completion_files[head] = {"file": filename, "rows": len(rows), "sha256": sha256_file(path)}
        train_val_metrics[head] = metrics
    write_json(output / "train_validation_metrics.json", train_val_metrics)
    write_json(output / "completion.json", {
        "schema": "idea01_scaling_predictions_complete_v1", "complete": True,
        "protocol_fingerprint": fingerprint, "files": completion_files,
        "test_answers_loaded": False, "expected_rows_per_head": len(ids) * (len(views) + 1),
        "elapsed_seconds": time.perf_counter() - started,
    })
    print(f"Frozen {len(models)} shared/reference readouts; primary={primary}. Test labels not loaded.", flush=True)
    return 0


def score(plan_path: Path, output: Path) -> int:
    """Require complete frozen public predictions before accessing private answers."""
    plan, split_path, _, _ = load_plan(plan_path)
    split = read_split(split_path)
    root = Path(split["root"])
    ids = [case.id for case in load_forecast_cases(root)]
    completion = json.loads((output / "completion.json").read_text())
    installed = json.loads((output / "protocol.json").read_text())
    protocol = installed["protocol"]
    if (completion.get("complete") is not True or completion.get("test_answers_loaded") is not False
            or installed["protocol_fingerprint"] != canonical_sha256(protocol)
            or completion["protocol_fingerprint"] != installed["protocol_fingerprint"]
            or protocol["plan_sha256"] != sha256_file(plan_path)
            or protocol["split_sha256"] != sha256_file(split_path)
            or protocol["input_manifest_sha256"] != sha256_file(root / "inputs_mcq.jsonl")
            or protocol["splits"] != split["splits"]):
        raise ValueError("frozen protocol/plan/split mismatch")
    views = protocol["views"] + ["text_prior"]
    expected = {(sid, view) for sid in ids for view in views}
    if completion["expected_rows_per_head"] != len(expected):
        raise ValueError("unexpected frozen prediction row count")
    checked_paths = {}
    if set(completion["files"]) != set(protocol["heads"]):
        raise ValueError("readout completion/head mismatch")
    for head, item in completion["files"].items():
        if Path(item["file"]).name != item["file"]:
            raise ValueError("prediction path must be an output basename")
        path = output / item["file"]
        if (sha256_file(path) != item["sha256"] or item["rows"] != len(expected)
                or sha256_file(output / f"{head}.pt") != installed["readout_sha256"][head]):
            raise ValueError("prediction/model hash or row count mismatch")
        rows, fingerprint = load_predictions([path], ids, views)
        if set(rows) != expected or fingerprint != installed["protocol_fingerprint"]:
            raise ValueError("frozen predictions incomplete or protocol mismatch")
        for row in rows.values():
            values = row.get("candidate_scores", {})
            if (set(values) != set("ABCDEF") or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in values.values())):
                raise ValueError("six finite candidate scores required")
            best = max(values.values())
            if (type(row.get("tie_count")) is not int
                    or row["tie_count"] != sum(value == best for value in values.values())
                    or row["prediction"] not in values or values[row["prediction"]] != best
                    or row.get("readout") != head):
                raise ValueError("prediction does not match frozen readout scores")
        checked_paths[head] = path
    report_dir = output / "report"
    if report_dir.exists():
        raise ValueError("score report already exists")
    # Private labels are accessed only after every head passed all public checks.
    summaries, aggregates = {}, {}
    for head, path in checked_paths.items():
        report, exports = summarize(root, [path], views, split_path)
        controls = {}
        for trial in plan["trials"]:
            name, arms = trial["name"], trial["arms"]
            controls[name] = {}
            for arm in arms:
                control = ("residual_no_update" if "residual" in arm and "residual_no_update" in arms
                           else "plain_no_update" if arm == "plain_delayed" and "plain_no_update" in arms
                           else "recent_only")
                if control not in arms or arm == control:
                    continue
                a = report["by_arm"][f"{name}__{arm}"]["cases"]
                b = report["by_arm"][f"{name}__{control}"]["cases"]
                controls[name][arm] = {"control": control, "by_split": {
                    role: _paired_scores([(x, y) for x, y in zip(a, b)
                                          if x["research_split"] == role]) for role in ROLES}}
        report.update({"readout": head, "paired_within_trial": controls,
                       "test_status": protocol["test_status"], "primary_head": protocol["primary_head"],
                       "fit_train_ids": protocol["head_train_ids"].get(head, split["splits"]["train"]),
                       "train_metric_scope": "entire train pool; only fit_train_ids were used to fit this head"})
        summaries[head] = report
        aggregates[head] = {**report, "by_arm": {
            name: {key: value for key, value in row.items() if key != "cases"}
            for name, row in report["by_arm"].items()}}
    report_dir.mkdir()
    write_json(report_dir / "summary.json", summaries)
    write_json(report_dir / "aggregate_metrics.json", {
        "schema": "idea01_scaling_results_v1", "protocol_fingerprint": installed["protocol_fingerprint"],
        "primary_head": protocol["primary_head"], "test_status": protocol["test_status"],
        "by_head": aggregates,
    })
    primary = summaries[protocol["primary_head"]]
    lines = ["# Idea 01 scaling results", "", protocol["test_status"], "",
             f"Primary shared readout: {protocol['primary_head']}; test cases={len(split['splits']['test'])}.", "",
             "| Trial / arm | Train | Validation | Test |", "|---|---:|---:|---:|"]
    for view in views:
        row = primary["by_arm"][view]
        cells = [f"{row['by_research_split'][role]['correct']}/{row['by_research_split'][role]['n']} "
                 f"({row['by_research_split'][role]['accuracy_all']:.2%})" for role in ROLES]
        lines.append(f"| {view} | " + " | ".join(cells) + " |")
    lines += ["", "## Shared-head training-size comparison", "",
              "Training subsets are nested and day-stratified; visual predictor training remains fixed.", "",
              "| Readout | Trial | Residual test | Residual no-update test | Plain test |", "|---|---|---:|---:|---:|"]
    for head, report in summaries.items():
        for trial in plan["trials"]:
            cells = []
            for arm in ("residual", "residual_no_update", "plain_delayed"):
                row = report["by_arm"].get(f"{trial['name']}__{arm}")
                value = row["by_research_split"]["test"] if row else None
                cells.append(f"{value['correct']}/{value['n']} ({value['accuracy_all']:.2%})" if value else "n/a")
            lines.append(f"| {head} | {trial['name']} | " + " | ".join(cells) + " |")
    (report_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Scored {len(summaries)} readouts; results in {report_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("predict", "score"), default="predict")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    return predict(args.plan, args.output_dir, args.threads) if args.stage == "predict" else score(args.plan, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
