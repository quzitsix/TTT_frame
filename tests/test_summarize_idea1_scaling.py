"""A compact scaling archive stays bound to frozen artifacts and omits answers."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts import summarize_idea1_scaling as runner


def dump(path, value):
    runner.write_json(path, value)


@pytest.fixture
def scored_run(tmp_path):
    run = tmp_path / "run"
    directory = run / "forecasts/h128_random_u1"
    directory.mkdir(parents=True)
    split = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    arms = ["recent_only", "residual", "residual_no_update", "plain_delayed"]
    config = {"hidden_dim": 128, "inner_steps": 1, "pretraining_steps": 0}
    trial = {"name": "h128_random_u1", "configuration": config,
             "forecast_dir": str(directory), "arms": arms}
    plan = {"schema": "idea01_scaling_experiment_plan_v1", "trials": [trial],
            "unique_labeled_training_cases_increase": False}
    dump(run / "plan.json", plan)
    trial_protocol = {"configuration": config, "splits": split}
    installed_trial = {"protocol": trial_protocol,
                       "protocol_sha256": runner.canonical_sha256(trial_protocol)}
    dump(directory / "protocol.json", installed_trial)
    hashes = {}
    for sample_id in "abcd":
        stats = {arm: {"mature_count": 4, "mature_mse": 0.5, "updates": 12,
                       "gradient_steps": 12, "state_bytes": 1048576,
                       "pending_bytes": 100, "fast_weight_delta_l2": 2,
                       "mature_mse_by_horizon": {"1": 0.4, "2": 0.6}} for arm in arms}
        artifact = {"sample_id": sample_id, "trial": trial["name"], "stats": stats,
                    "protocol_sha256": installed_trial["protocol_sha256"],
                    "result_metadata": {"horizons": [1, 2, 4, 10], "feature_shape": [8, 256]},
                    "batch_cases": 2, "batch_seconds": 4}
        artifact["metadata_sha256"] = runner.canonical_sha256(artifact)
        path = directory / f"{sample_id}_forecasts.pt"
        torch.save(artifact, path)
        hashes[sample_id] = runner.sha256_file(path)
    dump(directory / "summary.json", {"complete": True, "elapsed_seconds": 10,
         "protocol_sha256": installed_trial["protocol_sha256"],
         "cases": [{"sample_id": sid, "artifact_sha256": sha} for sid, sha in hashes.items()]})
    views = [f"{trial['name']}__{arm}" for arm in arms]
    protocol = {"plan_sha256": runner.sha256_file(run / "plan.json"), "plan": plan,
                "splits": split, "views": views, "heads": ["shared_n2", "baseline_frozen"],
                "primary_head": "shared_n2", "head_train_ids": {"shared_n2": ["a", "b"]},
                "selected_lambdas": {"shared_n2": 100, "baseline_frozen": 10},
                "trial_protocols": {trial["name"]: installed_trial},
                "trial_forecast_sha256": {trial["name"]: hashes}}
    fingerprint = runner.canonical_sha256(protocol)
    dump(run / "mcq/protocol.json", {"protocol": protocol, "protocol_fingerprint": fingerprint})
    dump(run / "mcq/validation_selection.json", {})
    by_head, files = {}, {}
    for head in protocol["heads"]:
        rows = [{"id": sid, "arm": view, "prediction": "A" if "__residual" in view else "B",
                 "research_split": role, "readout": head, "status": "ok",
                 "candidate_scores": {"A": 10}, "raw_answer": "should not be archived"}
                for role in runner.ROLES for sid in split[role] for view in views + ["text_prior"]]
        path = run / "mcq" / f"{head}_predictions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        files[head] = {"file": path.name, "sha256": runner.sha256_file(path), "rows": len(rows)}
        metrics = {role: {"n": len(ids), "correct": len(ids), "accuracy_all": 1}
                   for role, ids in split.items()}
        by_head[head] = {"by_arm": {view: {"by_research_split": metrics,
                                          "cases": [{"correct_answer": "DO_NOT_EXPORT"}]}
                                    for view in views + ["text_prior"]},
                         "paired_within_trial": {trial["name"]: {"residual": {"control": "residual_no_update"}}}}
    dump(run / "mcq/completion.json", {"complete": True, "test_answers_loaded": False,
         "protocol_fingerprint": fingerprint, "files": files, "elapsed_seconds": 3})
    dump(run / "mcq/report/aggregate_metrics.json", {"schema": "idea01_scaling_results_v1",
         "protocol_fingerprint": fingerprint, "primary_head": "shared_n2",
         "test_status": "exploratory reuse", "by_head": by_head})
    dump(run / "pretraining_audit.json", {"by_width": {}})
    return run, tmp_path / "archive"


def test_archive_without_benchmark_access_is_complete_compact_and_checksummed(scored_run):
    run, output = scored_run
    result = runner.archive(run, output)
    assert result["answer_keys_read_by_archiver"] is False
    trial = result["by_trial"]["h128_random_u1"]
    assert trial["parameter_values_per_target"] == 262144
    assert trial["forecast_batch_compute_seconds_summed"] == 8
    assert trial["forecast_seconds_per_case_amortized"] == 2
    differences = result["by_head"]["shared_n2"]["by_trial"]["h128_random_u1"]["answer_disagreements"]
    assert differences["residual_vs_recent_only"]["test"] == {"n": 1, "different_answers": 1}
    rows = [json.loads(line) for line in (output / "shared_n2_predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and len(rows[0]["predictions"]) == 5
    text = (output / "aggregate_metrics.json").read_text()
    assert "DO_NOT_EXPORT" not in text and "correct_answer" not in text
    assert "candidate_scores" not in (output / "shared_n2_predictions.jsonl").read_text()
    manifest = runner.read_json(output / "manifest.json")
    archived = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    assert set(manifest["files"]) == archived - {"manifest.json"}
    assert not list(output.rglob("*.pt"))
    for relative, expected in manifest["files"].items():
        path = output / relative
        assert runner.sha256_file(path) == expected["sha256"]
        assert path.stat().st_size == expected["bytes"]


@pytest.mark.parametrize("change", ["predictions", "forecast", "completion", "aggregate_protocol"])
def test_changed_frozen_inputs_are_rejected_before_archive_creation(scored_run, change):
    run, output = scored_run
    if change == "predictions":
        path = run / "mcq/shared_n2_predictions.jsonl"
        path.write_text(path.read_text() + "\n")
    elif change == "forecast":
        path = run / "forecasts/h128_random_u1/a_forecasts.pt"
        artifact = torch.load(path, weights_only=True)
        artifact["stats"]["residual"]["mature_mse"] = 0.1
        torch.save(artifact, path)
    elif change == "completion":
        path = run / "mcq/completion.json"
        record = runner.read_json(path)
        record["complete"] = False
        dump(path, record)
    else:
        path = run / "mcq/report/aggregate_metrics.json"
        record = runner.read_json(path)
        record["protocol_fingerprint"] = "different"
        dump(path, record)
    with pytest.raises(ValueError):
        runner.archive(run, output)
    assert not output.exists()


def test_pair_weighted_mse_and_diagnostic_bounds():
    records = [{"mature_count": 1, "mature_mse": 2}, {"mature_count": 3, "mature_mse": 4}]
    result = runner.summarize_stats(records)
    assert result["mature_mse_pair_weighted"] == 3.5
    assert result["mature_mse"]["mean"] == 3
    damaged = deepcopy(records)
    damaged[0]["mature_count"] = 0
    with pytest.raises(ValueError, match="positive mature counts"):
        runner.summarize_stats(damaged)


def test_duplicate_predictions_rejected_even_with_updated_file_checksum(scored_run):
    run, output = scored_run
    path = run / "mcq/shared_n2_predictions.jsonl"
    rows = path.read_text().splitlines()
    rows[-1] = rows[0]
    path.write_text("\n".join(rows) + "\n")
    completion_path = run / "mcq/completion.json"
    completion = runner.read_json(completion_path)
    completion["files"]["shared_n2"]["sha256"] = runner.sha256_file(path)
    dump(completion_path, completion)
    with pytest.raises(ValueError, match="duplicate"):
        runner.archive(run, output)
    assert not output.exists()


def add_pretraining_summary(run):
    protocol = {"config": {"hidden_dim": 128}, "horizons": [1, 2, 4, 10],
                "train_ids": ["a", "b"], "test_ids_or_labels_used": False,
                "mcq_answers_used": False, "future_media_used": False, "meta_outer_training": False}
    fingerprint = runner.canonical_sha256(protocol)
    snapshots = {"0": {"hidden_dim": 128, "width_output_gain": 1.0,
                        "steps": 0, "protocol_sha256": fingerprint,
                        "train_ids": ["a", "b"], "horizons": [1, 2, 4, 10],
                        "validation": {"plain": {"mse": 1.0}, "residual": {"mse": 0.5}},
                        "test_ids_or_labels_used": False}}
    dump(run / "pretraining/h128/protocol.json",
         {"protocol": protocol, "protocol_sha256": fingerprint})
    dump(run / "pretraining/h128/summary.json", {
        "schema": "idea01_offline_initialization_summary_v1", "complete": True,
        "protocol_sha256": fingerprint, "elapsed_seconds": 2.0,
        "training_seconds": 1.0, "snapshots": snapshots})


def test_fresh_run_without_handwritten_pretraining_audit_can_be_archived(scored_run):
    run, output = scored_run
    (run / "pretraining_audit.json").unlink()
    add_pretraining_summary(run)
    summary = runner.archive(run, output)
    audit = runner.read_json(output / "pretraining_audit.json")
    assert audit == summary["pretraining"]
    assert audit["derived_from_saved_summaries"] is True
    assert audit["independent_audit_available"] is False
    assert audit["by_width"]["128"]["parameter_values_per_target"] == 262144
    assert audit["by_width"]["128"]["persistent_state_bytes_per_target"] == 1048576
    assert audit["by_width"]["128"]["width_output_gain"] == 1
    assert audit["by_width"]["128"]["snapshots"]["0"]["validation"]["plain"]["mse"] == 1
    manifest = runner.read_json(output / "manifest.json")
    assert manifest["files"]["pretraining_audit.json"]["sha256"] == runner.sha256_file(output / "pretraining_audit.json")


def test_optional_integrity_audit_is_copied_and_checksummed(scored_run):
    run, output = scored_run
    integrity = {"schema": "test_integrity_audit", "complete": True, "checks": ["frozen"]}
    dump(run / "integrity_audit.json", integrity)
    runner.archive(run, output)
    assert runner.read_json(output / "integrity_audit.json") == integrity
    manifest = runner.read_json(output / "manifest.json")
    assert manifest["files"]["integrity_audit.json"]["sha256"] == runner.sha256_file(run / "integrity_audit.json")


def test_derived_pretraining_rejects_changed_protocol_before_creating_archive(scored_run):
    run, output = scored_run
    (run / "pretraining_audit.json").unlink()
    add_pretraining_summary(run)
    path = run / "pretraining/h128/summary.json"
    value = runner.read_json(path)
    value["protocol_sha256"] = "changed"
    dump(path, value)
    with pytest.raises(ValueError, match="invalid pretraining"):
        runner.archive(run, output)
    assert not output.exists()
