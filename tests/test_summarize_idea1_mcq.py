"""Integrity tests for the post-hoc Idea 01 scorer.

The fixtures are deliberately tiny and synthetic. In particular, tests that
exercise preflight failure remove the private answer file so a scorer that
reads it too early fails the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_idea1_mcq import digest_json
from scripts.summarize_idea1_mcq import ARMS, file_sha, main
from scripts.summarize_forecast_mcq import DEFAULT_ARMS


ALL_ARMS = tuple(ARMS) + ("text_prior",)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path, *, complete: bool = True) -> dict:
    root = tmp_path / "bench"
    run = tmp_path / "run"
    archive = tmp_path / "baseline"
    root.mkdir()
    run.mkdir()
    archive.mkdir()
    ids = ["egolife_001", "egolife_002", "egolife_003"]
    days = [1, 2, 3]
    public, metadata, answers = [], [], []
    for index, (sample_id, day) in enumerate(zip(ids, days)):
        sample = root / "samples" / sample_id
        sample.mkdir(parents=True)
        (sample / "context.mp4").touch()
        (sample / "input.mp4").touch()
        options = [{"label": label, "text": f"action {label}"} for label in "ABCDEF"]
        public.append({
            "id": sample_id, "task": "egocentric_action_forecasting_5s",
            "input": {"context_video": f"samples/{sample_id}/context.mp4",
                      "video": f"samples/{sample_id}/input.mp4",
                      "question": "What happens next?", "context_duration_seconds": 1800,
                      "input_duration_seconds": 30, "prediction_horizon_seconds": 5,
                      "options": options},
        })
        output = {"correct_answer": "ABC"[index], "options": options,
                  "option_type_counts": {"plausible_not_observed": index,
                                         "implausible_not_observed": 6 - index}}
        answers.append({"id": sample_id, **output})
        metadata.append({"id": sample_id, "output": output,
                         "metadata": {"participant": "wearer_a", "day": day,
                                      "context_start_seconds": 0, "context_end_seconds": 1800,
                                      "input_start_seconds": 1800, "cutoff_seconds": 1830,
                                      "future_end_seconds": 1835},
                         "verification": {"gt_consistency": "supported", "human_verified": False}})
    _write_jsonl(root / "inputs_mcq.jsonl", public)
    _write_jsonl(root / "metadata.jsonl", metadata)
    answers_path = root / "answers.jsonl"
    _write_jsonl(answers_path, answers)

    split = {
        "schema": "forecast_bench_research_day_split_v1", "official_split": False,
        "grouping": "recording_day_all_participants", "inputs_manifest_sha256": file_sha(root / "inputs_mcq.jsonl"),
        "splits": {"train": [ids[0]], "validation": [ids[1]], "test": [ids[2]]},
        "days": {"train": [1], "validation": [2], "test": [3]},
        "id_to_day": dict(zip(ids, days)), "counts": {"train": 1, "validation": 1, "test": 1},
    }
    split_path = tmp_path / "research_day_split.json"
    _write_json(split_path, split)

    protocol = {"schema": "idea01_synthetic_protocol_v1",
                "split_sha256": file_sha(split_path),
                "input_manifest_sha256": file_sha(root / "inputs_mcq.jsonl"),
                "splits": split["splits"], "arms": list(ALL_ARMS)}
    protocol_fingerprint = digest_json(protocol)
    (run / "shared_readout.pt").write_bytes(b"synthetic readout")

    prediction_rows = []
    for sample_id, gold in zip(ids, "ABC"):
        for arm_index, arm in enumerate(ALL_ARMS):
            # Give each arm finite, explicit six-way scores. The selected
            # answer is intentionally known without requiring scorer internals.
            choice = "ABCDEF"[(ord(gold) - ord("A") + arm_index) % 6]
            scores = {label: (1.0 if label == choice else 0.0) for label in "ABCDEF"}
            prediction_rows.append({"schema": "forecast_mcq_prediction_v1", "id": sample_id,
                                    "arm": arm, "prediction": choice, "raw_answer": choice,
                                    "status": "ok", "elapsed_seconds": 0.0,
                                    "protocol_fingerprint": protocol_fingerprint,
                                    "candidate_scores": scores, "research_split": next(
                                        role for role, members in split["splits"].items() if sample_id in members),
                                    "tie_count": 1})
    if not complete:
        prediction_rows.pop()
    prediction_path = run / "predictions.jsonl"
    _write_jsonl(prediction_path, prediction_rows)
    completion = {"complete": True, "rows": len(prediction_rows),
                  "expected_rows": len(ids) * len(ALL_ARMS),
                  "protocol_fingerprint": protocol_fingerprint,
                  "predictions_sha256": file_sha(prediction_path), "test_answers_loaded": False}
    run_record = {"protocol_fingerprint": protocol_fingerprint, "protocol": protocol,
                  "shared_readout_sha256": file_sha(run / "shared_readout.pt")}
    _write_json(run / "completion.json", completion)
    _write_json(run / "protocol.json", run_record)

    # The baseline archive is a frozen, answer-free prediction export plus
    # aggregate test metrics. Its IDs intentionally match the public set.
    baseline_rows = [{"id": sample_id, "prediction": "A"} for sample_id in ids]
    for baseline_name in DEFAULT_ARMS:
        _write_jsonl(archive / f"{baseline_name}_predictions.jsonl", baseline_rows)
    test_correct = 0  # egolife_003 is the held-out case; its gold label is C.
    archive_metrics = {"inputs_sha256": protocol["input_manifest_sha256"],
                       "by_arm": {baseline_name: {"by_research_split": {
                           "test": {"n": 1, "correct": test_correct,
                                     "accuracy_all": float(test_correct)}}}
                                  for baseline_name in DEFAULT_ARMS}}
    _write_json(archive / "aggregate_metrics.json", archive_metrics)
    _write_json(archive / "research_day_split.json", split)
    return {"root": root, "run": run, "archive": archive, "split": split_path,
            "answers": answers_path, "ids": ids, "protocol": protocol,
            "protocol_fingerprint": protocol_fingerprint, "prediction": prediction_path}


def _args(fixture: dict, output: Path) -> list[str]:
    return ["--root", str(fixture["root"]), "--run-dir", str(fixture["run"]),
            "--split-manifest", str(fixture["split"]), "--baseline-archive", str(fixture["archive"]),
            "--output-dir", str(output)]


def _forbid_private_reads(monkeypatch):
    original = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name in ("answers.jsonl", "metadata.jsonl"):
            raise AssertionError(f"private data opened before integrity validation: {path.name}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)


def _refresh_prediction_hash(fixture):
    path = fixture["run"] / "completion.json"
    completion = json.loads(path.read_text())
    completion["predictions_sha256"] = file_sha(fixture["prediction"])
    _write_json(path, completion)


def test_complete_synthetic_run_reports_all_arms_and_held_out_split(tmp_path):
    fixture = _fixture(tmp_path)
    output = tmp_path / "report"
    assert main(_args(fixture, output)) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema"] == "idea01_mcq_results_v1"
    assert summary["primary_split"] == "test"
    assert set(summary["by_arm"]) == set(ALL_ARMS)
    assert summary["sample_count"] == 3
    assert summary["by_arm"]["residual"]["by_research_split"]["test"]["n"] == 1
    assert set(summary["vlm_reference_results"]) == set(DEFAULT_ARMS)
    for arm in ALL_ARMS:
        assert set(summary["by_arm"][arm]["by_research_split"]) == {"train", "validation", "test"}
        assert len(summary["paired_controls"][arm]) == len(ALL_ARMS) - 1
        assert summary["readout_score_diagnostics"][arm]["test_cases"] == 1
        rows = [json.loads(line) for line in (output / f"{arm}_predictions.jsonl").read_text().splitlines()]
        assert [row["id"] for row in rows] == fixture["ids"]
        assert all(set(row) == {"id", "prediction"} for row in rows)
    paired = summary["paired_controls"]["persistence"]["recent_only"]["test"]
    assert paired["arm_only_correct"] == 1 and paired["accuracy_delta_all"] == 1.0
    aggregate_text = (output / "aggregate_metrics.json").read_text()
    assert '"correct_answer"' not in aggregate_text and '"cases"' not in aggregate_text
    assert summary["full_200_accuracy_is_held_out"] is False
    assert all(record["matched_input_budget_or_training"] is False
               for record in summary["vlm_reference_results"].values())
    assert (output / "results.md").is_file()


def test_incomplete_predictions_are_rejected_before_private_answers_are_opened(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, complete=False)
    _forbid_private_reads(monkeypatch)
    with pytest.raises(ValueError, match="incomplete|prediction|rows|complete"):
        main(_args(fixture, tmp_path / "report"))


def test_prediction_protocol_mismatch_is_rejected_before_private_answers(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["prediction"].read_text().splitlines()]
    rows[0]["protocol_fingerprint"] = "different"
    _write_jsonl(fixture["prediction"], rows)
    completion = json.loads((fixture["run"] / "completion.json").read_text())
    completion["predictions_sha256"] = file_sha(fixture["prediction"])
    _write_json(fixture["run"] / "completion.json", completion)
    _forbid_private_reads(monkeypatch)
    with pytest.raises(ValueError, match="protocol|fingerprint|prediction"):
        main(_args(fixture, tmp_path / "report"))


def test_baseline_prediction_ids_must_match_public_ids(tmp_path):
    fixture = _fixture(tmp_path)
    path = fixture["archive"] / "qwen_blind_predictions.jsonl"
    rows = [{"id": "unknown", "prediction": "A"},
            {"id": fixture["ids"][1], "prediction": "A"},
            {"id": fixture["ids"][2], "prediction": "A"}]
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="ID|complete|reference|prediction"):
        main(_args(fixture, tmp_path / "report"))


def test_baseline_archived_test_metrics_must_match_frozen_exports(tmp_path):
    fixture = _fixture(tmp_path)
    metrics_path = fixture["archive"] / "aggregate_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["by_arm"]["qwen_blind"]["by_research_split"]["test"]["correct"] = 1
    metrics["by_arm"]["qwen_blind"]["by_research_split"]["test"]["accuracy_all"] = 1.0
    _write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="metric|accuracy|baseline|disagree|consistent"):
        main(_args(fixture, tmp_path / "report"))


def test_public_input_hash_change_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    with (fixture["root"] / "inputs_mcq.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="mismatch|hash|protocol|input"):
        main(_args(fixture, tmp_path / "report"))


@pytest.mark.parametrize("problem", [
    "duplicate", "unknown_id", "unknown_arm", "bad_schema", "nonfinite_score",
    "boolean_score", "missing_score", "wrong_maximum", "wrong_tie_count",
])
def test_malformed_predictions_never_reach_private_labels(tmp_path, monkeypatch, problem):
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["prediction"].read_text().splitlines()]
    if problem == "duplicate":
        rows[1] = rows[0].copy()
    elif problem == "unknown_id":
        rows[0]["id"] = "egolife_999"
    elif problem == "unknown_arm":
        rows[0]["arm"] = "unexpected_arm"
    elif problem == "bad_schema":
        rows[0]["schema"] = "other"
    elif problem == "nonfinite_score":
        rows[0]["candidate_scores"]["A"] = float("nan")
    elif problem == "boolean_score":
        rows[0]["candidate_scores"]["A"] = True
    elif problem == "missing_score":
        del rows[0]["candidate_scores"]["F"]
    elif problem == "wrong_maximum":
        rows[0]["prediction"] = "F"
    else:
        rows[0]["tie_count"] = 2
    _write_jsonl(fixture["prediction"], rows)
    _refresh_prediction_hash(fixture)
    _forbid_private_reads(monkeypatch)
    with pytest.raises(ValueError):
        main(_args(fixture, tmp_path / "report"))


@pytest.mark.parametrize("problem", [
    "incomplete_marker", "wrong_expected_rows", "wrong_actual_rows", "test_labels_used",
    "prediction_hash", "model_hash", "protocol_fingerprint", "completion_protocol",
])
def test_completion_and_hash_failures_precede_private_label_access(tmp_path, monkeypatch, problem):
    fixture = _fixture(tmp_path)
    completion_path = fixture["run"] / "completion.json"
    protocol_path = fixture["run"] / "protocol.json"
    completion, protocol = json.loads(completion_path.read_text()), json.loads(protocol_path.read_text())
    if problem == "incomplete_marker":
        completion["complete"] = False
    elif problem == "wrong_expected_rows":
        completion["expected_rows"] -= 1
    elif problem == "wrong_actual_rows":
        completion["rows"] -= 1
    elif problem == "test_labels_used":
        completion["test_answers_loaded"] = True
    elif problem == "prediction_hash":
        completion["predictions_sha256"] = "wrong"
    elif problem == "model_hash":
        (fixture["run"] / "shared_readout.pt").write_bytes(b"changed model")
    elif problem == "protocol_fingerprint":
        protocol["protocol_fingerprint"] = "wrong"
    else:
        completion["protocol_fingerprint"] = "wrong"
    _write_json(completion_path, completion)
    _write_json(protocol_path, protocol)
    _forbid_private_reads(monkeypatch)
    with pytest.raises(ValueError):
        main(_args(fixture, tmp_path / "report"))


@pytest.mark.parametrize("problem", ["public_hash", "split_membership", "duplicate_id", "nonfinite_accuracy"])
def test_baseline_archive_identity_mismatch_is_rejected(tmp_path, problem):
    fixture = _fixture(tmp_path)
    if problem == "public_hash":
        path = fixture["archive"] / "aggregate_metrics.json"
        value = json.loads(path.read_text())
        value["inputs_sha256"] = "wrong"
        _write_json(path, value)
    elif problem == "nonfinite_accuracy":
        path = fixture["archive"] / "aggregate_metrics.json"
        value = json.loads(path.read_text())
        value["by_arm"]["qwen_blind"]["by_research_split"]["test"]["accuracy_all"] = float("nan")
        _write_json(path, value)
    elif problem == "split_membership":
        path = fixture["archive"] / "research_day_split.json"
        value = json.loads(path.read_text())
        value["splits"]["train"], value["splits"]["test"] = value["splits"]["test"], value["splits"]["train"]
        _write_json(path, value)
    else:
        path = fixture["archive"] / "qwen_blind_predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0] = rows[1].copy()
        _write_jsonl(path, rows)
    with pytest.raises(ValueError):
        main(_args(fixture, tmp_path / "report"))
