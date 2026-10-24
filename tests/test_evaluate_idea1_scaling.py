"""Shared scaling readout: numerical parity, data isolation and frozen scoring."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from scripts import evaluate_idea1_scaling as runner
from scripts.cache_idea1_text import canonical_sha256, case_text_sha256, tensor_sha256
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.idea1_mcq_readout import SharedRidgeReadout


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def test_primal_fit_matches_original_dual_and_is_candidate_permutation_equivariant():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(5, 3, 7, generator=generator, dtype=torch.float64)
    choices = torch.randn(5, 6, 11, generator=generator, dtype=torch.float64)
    gold = torch.tensor([2, 0, 5, 1, 3])
    ids, views = [f"case_{i}" for i in range(5)], ["a", "b", "c"]
    stats = runner.prepare_ridge_statistics(x, choices, gold, train_ids=ids, view_names=views)
    actual = stats.solve(0.1)
    expected = SharedRidgeReadout.fit(x, choices, gold, ridge_lambda=0.1,
                                      train_ids=ids, arm_names=views)
    torch.testing.assert_close(actual.coefficients, expected.coefficients, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(actual.predict_scores(x, choices), expected.predict_scores(x, choices),
                               atol=1e-12, rtol=1e-11)
    permutation = torch.tensor([4, 1, 5, 2, 0, 3])
    torch.testing.assert_close(actual.predict_scores(x, choices[:, permutation]),
                               actual.predict_scores(x, choices)[:, :, permutation])


def test_view_weighting_is_invariant_to_duplicating_a_view_and_splitting_its_weight():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(5, 2, 7, generator=generator)
    options = torch.randn(5, 6, 11, generator=generator)
    gold = torch.arange(5)
    first = runner.prepare_ridge_statistics(
        x, options, gold, view_weights=torch.tensor([0.7, 0.3]),
        train_ids=list("abcde"), view_names=["one", "two"]).solve(1)
    repeated = runner.prepare_ridge_statistics(
        x[:, [0, 0, 1]], options, gold, view_weights=torch.tensor([0.35, 0.35, 0.3]),
        train_ids=list("abcde"), view_names=["one_a", "one_b", "two"]).solve(1)
    torch.testing.assert_close(first.coefficients, repeated.coefficients, atol=1e-8, rtol=1e-7)


def test_nested_training_order_is_deterministic_label_free_and_day_stratified():
    split = {"splits": {"train": [f"id_{i}" for i in range(12)]},
             "id_to_day": {f"id_{i}": 1 + i % 3 for i in range(12)}}
    short = runner.nested_training_ids(split, 6)
    large = runner.nested_training_ids(split, 9)
    assert short == large[:6]
    assert len({split["id_to_day"][sid] for sid in short[:3]}) == 3
    changed = deepcopy(split)
    changed["splits"]["train"].reverse()
    assert runner.nested_training_ids(changed, 6) == short
    assert {split["id_to_day"][sid] for sid in short} == {1, 2, 3}
    with pytest.raises(ValueError):
        runner.nested_training_ids(split, 1)


def test_regularization_selection_is_one_weighted_validation_objective():
    assert runner.choose_regularization([
        {"ridge_lambda": 0.01, "weighted_validation_accuracy": 0.25},
        {"ridge_lambda": 100, "weighted_validation_accuracy": 0.25},
        {"ridge_lambda": 10, "weighted_validation_accuracy": 0.1},
    ]) == 100


def test_declared_trial_configuration_cannot_be_replaced_by_another_width(experiment):
    plan = deepcopy(experiment["plan"])
    trial = plan["trials"][0]
    trial["configuration"] = {"hidden_dim": 128}
    write_json(experiment["plan_path"], plan)
    path = runner.Path(trial["forecast_dir"]) / "protocol.json"
    installed = json.loads(path.read_text())
    installed["protocol"].update(configuration={"hidden_dim": 2048}, trial_name=trial["name"],
                                  plan_sha256=canonical_sha256(plan))
    installed["protocol_sha256"] = canonical_sha256(installed["protocol"])
    write_json(path, installed)
    with pytest.raises(ValueError, match="configuration or installed plan"):
        runner.load_inputs(experiment["plan_path"])


def test_authoritative_plan_head_sizes_are_used(experiment):
    plan = deepcopy(experiment["plan"])
    expected = plan.pop("readout_train_sizes")
    plan["head_training_cases"] = expected
    write_json(experiment["plan_path"], plan)
    runner.main(["--plan", str(experiment["plan_path"]), "--output-dir", str(experiment["output"]),
                 "--threads", "1"])
    completion = json.loads((experiment["output"] / "completion.json").read_text())
    assert set(completion["files"]) == {f"shared_n{size}" for size in expected}


def test_conflicting_readout_size_declarations_are_rejected(experiment):
    plan = deepcopy(experiment["plan"])
    plan["head_training_cases"] = [3]
    write_json(experiment["plan_path"], plan)
    with pytest.raises(ValueError, match="conflicting readout"):
        runner.main(["--plan", str(experiment["plan_path"]), "--output-dir", str(experiment["output"]),
                     "--threads", "1"])


@pytest.fixture
def experiment(tmp_path):
    generator = torch.Generator().manual_seed(8)
    root = tmp_path / "bench"
    root.mkdir()
    ids = [f"egolife_{i:03d}" for i in range(1, 6)]
    days = [1, 2, 1, 5, 6]
    public, metadata, answers = [], [], []
    for index, (sample_id, day) in enumerate(zip(ids, days)):
        sample = root / "samples" / sample_id
        sample.mkdir(parents=True)
        (sample / "context.mp4").touch()
        (sample / "input.mp4").touch()
        options = [{"label": label, "text": f"action {label}"} for label in "ABCDEF"]
        public.append({"id": sample_id, "task": "egocentric_action_forecasting_5s", "input": {
            "context_video": f"samples/{sample_id}/context.mp4",
            "video": f"samples/{sample_id}/input.mp4", "question": "What happens next?",
            "context_duration_seconds": 1800, "input_duration_seconds": 30,
            "prediction_horizon_seconds": 5, "options": options,
        }})
        output = {"correct_answer": "ABCDE"[index], "options": options,
                  "option_type_counts": {"plausible_not_observed": 3, "implausible_not_observed": 2}}
        answers.append({"id": sample_id, **output})
        metadata.append({"id": sample_id, "output": output,
                         "metadata": {"participant": "wearer_a", "day": day,
                                      "context_start_seconds": 1000, "context_end_seconds": 2800,
                                      "input_start_seconds": 2800, "cutoff_seconds": 2830,
                                      "future_end_seconds": 2835},
                         "verification": {"gt_consistency": "supported", "human_verified": False}})
    manifest = root / "inputs_mcq.jsonl"
    write_jsonl(manifest, public)
    manifest_sha = runner.sha256_file(manifest)
    split = {"schema": "forecast_bench_research_day_split_v1", "official_split": False,
             "root": str(root), "grouping": "recording_day_all_participants",
             "inputs_manifest_sha256": manifest_sha, "id_to_day": dict(zip(ids, days)),
             "days": {"train": [1, 2], "validation": [5], "test": [6]},
             "splits": {"train": ids[:3], "validation": ids[3:4], "test": ids[4:]}}
    split_path = tmp_path / "split.json"
    write_json(split_path, split)
    # The predict stage has no test answer row or metadata file.
    write_jsonl(root / "answers.jsonl", answers[:-1])
    text = {"schema": "idea01_candidate_text_features_v1", "input_manifest_sha256": manifest_sha,
            "protocol": {"prompt": "问题：候选动作"}, "records": {}}
    text["protocol_sha256"] = canonical_sha256(text["protocol"])
    for case in load_forecast_cases(root):
        embeddings = torch.randn(6, 2048, generator=generator)
        text["records"][case.id] = {"labels": list("ABCDEF"), "embeddings": embeddings,
                                     "text_sha256": case_text_sha256(case),
                                     "embedding_sha256": tensor_sha256(embeddings)}
    text_path = tmp_path / "text.pt"
    torch.save(text, text_path)
    trials = []
    for trial_index in range(2):
        name = f"trial_{trial_index}"
        directory = tmp_path / name
        directory.mkdir()
        arms = ["recent_only", "residual", "residual_no_update"] if trial_index == 0 else ["recent_only", "residual"]
        protocol = {"split_sha256": runner.sha256_file(split_path), "splits": split["splits"],
                    "input_manifest_sha256": manifest_sha, "arms": arms}
        protocol_sha = canonical_sha256(protocol)
        write_json(directory / "protocol.json", {"protocol": protocol, "protocol_sha256": protocol_sha})
        for sample_id in ids:
            forecasts = {arm: torch.randn(4, 8, 256, generator=generator) for arm in arms}
            artifact = {"schema": "idea01_scaling_case_forecasts_v1", "sample_id": sample_id,
                        "protocol_sha256": protocol_sha, "last_forecasts": forecasts,
                        "stats": {arm: {} for arm in arms},
                        "forecast_sha256": {arm: tensor_sha256(value) for arm, value in forecasts.items()}}
            artifact["metadata_sha256"] = canonical_sha256({key: value for key, value in artifact.items()
                                                             if key != "last_forecasts"})
            torch.save(artifact, directory / f"{sample_id}_forecasts.pt")
        trials.append({"name": name, "forecast_dir": str(directory), "arms": arms})
    plan = {"schema": "idea01_scaling_experiment_plan_v1", "split_manifest": str(split_path),
            "text_cache": str(text_path), "trials": trials, "readout_train_sizes": [2, 3]}
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    return {"root": root, "ids": ids, "split": split, "split_path": split_path,
            "plan": plan, "plan_path": plan_path, "text_path": text_path,
            "answers": answers, "metadata": metadata, "output": tmp_path / "readout"}


def test_all_trial_views_balanced_and_tensor_corruption_rejected(experiment):
    inputs = runner.load_inputs(experiment["plan_path"])
    torch.testing.assert_close(inputs["weights"][:3].sum(), torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(inputs["weights"][3:].sum(), torch.tensor(0.5, dtype=torch.float64))
    directory = experiment["plan"]["trials"][0]["forecast_dir"]
    path = runner.Path(directory) / f"{experiment['ids'][0]}_forecasts.pt"
    artifact = torch.load(path, weights_only=True)
    artifact["last_forecasts"]["residual"][0, 0, 0] += 1
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="tensor invalid"):
        runner.load_inputs(experiment["plan_path"])


def test_prediction_without_test_answers_and_later_separate_scoring(experiment, monkeypatch):
    fitted_ids = []
    original = runner.prepare_ridge_statistics

    def observe(*args, **kwargs):
        fitted_ids.append(kwargs["train_ids"])
        assert set(kwargs["train_ids"]) <= set(experiment["split"]["splits"]["train"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "prepare_ridge_statistics", observe)
    args = ["--plan", str(experiment["plan_path"]), "--output-dir", str(experiment["output"]),
            "--threads", "1"]
    assert runner.main(args) == 0
    assert [len(ids) for ids in fitted_ids] == [2, 3]
    completion = json.loads((experiment["output"] / "completion.json").read_text())
    assert completion["complete"] and completion["test_answers_loaded"] is False
    assert len(completion["files"]) == 2
    assert completion["files"]["shared_n3"]["file"] == "predictions.jsonl"
    assert completion["expected_rows_per_head"] == 5 * 6
    write_jsonl(experiment["root"] / "answers.jsonl", experiment["answers"])
    write_jsonl(experiment["root"] / "metadata.jsonl", experiment["metadata"])
    assert runner.main(args + ["--stage", "score"]) == 0
    aggregate = json.loads((experiment["output"] / "report/aggregate_metrics.json").read_text())
    assert aggregate["primary_head"] == "shared_n3"
    assert set(aggregate["by_head"]) == {"shared_n2", "shared_n3"}
    assert "exploratory" in aggregate["test_status"]
    case_summary = aggregate["by_head"]["shared_n3"]["by_arm"]["trial_0__residual"]
    assert case_summary["by_research_split"]["test"]["n"] == 1
    assert "cases" not in case_summary


def test_frozen_baseline_head_is_reused_without_fitting_or_reselection(experiment):
    inputs = runner.load_inputs(experiment["plan_path"])
    train_ids = experiment["split"]["splits"]["train"]
    x = torch.stack([inputs["features"][sid][:1] for sid in train_ids])
    options = torch.stack([inputs["options"][sid] for sid in train_ids])
    model = SharedRidgeReadout.fit(x, options, torch.arange(3), ridge_lambda=10,
                                   train_ids=train_ids, arm_names=["original"])
    frozen_path = experiment["output"].parent / "baseline.pt"
    torch.save(model.to_dict(), frozen_path)
    plan = experiment["plan"]
    plan["baseline_readout"] = str(frozen_path)
    write_json(experiment["plan_path"], plan)
    assert runner.main(["--plan", str(experiment["plan_path"]), "--output-dir", str(experiment["output"]),
                        "--threads", "1"]) == 0
    saved = torch.load(experiment["output"] / "baseline_frozen.pt", weights_only=True)
    torch.testing.assert_close(saved["coefficients"], model.coefficients)
    output_rows = [json.loads(line) for line in
                   (experiment["output"] / "baseline_frozen_predictions.jsonl").read_text().splitlines()]
    sid = experiment["ids"][-1]
    expected = model.predict_scores(inputs["features"][sid][None], inputs["options"][sid][None])[0]
    actual = [row for row in output_rows if row["id"] == sid and row["arm"] != "text_prior"]
    for index, row in enumerate(actual):
        torch.testing.assert_close(torch.tensor(list(row["candidate_scores"].values()), dtype=torch.float64),
                                   expected[index])


@pytest.mark.parametrize("problem", ["missing_row", "wrong_hash", "bad_score", "changed_split"])
def test_score_integrity_failures_precede_private_reference_access(experiment, monkeypatch, problem):
    runner.main(["--plan", str(experiment["plan_path"]), "--output-dir", str(experiment["output"]),
                 "--threads", "1"])
    output = experiment["output"]
    path = output / "predictions.jsonl"
    completion = json.loads((output / "completion.json").read_text())
    if problem in ("missing_row", "bad_score"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if problem == "missing_row":
            rows.pop()
        else:
            rows[0]["candidate_scores"]["A"] = float("nan")
        write_jsonl(path, rows)
        completion["files"]["shared_n3"]["sha256"] = runner.sha256_file(path)
        write_json(output / "completion.json", completion)
    elif problem == "wrong_hash":
        path.write_text(path.read_text() + "\n")
    else:
        split = json.loads(experiment["split_path"].read_text())
        split["description"] = "different source split"
        write_json(experiment["split_path"], split)

    def forbidden(*args, **kwargs):
        raise AssertionError("private references must not be accessed")

    monkeypatch.setattr(runner, "summarize", forbidden)
    with pytest.raises(ValueError):
        runner.main(["--plan", str(experiment["plan_path"]), "--output-dir", str(output),
                     "--threads", "1", "--stage", "score"])
