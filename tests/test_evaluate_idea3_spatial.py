"""Causal selection and paired reporting for the native Spatial address diagnostic."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import evaluate_idea3_spatial as runner


def make_inputs(tmp_path):
    root = tmp_path / "bench"
    root.mkdir()
    split = tmp_path / "split.json"
    split_value = {"schema": "forecast_bench_research_day_split_v1", "official_split": False,
                   "grouping": "recording_day_all_participants", "root": str(root),
                   "splits": {"train": ["a", "e"], "validation": ["b"], "test": ["c", "d"]},
                   "days": {"train": [1, 2], "validation": [5], "test": [6, 7]},
                   "id_to_day": {"a": 1, "e": 2, "b": 5, "c": 6, "d": 7}}
    rows, public = [], []
    for sid, role in (("a", "train"), ("b", "validation"), ("c", "test"), ("d", "test"), ("e", "train")):
        direct = [3., 0., 0., 0., 0., 0.]
        search = [3., 6., 0., 0., 0., 0.]
        values = {key: direct.copy() for key in runner.REQUIRED_SCORES}
        for key in ("search_r05", "search_r10"):
            values[key] = search.copy()
        for key in ("reset_direct", "reset_search"):
            values[key] = [0.] * 6
        rows.append({"id": sid, "research_split": role, "labels": list("ABCDEF"),
                     "option_texts": list("abcdef"), "prior_scores": direct, "scores": values})
        public.append({"id": sid, "input": {"context_duration_seconds": 1800, "input_duration_seconds": 30,
                       "context_video": f"samples/{sid}/context.mp4", "video": f"samples/{sid}/input.mp4",
                       "options": [{"label": label, "text": text}
                       for label, text in zip("ABCDEF", "abcdef")]}})
    scores = tmp_path / "scores.jsonl"
    runner.write_rows(root / "inputs_mcq.jsonl", public)
    split_value["inputs_manifest_sha256"] = runner.sha256_file(root / "inputs_mcq.jsonl")
    runner.write_json(split, split_value)
    sampling = {"context_frames": 30, "recent_frames": 60, "max_side": 224,
                "write_chunk_frames": 8, "audio_used": False}
    base = {"schema": "forecast_mcq_protocol_v1", "root": str(root),
            "inputs_sha256": split_value["inputs_manifest_sha256"], "case_ids": list("abcde"),
            "source_hashes": {"scripts/evaluate_forecast_mcq.py": runner.sha256_file(
                runner.REPO / "scripts/evaluate_forecast_mcq.py")}, "sampling": sampling,
            "model_path": "synthetic-base-model", "spatial_checkpoint": "synthetic-official-checkpoint",
            "spatial_checkpoint_sha256": "0" * 64}
    native = {"schema": "idea03_native_spatial_probe_v1", "baseline_protocol": {
        "payload": base, "protocol_fingerprint": runner.canonical_sha256(base)},
        "split": split_value, "split_sha256": runner.sha256_file(split),
        "sources": {"scripts/evaluate_idea3_spatial.py": runner.sha256_file(Path(runner.__file__))},
        "answer_keys_used": False, "future_media_used": False, "candidate_letters_or_positions_encoded": False,
        "searches": {f"search_{name}": radius for name, radius in runner.RADII},
        "steps": 3, "step_size": 1., "regularization": .01,
        "donors": {sid: "e" if sid == "a" else "a" for sid in "abcde"}}
    inference = {"payload": native, "protocol_fingerprint": runner.canonical_sha256(native)}
    runner.write_json(scores.with_suffix(".protocol.json"), inference)
    for row in rows:
        sid = row["id"]
        audit = {"schema": "bench_v0_mcq_past_frame_bundle_v1", "sample_id": sid,
                 "decoded": True, "audio_used": False,
                 "sampling": {**sampling, "cutoff_prefix_seconds": 1830}, "source_media": {},
                 **{key: "0" * 64 for key in ("frame_plan_sha256", "frame_bundle_sha256",
                    "context_frame_bundle_sha256", "recent_frame_bundle_sha256")}}
        for role, filename, offset, duration in (("context", "context.mp4", 0, 1800),
                                                 ("recent", "input.mp4", 1800, 30)):
            times = [offset + duration * i / sampling[f"{role}_frames"] for i in range(sampling[f"{role}_frames"])]
            audit[f"{role}_requested_seconds"] = times
            audit[f"{role}_actual_seconds"] = times.copy()
            audit["source_media"][role] = {"path": str(root / f"samples/{sid}/{filename}"),
                                           "size_bytes": 100, "mtime_ns": 1}
        row.update({"schema": "idea03_native_spatial_scores_v1", "status": "ok", "answer_keys_used": False,
                    "protocol_fingerprint": inference["protocol_fingerprint"], "donor_id": native["donors"][sid],
                    "sampling": audit, "diagnostics": {}})
        for prefix in ("", "wrong_"):
            for key, radius in native["searches"].items():
                row["diagnostics"][prefix + key] = {
                    "search": {"steps": 3, "step_size": 1., "regularization": .01, "radius": radius},
                    "layers": 1, "per_layer": {"0": {}}, "max_displacement": 0.,
                    "direct_scores": row["scores"][prefix + "direct"],
                    "direct_objectives": row["scores"][prefix + "direct"],
                    "search_scores": row["scores"][prefix + key],
                    "search_objectives": row["scores"][prefix + key]}
    runner.write_rows(scores, rows)
    runner.write_rows(root / "answers.jsonl", [{"id": sid, "correct_answer": answer}
                      for sid, answer in zip("abcde", "BBBAB")])
    baseline = tmp_path / "baseline.jsonl"
    runner.write_rows(baseline, [{"id": sid, "prediction": "A"} for sid in "abcde"])
    return scores, split, root, tmp_path / "evaluation", baseline


@pytest.fixture
def inputs(tmp_path):
    return make_inputs(tmp_path)


def test_predict_filters_before_reading_test_or_train_answer_fields(inputs):
    scores, split, root, output, _ = inputs
    runner.write_rows(root / "answers.jsonl", [{"id": sid, **({"correct_answer": "B"} if sid == "b" else {})}
                      for sid in "abcde"])
    selection = runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    assert selection["direct"]["beta"] == 0
    assert selection["search"]["beta"] == 10
    assert selection["search"]["radius"] == .5
    assert len(selection["search_grid"]) == 18
    completion = runner.read_json(output / "completion.json")
    assert completion["test_answer_fields_used"] is False
    rows = runner.read_rows(output / "predictions.jsonl")
    assert len(rows) == 5
    for row in rows:
        assert row["predictions"]["reset"] == row["predictions"]["frozen_init"] == row["predictions"]["prior"]
        assert row["predictions"]["search"] == "B"
        assert row["predictions"]["direct_matched"] == "A"


def test_test_answer_changes_cannot_change_selection_or_predictions(inputs):
    scores, split, root, output, _ = inputs
    first = runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    labels = runner.read_rows(root / "answers.jsonl")
    for row in labels:
        if row["id"] in {"c", "d"}:
            row["correct_answer"] = "F"
    runner.write_rows(root / "answers.jsonl", labels)
    second_output = output.with_name("evaluation2")
    second = runner.predict(scores, split, root, second_output, scores.with_suffix(".protocol.json"))
    assert first == second
    assert (output / "predictions.jsonl").read_bytes() == (second_output / "predictions.jsonl").read_bytes()


def test_beta_zero_rejects_memory_and_ties_choose_smallest_radius(inputs):
    scores, split, root, output, _ = inputs
    rows = runner.read_rows(root / "answers.jsonl")
    rows[1]["correct_answer"] = "A"
    runner.write_rows(root / "answers.jsonl", rows)
    selection = runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    assert selection["direct"]["beta"] == selection["search"]["beta"] == 0
    assert selection["search"]["radius"] == .1


def test_score_normalization_and_predictions_are_candidate_permutation_equivariant(inputs):
    scores, _, _, _, _ = inputs
    original = runner.read_rows(scores)[0]
    permutation = [4, 2, 1, 5, 0, 3]
    changed = deepcopy(original)
    for key in ("labels", "option_texts", "prior_scores"):
        changed[key] = [original[key][i] for i in permutation]
    changed["scores"] = {key: [values[i] for i in permutation] for key, values in original["scores"].items()}
    for beta in runner.BETAS:
        first = runner.mixed_scores(original, "search_r05", beta)
        second = runner.mixed_scores(changed, "search_r05", beta)
        assert second == [first[i] for i in permutation]
        assert runner.select_label(original, first) == runner.select_label(changed, second)
    # Even an all-zero diagnostic resolves ties by semantic text, not position.
    assert runner.select_label(original, [0.] * 6) == runner.select_label(changed, [0.] * 6)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "reset", "nan", "options", "role"])
def test_rejects_invalid_scores(inputs, problem):
    scores, split, root, output, _ = inputs
    rows = runner.read_rows(scores)
    if problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing":
        rows.pop()
    elif problem == "reset":
        rows[0]["scores"]["reset_search"][0] = .001
    elif problem == "nan":
        rows[0]["prior_scores"][0] = float("nan")
    elif problem == "options":
        rows[0]["option_texts"][0] = "private leaked answer"
    else:
        rows[0]["research_split"] = "test"
    scores.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    assert not output.exists()


def test_scoring_uses_frozen_predictions_and_reports_paired_gain_loss(inputs):
    scores, split, root, output, baseline = inputs
    runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    report = runner.score(scores, split, root, output, baseline, inference_protocol=scores.with_suffix(".protocol.json"))
    assert report["metrics"]["search"]["test"]["correct"] == 1
    assert report["metrics"]["full_icl"]["test"]["correct"] == 1
    pair = report["paired"]["search"]["full_icl"]["test"]
    assert pair["gains"] == pair["losses"] == 1
    assert pair["gain_ids"] == ["c"] and pair["loss_ids"] == ["d"]
    assert report["metrics"]["search"]["test"]["by_day"] == {
        "6": {"n": 1, "correct": 1}, "7": {"n": 1, "correct": 0}}
    assert (output / "report/per_case.csv").exists()
    assert "correct_answer" not in (output / "report/per_case.jsonl").read_text()
    with pytest.raises(FileExistsError):
        runner.score(scores, split, root, output, baseline, inference_protocol=scores.with_suffix(".protocol.json"))


@pytest.mark.parametrize("which", ["predictions", "scores", "protocol", "baseline"])
def test_scoring_rejects_changed_artifacts_before_opening_answers(inputs, which):
    scores, split, root, output, baseline = inputs
    runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    (root / "answers.jsonl").unlink()
    if which == "predictions":
        with (output / "predictions.jsonl").open("a") as stream:
            stream.write("\n")
    elif which == "scores":
        with scores.open("a") as stream:
            stream.write("\n")
    elif which == "protocol":
        content = runner.read_json(output / "protocol.json")
        content["protocol"]["selection"]["search"]["beta"] = 1000
        runner.write_json(output / "protocol.json", content)
    else:
        runner.write_rows(baseline, runner.read_rows(baseline)[:-1])
    with pytest.raises(ValueError):
        runner.score(scores, split, root, output, baseline, inference_protocol=scores.with_suffix(".protocol.json"))
    assert not (output / "report").exists()


def test_existing_outputs_are_never_overwritten(inputs):
    scores, split, root, output, _ = inputs
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    assert (output / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("problem", ["fingerprint", "status", "answers", "donor", "future_frame",
                                     "frame_count", "search_config", "diagnostic_scores"])
def test_prediction_rejects_mixed_or_noncausal_inference_rows(inputs, problem):
    scores, split, root, output, _ = inputs
    rows = runner.read_rows(scores)
    row = rows[0]
    if problem == "fingerprint":
        row["protocol_fingerprint"] = "foreign"
    elif problem == "status":
        row["status"] = "error"
    elif problem == "answers":
        row["answer_keys_used"] = True
    elif problem == "donor":
        row["donor_id"] = row["id"]
    elif problem == "future_frame":
        row["sampling"]["recent_actual_seconds"][-1] = 1831
    elif problem == "frame_count":
        row["sampling"]["context_actual_seconds"].pop()
    elif problem == "search_config":
        row["diagnostics"]["search_r01"]["search"]["steps"] = 10
    else:
        row["diagnostics"]["search_r01"]["search_objectives"][0] += 1
    runner.write_rows(scores, rows)
    with pytest.raises(ValueError):
        runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))
    assert not output.exists()


@pytest.mark.parametrize("problem", ["schema", "sources", "baseline_sources", "donors", "future"])
def test_prediction_requires_fingerprinted_current_inference_provenance(inputs, problem):
    scores, split, root, output, _ = inputs
    inference_path = scores.with_suffix(".protocol.json")
    inference = runner.read_json(inference_path)
    payload = inference["payload"]
    if problem == "schema":
        payload["schema"] = "anything"
    elif problem == "sources":
        payload["sources"]["scripts/evaluate_idea3_spatial.py"] = "0" * 64
    elif problem == "baseline_sources":
        base = payload["baseline_protocol"]
        base["payload"]["source_hashes"]["scripts/evaluate_forecast_mcq.py"] = "0" * 64
        base["protocol_fingerprint"] = runner.canonical_sha256(base["payload"])
    elif problem == "donors":
        payload["donors"]["c"] = "c"
    else:
        payload["future_media_used"] = True
    inference["protocol_fingerprint"] = runner.canonical_sha256(payload)
    runner.write_json(inference_path, inference)
    with pytest.raises(ValueError):
        runner.predict(scores, split, root, output, inference_path)


def test_declared_search_and_budget_are_derived_from_inference_protocol(inputs):
    scores, split, root, output, _ = inputs
    inference_path = scores.with_suffix(".protocol.json")
    inference = runner.read_json(inference_path)
    inference["payload"]["steps"] = 5
    inference["protocol_fingerprint"] = runner.canonical_sha256(inference["payload"])
    runner.write_json(inference_path, inference)
    rows = runner.read_rows(scores)
    for row in rows:
        row["protocol_fingerprint"] = inference["protocol_fingerprint"]
        for diagnostics in row["diagnostics"].values():
            diagnostics["search"]["steps"] = 5
    runner.write_rows(scores, rows)
    runner.predict(scores, split, root, output, inference_path)
    protocol = runner.read_json(output / "protocol.json")["protocol"]
    assert protocol["search"]["steps"] == 5
    assert protocol["input_budget"]["context_frames"] == 30
    assert protocol["inference_protocol_sha256"] == runner.sha256_file(output / "inference_protocol.json")
    assert protocol["split_sha256"] == runner.sha256_file(output / "split_manifest.json")


@pytest.mark.parametrize("problem", ["day_overlap", "wrong_schema", "input_hash"])
def test_day_split_cannot_be_silently_replaced(inputs, problem):
    scores, split, root, output, _ = inputs
    value = runner.read_json(split)
    if problem == "day_overlap":
        value["days"]["validation"] = [1]
        value["id_to_day"]["b"] = 1
    elif problem == "wrong_schema":
        value["schema"] = "random_question_split"
    else:
        value["inputs_manifest_sha256"] = "0" * 64
    runner.write_json(split, value)
    with pytest.raises(ValueError):
        runner.predict(scores, split, root, output, scores.with_suffix(".protocol.json"))


def test_output_cannot_be_written_into_source_benchmark(inputs):
    scores, split, root, _, _ = inputs
    with pytest.raises(ValueError, match="outside"):
        runner.predict(scores, split, root, root / "evaluation", scores.with_suffix(".protocol.json"))
    assert not (root / "evaluation").exists()
