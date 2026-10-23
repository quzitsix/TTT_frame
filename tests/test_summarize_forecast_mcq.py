"""Post-hoc MCQ score contracts: full denominator and strict run identities."""

import json
import hashlib

import pytest

from scripts.summarize_forecast_mcq import (
    load_references,
    main,
    parse_choice,
    summarize,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(root):
    public, metadata, answers = [], [], []
    for index, day in enumerate((1, 1, 2), 1):
        sample_id = f"egolife_{index:03d}"
        sample = root / "samples" / sample_id
        sample.mkdir(parents=True)
        for name in ("context.mp4", "input.mp4"):
            (sample / name).touch()
        options = [{"label": label, "text": f"action {label}"} for label in "ABCDEF"]
        public.append({
            "id": sample_id, "task": "egocentric_action_forecasting_5s",
            "input": {
                "context_video": f"samples/{sample_id}/context.mp4",
                "video": f"samples/{sample_id}/input.mp4", "question": "What happens next?",
                "context_duration_seconds": 1800, "input_duration_seconds": 30,
                "prediction_horizon_seconds": 5, "options": options,
            },
        })
        output = {"correct_answer": "ABC"[index - 1], "options": options,
                  "option_type_counts": {"plausible_not_observed": index - 1,
                                         "implausible_not_observed": 6 - index}}
        answers.append({"id": sample_id, **output})
        metadata.append({
            "id": sample_id, "output": output,
            "metadata": {
                "participant": "wearer_a" if index != 2 else "wearer_b", "day": day,
                "context_start_seconds": 1000, "context_end_seconds": 2800,
                "input_start_seconds": 2800, "cutoff_seconds": 2830,
                "future_end_seconds": 2835,
            },
            "verification": {"gt_consistency": "supported" if index < 3 else "unclear",
                             "human_verified": False},
        })
    for name, rows in (("inputs_mcq", public), ("metadata", metadata), ("answers", answers)):
        write_rows(root / f"{name}.jsonl", rows)
    return public, metadata, answers


def prediction(sample_id="egolife_001", arm="blind", choice="A", **extra):
    return {"schema": "forecast_mcq_prediction_v1", "id": sample_id, "arm": arm,
            "prediction": choice, "raw_answer": choice or "",
            "status": "ok", "elapsed_seconds": 1.5, "protocol_fingerprint": "same",
            **extra}


@pytest.mark.parametrize(("raw", "expected"), [
    ("a", "A"), (" B。", "B"), ("C)", "C"), ("action D!", "D"),
    ("The correct answer is A", None), ("A or B", None), ("A!", None),
    (None, None), (3, None), ("action E because", None),
])
def test_strict_choice_parser(raw, expected):
    assert parse_choice(raw, {label: f"action {label}" for label in "ABCDEF"}) == expected


def test_full_denominator_and_day_macro(tmp_path):
    root = tmp_path / "bench"
    fixture(root)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(), prediction("egolife_002", choice="A")])
    report, official = summarize(root, [path], ["blind", "full_icl"])
    blind = report["by_arm"]["blind"]
    assert blind["n"] == 3 and blind["correct"] == 1
    assert blind["accuracy_all"] == pytest.approx(1 / 3)
    assert blind["day_macro_accuracy"] == 0.25
    assert blind["missing"] == 1 and blind["valid"] == 2
    assert report["by_arm"]["full_icl"]["missing"] == 3
    assert not report["complete"] and report["expected_rows"] == 6
    assert len(official["blind"]) == 2 and official["full_icl"] == []
    assert blind["by_gt_consistency"]["unclear"]["n"] == 1
    assert len(blind["by_distractor_mix"]) == 3


def test_errors_and_invalids_never_count_correct(tmp_path):
    root = tmp_path / "bench"
    fixture(root)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(status="error"),
                     prediction("egolife_002", choice="B", status="parse_error"),
                     prediction("egolife_003", choice="I choose C")])
    report, official = summarize(root, [path], ["blind"])
    arm = report["by_arm"]["blind"]
    assert arm["correct"] == 0 and arm["errors"] == 1 and arm["invalid"] == 2
    assert arm["complete"] and arm["missing"] == 0
    assert all(row["prediction"] is None for row in official["blind"])


def test_paired_counts_include_missing_as_incorrect(tmp_path):
    root = tmp_path / "bench"
    fixture(root)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(), prediction("egolife_002", choice="C"),
                     prediction(arm="full_icl", choice="B"),
                     prediction("egolife_002", arm="full_icl", choice="B"),
                     prediction("egolife_003", arm="full_icl", choice="C")])
    report, _ = summarize(root, [path], ["blind", "full_icl"])
    paired = report["paired_vs_baselines"]["full_icl"]["blind"]
    assert paired["arm_only_correct"] == 2 and paired["baseline_only_correct"] == 1
    assert paired["both_correct"] == 0 and paired["both_incorrect"] == 0
    assert paired["both_valid"] == 2 and paired["valid_prediction_disagreements"] == 2
    assert paired["accuracy_delta_all"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("change, message", [
    ({"protocol_fingerprint": "other"}, "inconsistent protocol"),
    ({"id": "egolife_999"}, "unknown prediction"),
    ({"arm": "unknown"}, "unknown prediction"),
    ({"schema": "other"}, "schema"),
    ({"protocol_fingerprint": ""}, "protocol_fingerprint"),
    ({"elapsed_seconds": float("nan")}, "elapsed_seconds"),
    ({"status": "mystery"}, "status"),
])
def test_malformed_or_mixed_runs_rejected(tmp_path, change, message):
    root = tmp_path / "bench"
    fixture(root)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(), prediction("egolife_002", **change)])
    with pytest.raises(ValueError, match=message):
        summarize(root, [path], ["blind"])


def test_duplicate_rows_across_shards_rejected(tmp_path):
    root = tmp_path / "bench"
    fixture(root)
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path in paths:
        write_rows(path, [prediction()])
    with pytest.raises(ValueError, match="duplicate prediction"):
        summarize(root, paths, ["blind"])


def test_private_public_key_mismatch_rejected(tmp_path):
    root = tmp_path / "bench"
    _, _, answers = fixture(root)
    answers[0]["correct_answer"] = "F"
    write_rows(root / "answers.jsonl", answers)
    with pytest.raises(ValueError, match="disagree"):
        load_references(root)


def test_cli_outputs_official_exports_and_prevents_overwrite(tmp_path):
    root = tmp_path / "bench"
    fixture(root)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(choice="action A。")])
    output = tmp_path / "report"
    args = ["--root", str(root), "--predictions", str(path),
            "--expected-arms", "blind", "--output-dir", str(output)]
    assert main(args) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["by_arm"]["blind"]["correct"] == 1
    assert json.loads((output / "blind_predictions.jsonl").read_text()) == {
        "id": "egolife_001", "prediction": "A",
    }
    assert "1/3" in (output / "results.md").read_text()
    with pytest.raises(SystemExit):
        main(args)
    args[-1] = str(root / "report")
    with pytest.raises(SystemExit):
        main(args)


def split_fixture(root, tmp_path):
    data = {
        "schema": "forecast_bench_research_day_split_v1", "official_split": False,
        "grouping": "recording_day_all_participants",
        "inputs_manifest_sha256": hashlib.sha256((root / "inputs_mcq.jsonl").read_bytes()).hexdigest(),
        "splits": {"train": ["egolife_001"], "validation": ["egolife_002"],
                   "test": ["egolife_003"]},
        "days": {"train": [1], "validation": [2], "test": [3]},
        "id_to_day": {"egolife_001": 1, "egolife_002": 2, "egolife_003": 3},
        "counts": {"train": 1, "validation": 1, "test": 1},
    }
    metadata = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines()]
    for index, row in enumerate(metadata, 1):
        row["metadata"]["day"] = index
    write_rows(root / "metadata.jsonl", metadata)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(data))
    return path, data


def test_fixed_research_split_and_qwen_paired_scores(tmp_path):
    root = tmp_path / "bench"
    fixture(root)
    split, _ = split_fixture(root, tmp_path)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(arm="qwen_blind"),
                     prediction("egolife_002", arm="qwen_full_icl", choice="B"),
                     prediction("egolife_003", arm="qwen_full_icl", choice="C")])
    report, _ = summarize(root, [path], ["qwen_blind", "qwen_full_icl"], split)
    assert report["research_split"]["counts"] == {"train": 1, "validation": 1, "test": 1}
    test = report["by_arm"]["qwen_full_icl"]["by_research_split"]["test"]
    assert test["n"] == 1 and test["accuracy_all"] == 1
    paired = report["paired_vs_baselines"]["qwen_full_icl"]["qwen_blind"]
    assert paired["by_research_split"]["test"]["arm_only_correct"] == 1
    assert paired["by_research_split"]["train"]["baseline_only_correct"] == 1
    output = tmp_path / "split_report"
    assert main(["--root", str(root), "--predictions", str(path),
                 "--output-dir", str(output), "--split-manifest", str(split)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["by_arm"]) == 9
    assert "Fixed research split" in (output / "results.md").read_text()


def test_label_position_and_optional_readout_diagnostics(tmp_path):
    root = tmp_path / "bench"
    fixture(root)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(label_probability_mass=0.1, unrestricted_top_token="A"),
                     prediction("egolife_002", choice="F", label_probability_mass=0.3,
                                unrestricted_top_token="G"),
                     prediction("egolife_003", choice="C", status="error",
                                label_probability_mass=0.9, unrestricted_top_token="C")])
    report, _ = summarize(root, [path], ["blind", "full_icl"])
    arm = report["by_arm"]["blind"]
    assert arm["predicted_label_counts"] == {"A": 1, "B": 0, "C": 0, "D": 0, "E": 0, "F": 1}
    assert arm["gold_label_counts"] == {"A": 1, "B": 1, "C": 1, "D": 0, "E": 0, "F": 0}
    assert arm["by_gold_label"]["A"]["accuracy_all"] == 1
    assert arm["by_gold_label"]["B"]["accuracy_all"] == 0
    assert arm["by_gold_label"]["F"]["accuracy_all"] is None
    diagnostic = arm["readout_diagnostics"]
    assert diagnostic["label_probability_mass_count"] == 2
    assert diagnostic["label_probability_mass_mean"] == pytest.approx(0.2)
    assert diagnostic["label_probability_mass_min"] == 0.1
    assert diagnostic["unrestricted_top_token_count"] == 2
    assert diagnostic["unrestricted_top_token_matches_prediction"] == 1
    assert diagnostic["unrestricted_top_token_prediction_agreement"] == 0.5
    absent = report["by_arm"]["full_icl"]["readout_diagnostics"]
    assert absent["label_probability_mass_count"] == 0
    assert absent["label_probability_mass_mean"] is None
    assert absent["unrestricted_top_token_prediction_agreement"] is None
    assert arm["cases"][1]["unrestricted_top_token"] == "G"


@pytest.mark.parametrize("field, value", [
    ("label_probability_mass", float("nan")), ("label_probability_mass", -0.1),
    ("label_probability_mass", 1.1), ("label_probability_mass", True),
    ("unrestricted_top_token", 10),
])
def test_malformed_readout_diagnostics_rejected(tmp_path, field, value):
    root = tmp_path / "bench"
    fixture(root)
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction(**{field: value})])
    with pytest.raises(ValueError, match=field):
        summarize(root, [path], ["blind"])


@pytest.mark.parametrize("mutation, message", [
    ("sha", "SHA"), ("day", "ID/day"), ("overlap_days", "overlapping"),
    ("overlap_ids", "overlapping IDs"), ("empty_role", "empty"),
    ("counts", "count"),
])
def test_research_split_rejects_wrong_identity_or_overlap(tmp_path, mutation, message):
    root = tmp_path / "bench"
    fixture(root)
    split, data = split_fixture(root, tmp_path)
    if mutation == "sha":
        data["inputs_manifest_sha256"] = "wrong"
    elif mutation == "day":
        data["id_to_day"]["egolife_001"] = 2
    elif mutation == "overlap_days":
        data["days"]["validation"] = [1]
    elif mutation == "overlap_ids":
        data["splits"]["validation"] = ["egolife_001"]
    elif mutation == "empty_role":
        data["splits"]["test"] = []
    elif mutation == "counts":
        data["counts"]["train"] = 9
    split.write_text(json.dumps(data))
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [prediction()])
    with pytest.raises(ValueError, match=message):
        summarize(root, [path], ["blind"], split)
