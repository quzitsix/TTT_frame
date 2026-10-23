"""Information-boundary and temporal-contract tests for forecasting inputs."""

import json
from dataclasses import asdict

import pytest

from ttt_frame.forecast_bench import (
    audit_forecast_bench,
    load_forecast_cases,
    make_research_day_split,
    make_sampling_plan,
)


def _fixture(root, days=(1,)):
    rows, metadata = [], []
    for index, day in enumerate(days, 1):
        case_id = f"egolife_{index:03d}"
        sample = root / "samples" / case_id
        sample.mkdir(parents=True)
        for filename in ("context.mp4", "input.mp4"):
            (sample / filename).write_bytes(b"media existence fixture")
        row = {
            "id": case_id, "task": "egocentric_action_forecasting_5s",
            "input": {
                "context_video": f"samples/{case_id}/context.mp4",
                "video": f"samples/{case_id}/input.mp4", "question": "What happens next?",
                "context_duration_seconds": 1800, "input_duration_seconds": 30,
                "prediction_horizon_seconds": 5,
                "options": [{"label": label, "text": f"action {label}"} for label in "ABCDEF"],
            },
        }
        rows.append(row)
        cutoff = 3000 + index * 40
        metadata.append({
            "id": case_id,
            "metadata": {
                "participant": f"A{index % 2}", "day": day, "split": "benchmark",
                "context_start_seconds": cutoff - 1830,
                "context_end_seconds": cutoff - 30, "input_start_seconds": cutoff - 30,
                "cutoff_seconds": cutoff, "future_end_seconds": cutoff + 5,
                "target_interval_in_future": [1.0, 3.0],
            },
            "output": {
                "answer_category": "take", "correct_answer": "A",
                "option_type_counts": {"plausible_not_observed": 2},
            },
            "verification": {
                "gt_consistency": "supported", "gt_uncertain": False,
                "human_verified": False,
            },
        })
    _write(root / "inputs_mcq.jsonl", rows)
    _write(root / "metadata.jsonl", metadata)
    return rows, metadata


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_inference_does_not_require_or_read_private_files(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "metadata.jsonl").write_text("invalid private data")
    (tmp_path / "answers.jsonl").write_text("invalid private data")
    case = load_forecast_cases(tmp_path)[0]
    assert case.id == "egolife_001"
    assert not {"answer", "correct_answer", "metadata", "future"} & set(asdict(case))
    assert set(asdict(case.options[0])) == {"label", "text"}


@pytest.mark.parametrize("location", ["case", "input", "option"])
def test_private_fields_are_rejected(tmp_path, location):
    rows, _ = _fixture(tmp_path)
    target = {"case": rows[0], "input": rows[0]["input"],
              "option": rows[0]["input"]["options"][0]}[location]
    target["correct_answer"] = "A"
    _write(tmp_path / "inputs_mcq.jsonl", rows)
    with pytest.raises(ValueError, match="public fields"):
        load_forecast_cases(tmp_path)


@pytest.mark.parametrize("path", [
    "samples/egolife_001/future.mp4", "../future.mp4", "/tmp/input.mp4",
    "samples/egolife_002/input.mp4",
])
def test_future_cross_case_and_path_escape_are_rejected(tmp_path, path):
    rows, _ = _fixture(tmp_path)
    rows[0]["input"]["video"] = path
    _write(tmp_path / "inputs_mcq.jsonl", rows)
    with pytest.raises(ValueError, match="past media path"):
        load_forecast_cases(tmp_path)


def test_symlink_cannot_escape_root(tmp_path):
    root = tmp_path / "bench"
    _fixture(root)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    path = root / "samples/egolife_001/input.mp4"
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_forecast_cases(root)


def test_symlink_cannot_redirect_past_to_future_inside_root(tmp_path):
    _fixture(tmp_path)
    sample = tmp_path / "samples/egolife_001"
    (sample / "future.mp4").write_bytes(b"private future")
    (sample / "input.mp4").unlink()
    (sample / "input.mp4").symlink_to(sample / "future.mp4")
    with pytest.raises(ValueError, match="redirect"):
        load_forecast_cases(tmp_path)


def test_sampling_uses_past_only_uniform_grid(tmp_path):
    _fixture(tmp_path)
    case = load_forecast_cases(tmp_path)[0]
    plan = make_sampling_plan(case)
    assert len(plan) == 3660
    assert plan[0].prefix_seconds == 0
    assert plan[3599].role == "context" and plan[3599].video_seconds == 1799.5
    assert plan[3600].role == "recent" and plan[3600].video_seconds == 0
    assert plan[-1].prefix_seconds == 1829.5 and plan[-1].video_seconds == 29.5
    assert {sample.path.name for sample in plan} == {"context.mp4", "input.mp4"}
    assert all(sample.prefix_seconds < case.prefix_seconds for sample in plan)


def test_research_split_holds_out_days_across_people(tmp_path):
    _fixture(tmp_path, days=(1, 1, 4, 5, 6, 7))
    split = make_research_day_split(tmp_path)
    assert split["official_split"] is False
    assert split["counts"] == {"train": 3, "validation": 1, "test": 2}
    assert split["splits"]["train"] == ["egolife_001", "egolife_002", "egolife_003"]
    assert split["splits"]["validation"] == ["egolife_004"]
    assert split["splits"]["test"] == ["egolife_005", "egolife_006"]
    assert split["research_only"] is True
    assert split["grouping"] == "recording_day_all_participants"
    assert split["id_to_day"]["egolife_006"] == 7
    assert len(split["inputs_manifest_sha256"]) == 64
    day_sets = {
        part: {split["id_to_day"][case_id] for case_id in ids}
        for part, ids in split["splits"].items()
    }
    assert not day_sets["train"] & day_sets["validation"]
    assert not day_sets["train"] & day_sets["test"]
    audit = audit_forecast_bench(tmp_path)
    assert audit["human_verified"] == 0
    assert audit["overlap"]["day_across_participants"]["overlapping_pairs"] == 1


def test_subset_selection_rejects_missing_ids_and_duplicate_manifest_ids(tmp_path):
    rows, _ = _fixture(tmp_path, days=(1, 2))
    assert load_forecast_cases(tmp_path, ids=["egolife_002"])[0].id == "egolife_002"
    with pytest.raises(ValueError, match="not present"):
        load_forecast_cases(tmp_path, ids=["egolife_999"])
    _write(tmp_path / "inputs_mcq.jsonl", [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_forecast_cases(tmp_path)
