import json
from pathlib import Path

import pytest

from scripts.summarize_ideax_answers import build, predictions


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture
def fixture(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    ids = [f"egolife_{i:03}" for i in range(1, 7)]
    options = [{"label": a, "text": f"动作{a}|原文"} for a in "ABCDEF"]
    jsonl(bench / "inputs_mcq.jsonl", [{"id": sid, "input": {"question": "下一步动作？", "options": options}} for sid in ids])
    jsonl(bench / "answers.jsonl", [{"id": sid, "correct_answer": "A", "answer": options[0]["text"],
        "options": [{**o, "role": "ground_truth" if o["label"] == "A" else "implausible_not_observed"} for o in options]} for sid in ids])
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"splits": {"train": ids[:2], "validation": ids[2:4], "test": ids[4:]},
                                "id_to_day": {sid: 1 + i//2 for i, sid in enumerate(ids)}}))
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    jsonl(old, [{"id": sid, "prediction": "A" if i % 2 else "B"} for i, sid in enumerate(ids)])
    jsonl(new, [{"id": sid, "predictions": {"search": "B" if i % 2 else "A"},
                "correct": {"search": True}} for i, sid in enumerate(ids)])
    config = {"benchmark_root": str(bench), "split_manifest": str(split), "baseline": "ICL",
              "arms": [{"name": "ICL", "path": str(old)}, {"name": "search", "path": str(new), "key": "search"}],
              "comparisons": [["search", "ICL"]], "example_limit": 4}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, config, ids


def test_literal_answers_metrics_and_posthoc_examples(fixture, tmp_path):
    path, _, _ = fixture
    output = tmp_path / "report"
    report = build(path, output)
    assert report["metrics"]["search"]["test"]["correct"] == 1
    pair = report["paired"]["search vs ICL"]["test"]
    assert pair["gain_ids"] == ["egolife_005"]
    assert pair["loss_ids"] == ["egolife_006"]
    assert report["example_ids"] == ["egolife_005", "egolife_006"]
    rows = [json.loads(s) for s in (output / "answers.jsonl").read_text().splitlines()]
    assert rows[-1]["predictions"]["search"]["text"] == "动作B|原文"
    assert rows[-1]["predictions"]["search"]["correct"] is False  # Ignore supplied correctness.
    assert "动作B\\|原文" in (output / "concrete_answers.md").read_text()
    assert report["selection_is_posthoc"] is True
    assert report["examples_are_generated_rationales"] is False
    assert b"\r\n" not in (output / "answers.csv").read_bytes()
    with pytest.raises(FileExistsError):
        build(path, output)


def test_prediction_coverage_and_duplicates_rejected(fixture):
    _, config, ids = fixture
    path = Path(config["arms"][0]["path"])
    with pytest.raises(ValueError, match="incomplete"):
        predictions(path, None, set(ids[:-1]))
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        predictions(path, None, set(ids))


def test_multiarms_require_explicit_selection(fixture):
    _, config, ids = fixture
    path = Path(config["arms"][1]["path"])
    with pytest.raises(ValueError, match="explicit key"):
        predictions(path, None, set(ids))


def test_no_answer_text_or_split_misalignment(fixture, tmp_path):
    path, config, _ = fixture
    answers = Path(config["benchmark_root"]) / "answers.jsonl"
    rows = [json.loads(s) for s in answers.read_text().splitlines()]
    rows[0]["options"][0]["text"] = "incorrect text"
    jsonl(answers, rows)
    with pytest.raises(ValueError, match="disagrees"):
        build(path, tmp_path / "report")


def test_refuses_benchmark_output(fixture):
    path, config, _ = fixture
    with pytest.raises(ValueError, match="benchmark"):
        build(path, Path(config["benchmark_root"]) / "report")


def test_overlapping_day_split_rejected(fixture, tmp_path):
    path, config, ids = fixture
    split = Path(config["split_manifest"])
    data = json.loads(split.read_text())
    data["splits"]["test"].append(ids[0])
    split.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap"):
        build(path, tmp_path / "report")


def test_changed_frozen_predictions_rejected(fixture, tmp_path):
    path, config, _ = fixture
    config["arms"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="checksum"):
        build(path, tmp_path / "report")
