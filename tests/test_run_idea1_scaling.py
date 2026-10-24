"""Scaling-plan comparability and saved-state integrity."""
from copy import deepcopy
import json

import pytest
import torch

from scripts.run_idea1_scaling import ARMS, make_plan, tensor_sha, validate_saved
from scripts.run_idea1_mcq_forecasts import sha256_file


def setup_source(tmp_path):
    base = tmp_path / "base"
    (base / "forecasts").mkdir(parents=True)
    split = {"schema": "forecast_bench_research_day_split_v1", "official_split": False,
             "grouping": "recording_day_all_participants", "inputs_manifest_sha256": "manifest",
             "days": {"train": [1], "validation": [5], "test": [6]},
             "splits": {"train": ["a"], "validation": ["b"], "test": ["c"]},
             "id_to_day": {"a": 1, "b": 5, "c": 6}}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    record = {"protocol_sha256": "source", "predictor_sha256": "predictor",
              "protocol": {"split_sha256": sha256_file(path), "splits": split["splits"]}}
    (base / "forecasts/protocol.json").write_text(json.dumps(record))
    return base, path


def test_plan_separates_capacity_online_and_offline_budget(tmp_path):
    base, split = setup_source(tmp_path)
    plan = make_plan(base, tmp_path / "output", split)
    assert len(plan["trials"]) == 12
    assert len({row["name"] for row in plan["trials"]}) == 12
    for width in (128, 512, 2048):
        configs = [row["configuration"] for row in plan["trials"] if row["configuration"]["hidden_dim"] == width]
        assert {(c["inner_steps"], c["pretraining_steps"]) for c in configs} == {
            (1, 0), (4, 0), (1, 200), (1, 800)}
    assert all(row["arms"] == list(ARMS) for row in plan["trials"])
    assert plan["unique_labeled_training_cases_increase"] is False
    assert "exploratory" in plan["test_interpretation"]


def test_plan_rejects_predictor_from_different_split(tmp_path):
    base, split = setup_source(tmp_path)
    path = base / "forecasts/protocol.json"
    row = json.loads(path.read_text())
    row["protocol"]["splits"]["train"] = ["c"]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="source predictor"):
        make_plan(base, tmp_path / "output", split)


def test_saved_forecasts_require_correct_tensor_hash_and_protocol(tmp_path):
    path = tmp_path / "case.pt"
    forecasts = {arm: torch.zeros(4, 8, 256) for arm in ARMS}
    value = {"schema": "idea01_scaling_case_forecasts_v1", "sample_id": "case",
             "protocol_sha256": "protocol", "last_forecasts": forecasts,
             "forecast_sha256": {arm: tensor_sha(tensor) for arm, tensor in forecasts.items()}}
    torch.save(value, path)
    assert validate_saved(path, "case", "protocol")["sample_id"] == "case"
    with pytest.raises(ValueError, match="conflicting"):
        validate_saved(path, "case", "changed")
    damaged = deepcopy(value)
    damaged["last_forecasts"]["residual"][0, 0, 0] += 1
    torch.save(damaged, path)
    with pytest.raises(ValueError, match="corrupt"):
        validate_saved(path, "case", "protocol")
