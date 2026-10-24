"""Frozen input protocol, independent donor and safe resume contracts."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_idea3_spatial as runner
from scripts.evaluate_forecast_mcq import file_hash, fingerprint, source_hashes as baseline_sources


def split_value():
    return {
        "schema": "forecast_bench_research_day_split_v1", "official_split": False,
        "grouping": "recording_day_all_participants", "inputs_manifest_sha256": "replaced",
        "days": {"train": [1, 2], "validation": [5], "test": [6]},
        "splits": {"train": ["egolife_002", "egolife_001"],
                   "validation": ["egolife_003"], "test": ["egolife_004"]},
        "id_to_day": {"egolife_001": 1, "egolife_002": 2,
                      "egolife_003": 5, "egolife_004": 6},
    }


def setup_protocol(tmp_path):
    root = tmp_path / "bench"
    root.mkdir()
    (root / "inputs_mcq.jsonl").write_text("public inputs only\n")
    split = split_value()
    split["inputs_manifest_sha256"] = file_hash(root / "inputs_mcq.jsonl")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    base = {
        "schema": "forecast_mcq_protocol_v1", "root": str(root),
        "case_ids": ["egolife_001", "egolife_002", "egolife_003", "egolife_004"],
        "inputs_sha256": split["inputs_manifest_sha256"], "source_hashes": baseline_sources(),
        "sampling": {"context_frames": 30, "recent_frames": 60,
                     "max_side": 224, "write_chunk_frames": 8},
    }
    baseline = {"payload": base, "protocol_fingerprint": fingerprint(base)}
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    args = SimpleNamespace(protocol=tmp_path / "run" / "protocol.json",
                           baseline_protocol=baseline_path, split_manifest=split_path)
    return args, root, split


def valid_row(sample_id="egolife_003", digest="digest"):
    score_names = ["direct", "wrong_direct", "reset_direct", "reset_search",
                   *runner.SEARCHES, *["wrong_" + name for name in runner.SEARCHES]]
    return {
        "schema": "idea03_native_spatial_scores_v1", "id": sample_id, "status": "ok",
        "protocol_fingerprint": digest, "research_split": "validation", "donor_id": "egolife_001",
        "labels": list("ABCDEF"), "option_texts": ["option " + label for label in "ABCDEF"],
        "prior_scores": [-1.0] * 6, "scores": {name: [0.0] * 6 for name in score_names},
        "diagnostics": {}, "visual_key_diagnostic": {}, "sampling": {}, "write_report": {},
        "query_tokens": 4, "candidate_tokens": [6] * 6, "position_offset": 23,
        "decode_seconds": 1.0, "extraction_seconds": 1.0, "search_seconds": {},
        "elapsed_seconds": 3.0, "answer_keys_used": False,
    }


def test_donors_are_deterministic_train_only_and_from_different_recording_days():
    split = split_value()
    mapping = runner.donor_mapping(split)
    assert mapping == {"egolife_001": "egolife_002", "egolife_002": "egolife_001",
                       "egolife_003": "egolife_001", "egolife_004": "egolife_001"}
    reordered = deepcopy(split)
    reordered["splits"]["train"].reverse()
    assert runner.donor_mapping(reordered) == mapping
    for recipient, donor in mapping.items():
        assert donor in split["splits"]["train"]
        assert split["id_to_day"][donor] != split["id_to_day"][recipient]
    for role in ("validation", "test"):
        assert all(split["id_to_day"][mapping[sample_id]] < split["id_to_day"][sample_id]
                   for sample_id in split["splits"][role])


def test_donor_mapping_rejects_single_day_training():
    split = split_value()
    split["id_to_day"]["egolife_002"] = 1
    with pytest.raises(ValueError, match="different train recording day"):
        runner.donor_mapping(split)


def test_prepare_freezes_sources_public_manifest_split_and_donors(tmp_path):
    args, _, split = setup_protocol(tmp_path)
    record = runner.prepare(args)
    payload = record["payload"]
    assert payload["split"] == split
    assert payload["sources"] == runner.source_hashes()
    assert payload["donors"] == runner.donor_mapping(split)
    assert payload["future_media_used"] is False and payload["answer_keys_used"] is False
    assert payload["candidate_letters_or_positions_encoded"] is False
    assert runner.validate_protocol(args.protocol) == record
    with pytest.raises(FileExistsError, match="protocol exists"):
        runner.prepare(args)


@pytest.mark.parametrize("symlink", [False, True])
def test_prepare_rejects_protocol_inside_source_benchmark(tmp_path, symlink):
    args, root, _ = setup_protocol(tmp_path)
    protocol_root = root
    if symlink:
        protocol_root = tmp_path / "redirect"
        protocol_root.symlink_to(root, target_is_directory=True)
    args.protocol = protocol_root / "new_run" / "protocol.json"
    with pytest.raises(ValueError, match="outside source benchmark"):
        runner.prepare(args)
    assert not args.protocol.exists()
    assert not (root / "new_run").exists()


@pytest.mark.parametrize("tamper", ["fingerprint", "source", "baseline_source", "public_input", "donor"])
def test_validate_rejects_changed_frozen_protocol_or_inputs(tmp_path, tamper):
    args, root, _ = setup_protocol(tmp_path)
    record = runner.prepare(args)
    if tamper == "public_input":
        (root / "inputs_mcq.jsonl").write_text("changed")
    else:
        payload = record["payload"]
        if tamper == "fingerprint":
            payload["steps"] = 9
        elif tamper == "source":
            payload["sources"]["ttt_frame/idea3_spatial_reader.py"] = "changed"
        elif tamper == "baseline_source":
            base = payload["baseline_protocol"]
            base["payload"]["source_hashes"]["ttt_frame/spatial_model.py"] = "changed"
            base["protocol_fingerprint"] = fingerprint(base["payload"])
        else:
            payload["donors"]["egolife_003"] = "egolife_002"
        if tamper != "fingerprint":
            record["protocol_fingerprint"] = fingerprint(payload)
        args.protocol.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        runner.validate_protocol(args.protocol)


def test_completed_accepts_only_complete_successful_unique_selected_rows(tmp_path):
    output = tmp_path / "scores.jsonl"
    assert runner.completed(output, {"egolife_003"}, "digest") == set()
    row = valid_row()
    output.write_text(json.dumps(row) + "\n")
    assert runner.completed(output, {"egolife_003"}, "digest") == {"egolife_003"}
    output.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError):
        runner.completed(output, {"egolife_003"}, "digest")


@pytest.mark.parametrize("change", ["foreign_id", "fingerprint", "error", "truncated_json"])
def test_completed_rejects_errors_and_mixed_or_truncated_rows(tmp_path, change):
    output = tmp_path / "scores.jsonl"
    row = valid_row()
    if change == "foreign_id":
        row["id"] = "egolife_004"
    elif change == "fingerprint":
        row["protocol_fingerprint"] = "other"
    elif change == "error":
        row["status"] = "error"
    text = json.dumps(row) if change != "truncated_json" else '{"id":"egolife_003"'
    output.write_text(text + "\n")
    with pytest.raises(ValueError):
        runner.completed(output, {"egolife_003"}, "digest")


@pytest.mark.parametrize("change", [
    "no_schema", "no_scores", "missing_arm", "extra_arm", "nan", "infinity", "bool",
    "short_scores", "short_prior", "missing_options", "empty_option", "duplicate_label", "answer_keys",
])
def test_completed_rejects_incomplete_or_nonfinite_json_objects(tmp_path, change):
    row = valid_row()
    if change == "no_schema":
        del row["schema"]
    elif change == "no_scores":
        del row["scores"]
    elif change == "missing_arm":
        del row["scores"]["direct"]
    elif change == "extra_arm":
        row["scores"]["unexpected"] = [0.0] * 6
    elif change in {"nan", "infinity", "bool"}:
        row["scores"]["direct"][0] = {"nan": float("nan"), "infinity": float("inf"), "bool": True}[change]
    elif change == "short_scores":
        row["scores"]["direct"] = [0.0] * 5
    elif change == "short_prior":
        row["prior_scores"] = []
    elif change == "missing_options":
        del row["option_texts"]
    elif change == "empty_option":
        row["option_texts"][0] = " "
    elif change == "duplicate_label":
        row["labels"][0] = "B"
    else:
        row["answer_keys_used"] = True
    output = tmp_path / "scores.jsonl"
    output.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        runner.completed(output, {"egolife_003"}, "digest")


@pytest.mark.parametrize("argv", [
    ["--num-shards", "0"], ["--num-shards", "2", "--shard-index", "2"], ["--threads", "0"],
])
def test_cli_rejects_invalid_shards_before_loading_protocol(tmp_path, argv):
    with pytest.raises(SystemExit) as error:
        runner.main(["--protocol", str(tmp_path / "absent.json"), *argv])
    assert error.value.code == 2


@pytest.mark.parametrize("target", ["inside", "protocol", "missing"])
def test_cli_rejects_source_root_or_protocol_output_before_reading_cases(tmp_path, monkeypatch, target):
    args, root, _ = setup_protocol(tmp_path)
    record = runner.prepare(args)
    monkeypatch.setattr(runner, "validate_protocol", lambda _: record)

    def unexpected_load(*_):
        raise AssertionError("invalid output must fail before loading cases")

    monkeypatch.setattr(runner, "load_forecast_cases", unexpected_load)
    argv = ["--protocol", str(args.protocol)]
    if target != "missing":
        output = root / "scores.jsonl" if target == "inside" else args.protocol
        argv += ["--output", str(output)]
    with pytest.raises(SystemExit) as error:
        runner.main(argv)
    assert error.value.code == 2


def test_resume_smoke_flags_skip_completed_case_without_loading_models(tmp_path, monkeypatch):
    args, _, _ = setup_protocol(tmp_path)
    record = runner.prepare(args)
    row = valid_row(digest=record["protocol_fingerprint"])
    output = tmp_path / "scores.jsonl"
    output.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(runner, "load_forecast_cases", lambda _: [SimpleNamespace(id="egolife_003")])
    assert runner.main(["--protocol", str(args.protocol), "--output", str(output),
                        "--device", "cuda:4", "--threads", "2", "--ids", "egolife_003",
                        "--num-shards", "1", "--shard-index", "0", "--resume"]) == 0
