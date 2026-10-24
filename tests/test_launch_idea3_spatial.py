"""Only complete, validated, disjoint Idea 3 shards may become a merged run."""

from copy import deepcopy
import json

import pytest

from scripts import launch_idea3_spatial as launcher
from scripts.evaluate_forecast_mcq import file_hash
from scripts.run_idea3_spatial import SEARCHES


def score_row(sample_id):
    names = {"direct", "wrong_direct", "reset_direct", "reset_search"} | {
        prefix + name for prefix in ("", "wrong_") for name in SEARCHES}
    return {"schema": "idea03_native_spatial_scores_v1", "id": sample_id,
            "status": "ok", "protocol_fingerprint": "frozen", "answer_keys_used": False,
            "labels": list("ABCDEF"), "option_texts": ["action " + label for label in "ABCDEF"],
            "prior_scores": [-1.0] * 6, "scores": {name: [0.0] * 6 for name in names}}


def setup_run(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    directory.mkdir()
    protocol = tmp_path / "protocol.json"
    ids = [f"egolife_{index:03d}" for index in range(1, 6)]
    record = {"protocol_fingerprint": "frozen", "payload": {"baseline_protocol": {
        "payload": {"case_ids": ids, "root": str(tmp_path / "bench")}}}}
    protocol.write_text(json.dumps(record))
    monkeypatch.setattr(launcher, "validate_protocol", lambda _: deepcopy(record))
    for index in range(2):
        path = directory / f"scores_{index}.jsonl"
        # Worker rows can arrive in arbitrary order; the merge restores case order.
        path.write_text("".join(json.dumps(score_row(sid)) + "\n" for sid in reversed(ids[index::2])))
        path.with_suffix(".completion.json").write_text(json.dumps({
            "complete": True, "test_answers_loaded": False, "rows": len(ids[index::2]),
            "protocol_fingerprint": "frozen", "scores_sha256": file_hash(path),
        }))
    return directory, protocol, ids, record


def test_merge_restores_case_order_records_hashes_and_is_idempotent(tmp_path, monkeypatch):
    directory, protocol, ids, _ = setup_run(tmp_path, monkeypatch)
    result = launcher.merge(protocol, directory, 2)
    assert result == directory / "scores.jsonl"
    assert [json.loads(line)["id"] for line in result.read_text().splitlines()] == ids
    completion = json.loads((directory / "scores.completion.json").read_text())
    assert completion["complete"] is True
    assert completion["rows"] == 5
    assert completion["scores_sha256"] == file_hash(result)
    assert completion["test_answers_loaded"] is False
    assert completion["shards"] == {
        f"scores_{index}.jsonl": file_hash(directory / f"scores_{index}.jsonl") for index in range(2)}
    before = result.read_bytes()
    assert launcher.merge(protocol, directory, 2) == result
    assert result.read_bytes() == before


@pytest.mark.parametrize("corruption", ["missing_file", "missing_row", "duplicate", "foreign_shard",
                                         "mixed_protocol", "missing_scores", "invalid_status"])
def test_merge_rejects_incomplete_duplicate_or_invalid_shard(tmp_path, monkeypatch, corruption):
    directory, protocol, ids, _ = setup_run(tmp_path, monkeypatch)
    path = directory / "scores_0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "missing_file":
        path.unlink()
    else:
        if corruption == "missing_row":
            rows.pop()
        elif corruption == "duplicate":
            rows.append(rows[0])
        elif corruption == "foreign_shard":
            rows[0]["id"] = ids[1]
        elif corruption == "mixed_protocol":
            rows[0]["protocol_fingerprint"] = "other"
        elif corruption == "missing_scores":
            del rows[0]["scores"]
        else:
            rows[0]["status"] = "error"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        launcher.merge(protocol, directory, 2)
    assert not (directory / "scores.jsonl").exists()
    assert not (directory / "scores.completion.json").exists()


@pytest.mark.parametrize("corruption", ["missing", "unfinished", "wrong_protocol", "answers_used",
                                         "hash", "modified_valid_score"])
def test_merge_requires_completion_and_exact_shard_hash(tmp_path, monkeypatch, corruption):
    directory, protocol, _, _ = setup_run(tmp_path, monkeypatch)
    path = directory / "scores_1.jsonl"
    sidecar = path.with_suffix(".completion.json")
    completion = json.loads(sidecar.read_text())
    if corruption == "missing":
        sidecar.unlink()
    elif corruption == "modified_valid_score":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["scores"]["direct"][0] = 1.0
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        key, value = {"unfinished": ("complete", False), "wrong_protocol": ("protocol_fingerprint", "other"),
                      "answers_used": ("test_answers_loaded", True), "hash": ("scores_sha256", "wrong")}[corruption]
        completion[key] = value
        sidecar.write_text(json.dumps(completion))
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.merge(protocol, directory, 2)
    assert not (directory / "scores.jsonl").exists()


def test_merge_preserves_existing_different_combined_output(tmp_path, monkeypatch):
    directory, protocol, _, _ = setup_run(tmp_path, monkeypatch)
    result = directory / "scores.jsonl"
    result.write_text("existing unrelated result\n")
    with pytest.raises(ValueError, match="existing combined scores differ"):
        launcher.merge(protocol, directory, 2)
    assert result.read_text() == "existing unrelated result\n"


def test_merge_only_cli_does_not_start_workers_or_query_gpu(tmp_path, monkeypatch):
    directory, protocol, _, _ = setup_run(tmp_path, monkeypatch)

    def forbidden(*_, **__):
        raise AssertionError("merge-only must not use subprocesses or GPU")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    import torch
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    assert launcher.main(["--protocol", str(protocol), "--output-dir", str(directory),
                          "--workers", "2", "--merge-only"]) == 0


@pytest.mark.parametrize("mode", ["too_many_workers", "zero_workers", "zero_threads", "benchmark_dir"])
def test_cli_rejects_invalid_worker_or_destination_settings(tmp_path, monkeypatch, mode):
    directory, protocol, _, record = setup_run(tmp_path, monkeypatch)
    workers, threads = 2, 2
    if mode == "too_many_workers":
        workers = 6
    elif mode == "zero_workers":
        workers = 0
    elif mode == "zero_threads":
        threads = 0
    else:
        directory = tmp_path / "bench" / "run"
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--protocol", str(protocol), "--output-dir", str(directory),
                       "--workers", str(workers), "--threads", str(threads), "--merge-only"])
    assert exc.value.code == 2
