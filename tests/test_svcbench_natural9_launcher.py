"""Tests for the fail-closed Natural9 multi-GPU launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_svcbench_natural9_multigpu import (  # noqa: E402
    CONDITIONS,
    SUMMARY_SCHEMA,
    ShardPaths,
    _completion_status,
    _parse_gpu_list,
)


def _write_complete_fixture(tmp_path: Path) -> tuple[ShardPaths, dict]:
    paths = ShardPaths(
        output=tmp_path / "shard3.jsonl",
        summary=tmp_path / "shard3_summary.json",
        log=tmp_path / "logs" / "shard3.log",
    )
    fingerprint = "f" * 64
    expected_protocol = {
        "runner_sha256": "a" * 64,
        "num_shards": 9,
        "shard_index": 3,
        "sampling": {"max_frames": 512},
    }
    protocol = {**expected_protocol, "comparison_fingerprint": fingerprint}
    rows = [
        {
            "episode_id": "video::question::point",
            "condition": condition,
            "status": "ok",
            "run_protocol": {
                "comparison_fingerprint": fingerprint,
                "num_shards": 9,
                "shard_index": 3,
            },
        }
        for condition in CONDITIONS
    ]
    paths.output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    digest = hashlib.sha256(paths.output.read_bytes()).hexdigest()
    paths.summary.write_text(
        json.dumps({
            "schema": SUMMARY_SCHEMA,
            "run_protocol": protocol,
            "n_rows": len(rows),
            "expected_rows": len(rows),
            "output_sha256": digest,
        }),
        encoding="utf-8",
    )
    return paths, expected_protocol


def test_gpu_list_requires_unique_non_negative_integer_indices():
    assert _parse_gpu_list("0, 1,7") == [0, 1, 7]
    for invalid in ("", "0,,1", "0,0", "-1", "gpu0"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_gpu_list(invalid)


def test_completed_shard_requires_summary_hash_rows_protocol_and_five_arms(tmp_path):
    paths, expected = _write_complete_fixture(tmp_path)
    status, reason = _completion_status(paths, expected)
    assert status == "complete"
    assert "five-arm" in reason


def test_partial_or_modified_shard_is_blocked_instead_of_overwritten(tmp_path):
    paths, expected = _write_complete_fixture(tmp_path)
    paths.summary.unlink()
    status, reason = _completion_status(paths, expected)
    assert status == "blocked"
    assert "partial artifact" in reason

    paths, expected = _write_complete_fixture(tmp_path)
    with paths.output.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    status, reason = _completion_status(paths, expected)
    assert status == "blocked"
    assert "SHA-256" in reason


def test_runtime_error_or_protocol_change_prevents_skip(tmp_path):
    paths, expected = _write_complete_fixture(tmp_path)
    rows = [json.loads(line) for line in paths.output.read_text().splitlines()]
    rows[0]["status"] = "error"
    paths.output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    summary = json.loads(paths.summary.read_text())
    summary["output_sha256"] = hashlib.sha256(paths.output.read_bytes()).hexdigest()
    paths.summary.write_text(json.dumps(summary), encoding="utf-8")
    status, reason = _completion_status(paths, expected)
    assert status == "blocked"
    assert "runtime-error" in reason

    paths, expected = _write_complete_fixture(tmp_path)
    expected["num_shards"] = 8
    status, reason = _completion_status(paths, expected)
    assert status == "blocked"
    assert "protocol mismatch" in reason
