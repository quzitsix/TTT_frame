"""Evidence-match audit rejects partial or altered experiment inputs."""
from copy import deepcopy
import json

import pytest

from scripts import audit_idea3_frames as runner


def write_rows(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.fixture
def inputs(tmp_path):
    baseline = tmp_path / "baseline"
    (baseline / "eval").mkdir(parents=True)
    sampling = {"context_frames": 2, "recent_frames": 2, "max_side": 224}
    payload = {"case_ids": ["a", "b"], "sampling": sampling,
               "root": str(tmp_path / "benchmark")}
    fingerprint = runner.canonical_digest(payload)
    (baseline / "protocol.json").write_text(json.dumps({"payload": payload,
        "protocol_fingerprint": fingerprint}))
    records, originals = [], []
    for sid, role in (("a", "validation"), ("b", "test")):
        bundle = {"sample_id": sid, "decoded": True, "audio_used": False,
                  "sampling": {**sampling, "cutoff_prefix_seconds": 4.0},
                  **{key: "a" * 64 for key in runner.HASH_KEYS},
                  "context_requested_seconds": [0., 1.], "context_actual_seconds": [0., 1.],
                  "recent_requested_seconds": [2., 3.], "recent_actual_seconds": [2., 3.]}
        row = {"id": sid, "status": "ok", "protocol_fingerprint": fingerprint,
               "sampling": bundle, "arm": "qwen_full_icl"}
        originals.append(row)
        records.append({"id": sid, "status": "ok", "answer_keys_used": False,
            "research_split": role, "protocol_fingerprint": "b" * 64, "sampling": deepcopy(bundle),
            "elapsed_seconds": 10., "decode_seconds": 2., "extraction_seconds": 1.,
            "cuda_peak_allocated_bytes": 1000, "device": "cuda:0",
            "write_report": {"stats": {"ingest_seconds": 1.}},
            "search_seconds": {key: .5 for key in runner.SEARCH_KEYS},
            "visual_key_diagnostic": {"visual_key": 3., "text_key": 1.,
                                      "inverse_text_key": 2., "inverse_text_objective": 1.9}})
    raw = baseline / "eval/qwen_shard00.jsonl"
    write_rows(raw, originals)
    scores = tmp_path / "scores.jsonl"
    write_rows(scores, records)
    return scores, baseline, tmp_path / "audit.json", raw


def test_complete_audit_and_runtime_without_any_answer_files(inputs):
    scores, baseline, output, _ = inputs
    result = runner.audit(scores, baseline, output)
    assert result["complete_benchmark_coverage"] is True
    assert result["matched_cases"] == 2
    assert result["answer_files_read"] is False
    assert result["runtime"]["all"]["elapsed_seconds"]["sum"] == 20
    assert result["runtime"]["test"]["all_search_seconds"]["sum"] == 3
    assert not any(key in result["runtime"]["all"] for key in runner.PROBE_TIME_KEYS)
    assert result["visual_key_diagnostic"]["all"]["gaps"]["visual_minus_text"]["mean"] == 2
    assert "not descriptions of the same fact" in result["visual_diagnostic_caveat"]
    assert result["score_file"]["sha256"] == runner.sha256_file(scores)
    with pytest.raises(FileExistsError):
        runner.audit(scores, baseline, output)


@pytest.mark.parametrize("change", ["pixels", "time", "protocol", "duplicate", "missing", "timing"])
def test_rejects_mismatched_incomplete_and_invalid_scores(inputs, change):
    scores, baseline, output, _ = inputs
    records = runner.rows(scores)
    if change == "pixels":
        records[0]["sampling"]["frame_bundle_sha256"] = "c" * 64
    elif change == "time":
        records[0]["sampling"]["recent_actual_seconds"][0] = 1.99
    elif change == "protocol":
        records[0]["protocol_fingerprint"] = "d" * 64
    elif change == "duplicate":
        records.append(records[0])
    elif change == "missing":
        records.pop()
    else:
        records[0]["decode_seconds"] = -1
    write_rows(scores, records)
    with pytest.raises(ValueError):
        runner.audit(scores, baseline, output)
    assert not output.exists()


def test_reports_optional_probe_wall_times_by_split(inputs):
    scores, baseline, output, _ = inputs
    records = runner.rows(scores)
    for row in records:
        row.update(visual_probe_seconds=.25, reset_probe_seconds=.75)
    write_rows(scores, records)
    result = runner.audit(scores, baseline, output)
    assert result["runtime"]["all"]["visual_probe_seconds"]["sum"] == .5
    assert result["runtime"]["all"]["reset_probe_seconds"]["sum"] == 1.5
    assert result["runtime"]["test"]["visual_probe_seconds"]["mean"] == .25


@pytest.mark.parametrize("change", ["partial_rows", "partial_fields", "negative", "nonfinite"])
def test_rejects_partial_or_invalid_probe_wall_times(inputs, change):
    scores, baseline, output, _ = inputs
    records = runner.rows(scores)
    for row in records:
        row.update(visual_probe_seconds=.25, reset_probe_seconds=.75)
    if change == "partial_rows":
        for key in runner.PROBE_TIME_KEYS:
            del records[0][key]
    elif change == "partial_fields":
        for row in records:
            del row["reset_probe_seconds"]
    elif change == "negative":
        records[0]["visual_probe_seconds"] = -.25
    else:
        records[0]["reset_probe_seconds"] = float("nan")
    write_rows(scores, records)
    with pytest.raises(ValueError, match="probe|nonnegative|finite"):
        runner.audit(scores, baseline, output)
    assert not output.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_rejects_output_within_source_benchmark(inputs, symlink):
    scores, baseline, _, _ = inputs
    source = scores.parent / "benchmark"
    source.mkdir()
    destination = source
    if symlink:
        destination = scores.parent / "source_link"
        destination.symlink_to(source, target_is_directory=True)
    output = destination / "new_directory/audit.json"
    with pytest.raises(ValueError, match="outside source benchmark"):
        runner.audit(scores, baseline, output)
    assert not output.parent.exists()


def test_smoke_subset_is_explicit_and_cannot_claim_complete(inputs):
    scores, baseline, output, _ = inputs
    write_rows(scores, runner.rows(scores)[:1])
    result = runner.audit(scores, baseline, output, allow_subset=True)
    assert result["complete_benchmark_coverage"] is False
    assert result["missing_ids"] == ["b"]
    assert result["matched_cases"] == 1


@pytest.mark.parametrize("change", ["duplicate", "missing", "protocol", "shared_evidence"])
def test_rejects_corrupt_baseline_shards(inputs, change):
    scores, baseline, output, raw = inputs
    records = runner.rows(raw)
    if change == "duplicate":
        records.append(records[0])
    elif change == "missing":
        records.pop()
    elif change == "protocol":
        records[0]["protocol_fingerprint"] = "e" * 64
    else:
        changed = deepcopy(records[0])
        changed["arm"] = "qwen_recent_icl"
        changed["sampling"]["recent_frame_bundle_sha256"] = "f" * 64
        records.append(changed)
    write_rows(raw, records)
    with pytest.raises(ValueError):
        runner.audit(scores, baseline, output)
    assert not output.exists()
