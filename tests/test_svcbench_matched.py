"""Protocol-level tests for the matched real-video SVCBench evaluation."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_svcbench_matched as matched_runner
from scripts.evaluate_svcbench_matched import (
    CONDITIONS,
    SCHEMA,
    _chunks,
    _processor_frame_count,
    _timed_question,
    _uniform_prefix,
)
from scripts.summarize_svcbench_matched import (
    COMPARISON_PROTOCOL_FIELDS,
    ValidationError,
    _trajectory_metrics_strict,
    _validate_and_summarize,
    write_report,
)


def _refresh_comparison_fingerprint(protocol: dict) -> None:
    payload = {field: protocol[field] for field in COMPARISON_PROTOCOL_FIELDS}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    protocol["comparison_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()


def _matched_fixture() -> tuple[list[dict], list[dict], dict]:
    video_key = "Ego4D/fixture.mp4"
    video_hash = "a" * 64
    annotation = [{
        "id": "q1",
        "source_dataset": "Ego4D",
        "video_path": "fixture.mp4",
        "question": "How many doors has the wearer passed through so far?",
        "counting_subtype": "E1-Act",
        "counting_subtype_normalized": "E1-Action",
        "query_points": [
            {"q_id": "p0", "query_index": 0, "query_time": 5.0, "count": 0},
            {"q_id": "p1", "query_index": 1, "query_time": 10.0, "count": 2},
        ],
    }]
    selection = {
        "schema": "ttt_frame.svcbench_real_selection/1",
        "name": "unit fixture",
        "dataset_repo": "fixture/SVCBench",
        "dataset_revision": "fixture-revision",
        "annotation_sha256": "b" * 64,
        "videos": [{
            "video": video_key,
            "sha256": video_hash,
            "question_ids": ["q1"],
            "question_point_counts": {"q1": 2},
            "semantic_families": {"q1": "door_passage"},
        }],
        "n_videos": 1,
        "n_questions": 1,
        "n_semantic_families": 1,
        "n_query_points": 2,
    }
    protocol = {
        "schema": SCHEMA,
        "runner_sha256": "c" * 64,
        "annotations_sha256": "d" * 64,
        "model_artifact_sha256": "e" * 64,
        "base_config_fingerprint": "fixture-base",
        "spatial_checkpoint_sha256": "f" * 64,
        "conditions": list(CONDITIONS),
        "dtype": "bfloat16",
        "max_new_tokens": 8,
        "sampling": {
            "max_frames": 4,
            "write_chunk_frames": 4,
            "cutoff_margin": 0.0,
            "query_stride": 1,
            "max_query_points": 0,
        },
        "num_shards": 1,
        "shard_index": 0,
        "device": "cuda:0",
    }
    _refresh_comparison_fingerprint(protocol)

    rows = []
    for point in annotation[0]["query_points"]:
        timestamps = [0.0, point["query_time"] - 1.0]
        prompt = _timed_question(
            annotation[0]["question"],
            query_time=point["query_time"],
            cutoff_time=point["query_time"],
        )
        common = {
            "schema": SCHEMA,
            "episode_id": f"{video_key}::q1::{point['q_id']}",
            "query_episode_independent": True,
            "question_id": "q1",
            "q_id": point["q_id"],
            "query_index": point["query_index"],
            "source_dataset": "Ego4D",
            "video_path": "fixture.mp4",
            "video_file": "/tmp/fixture.mp4",
            "video_sha256": video_hash,
            "question": annotation[0]["question"],
            "prompt": prompt,
            "counting_subtype": "E1-Action",
            "query_time": point["query_time"],
            "cutoff_time": point["query_time"],
            "gold": point["count"],
            "sampled_prefix_frame_count": 2,
            "selected_frame_count": 2,
            "selected_sample_indices": [0, 1],
            "frame_timestamps": timestamps,
            "frame_bundle_sha256": (point["q_id"][-1] or "0") * 64,
            "future_frames_filtered": True,
            "max_decoded_timestamp": timestamps[-1],
            "run_protocol": protocol,
        }
        for condition in CONDITIONS:
            spatial = condition.startswith("spatial")
            visual = condition == "icl"
            rows.append({
                **common,
                "condition": condition,
                "answer": str(point["count"]),
                "prediction": point["count"],
                "correct": True,
                "status": "ok",
                "error": None,
                "elapsed_sec": 0.01,
                "visual_frames_provided_to_answer": 2 if visual else 0,
                "visual_frames_encoded": 4 if visual else None,
                "visual_padding_frames": 2 if visual else None,
                "visual_encode_attempted": visual,
                "spatial_frames_ingested": 2 if spatial else 0,
                "spatial_frames_encoded": 4 if spatial else 0,
                "spatial_padding_frames": 2 if spatial else 0,
                "visual_encode_passes": 1 if visual else 0,
                "spatial_write_calls": 1 if spatial else 0,
                "spatial_write_calls_planned": 1 if spatial else 0,
                "shared_spatial_write": spatial,
                "episode_memory_state": {
                    "blind": "no_visual_input_at_position_zero",
                    "icl": "matched_visual_context",
                    "spatial": "episode_fast_state",
                    "spatial_slow_only": "official_learned_initial_state",
                    "spatial_base": "spatial_branch_bypassed_after_ingest",
                }[condition],
                "episode_elapsed_sec": 0.1,
            })
    return rows, annotation, selection


def test_uniform_prefix_is_deterministic_future_safe_and_capped():
    images = [f"frame-{index}" for index in range(10)]
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0001, 9.0]

    first = _uniform_prefix(images, timestamps, cutoff=7.0, max_frames=4)
    second = _uniform_prefix(images, timestamps, cutoff=7.0, max_frames=4)

    assert first == second
    selected_images, selected_times, selected_indices, eligible_count = first
    assert eligible_count == 8
    assert len(selected_images) == len(selected_times) == len(selected_indices) == 4
    assert selected_indices == sorted(set(selected_indices))
    assert selected_indices[0] == 0
    assert selected_indices[-1] == 7
    assert all(timestamp <= 7.0 for timestamp in selected_times)
    assert "frame-8" not in selected_images


def test_spatial_write_chunks_merge_a_short_temporal_tail():
    chunks = list(_chunks(list(range(34)), 8))
    assert [len(chunk) for chunk in chunks] == [8, 8, 8, 10]
    assert [item for chunk in chunks for item in chunk] == list(range(34))


@pytest.mark.parametrize("size", [0, 1, 2, 3, 5, 7, 9])
def test_spatial_write_chunks_reject_odd_or_too_small_sizes(size):
    with pytest.raises(ValueError, match="even size of at least four"):
        list(_chunks(list(range(12)), size))


@pytest.mark.parametrize(
    ("real_frames", "encoded_frames"),
    [(0, 0), (1, 4), (2, 4), (3, 4), (4, 4), (5, 6), (6, 6), (7, 8), (8, 8)],
)
def test_processor_frame_count_matches_qwen_temporal_padding(real_frames, encoded_frames):
    assert _processor_frame_count(real_frames) == encoded_frames


def test_decode_until_filters_the_boundary_chunk_and_stops_early(monkeypatch, tmp_path):
    visited: list[int] = []
    chunks = [
        SimpleNamespace(index=0, images=["a", "b"], timestamps=[0.0, 3.9]),
        SimpleNamespace(index=1, images=["c", "future"], timestamps=[4.0, 4.5]),
        SimpleNamespace(index=2, images=["far-future"], timestamps=[8.0]),
    ]

    def fake_chunks(*args, **kwargs):
        del args, kwargs
        for chunk in chunks:
            visited.append(chunk.index)
            yield chunk

    monkeypatch.setattr(matched_runner, "iter_video_chunks", fake_chunks)
    images, timestamps = matched_runner._decode_until(
        tmp_path / "unused.mp4",
        stop_time=4.2,
        chunk_seconds=4.0,
        frames_per_chunk=8,
        max_side=224,
    )

    assert images == ["a", "b", "c"]
    assert timestamps == [0.0, 3.9, 4.0]
    assert max(timestamps) <= 4.2
    assert visited == [0, 1]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_bundle_sha256", "9" * 64, "mismatched prefix"),
        ("frame_timestamps", [0.0, 3.0], "mismatched prefix"),
        ("query_time", 5.5, "query_time differs from annotation"),
    ],
)
def test_summary_rejects_cross_arm_or_annotation_mismatch(field, value, message):
    rows, annotation, selection = _matched_fixture()
    target = next(
        row for row in rows
        if row["condition"] == "icl" and row["q_id"] == "p0"
    )
    target[field] = value

    with pytest.raises(ValidationError, match=message):
        _validate_and_summarize(rows, annotation, selection)


def test_summary_rejects_a_selected_frame_after_the_query_cutoff():
    rows, annotation, selection = _matched_fixture()
    for row in rows:
        if row["q_id"] == "p0":
            row["frame_timestamps"] = [0.0, 5.01]

    with pytest.raises(ValidationError, match="frame after cutoff_time"):
        _validate_and_summarize(rows, annotation, selection)


def test_summary_validates_optional_selection_question_point_counts():
    rows, annotation, selection = _matched_fixture()
    selection["videos"][0]["question_point_counts"]["q1"] = 3

    with pytest.raises(ValidationError, match=r"question_point_counts\['q1'\].*contains 2"):
        _validate_and_summarize(rows, annotation, selection)

    del selection["videos"][0]["question_point_counts"]
    summary = _validate_and_summarize(rows, annotation, selection)
    assert summary["selection"]["n_query_points"] == 2


def test_official_metric_subtype_applicability_is_preserved():
    snapshot = _trajectory_metrics_strict([0, 1], [0, 1], "O1-Snapshot")
    action = _trajectory_metrics_strict([0, 1], [0, 1], "E1-Act")
    unknown = _trajectory_metrics_strict([0, 1], [0, 1], "Unknown")

    assert snapshot["counting_subtype"] == "O1-Snap"
    assert snapshot["gpa"] == pytest.approx(1.0)
    assert snapshot["moc_applicable"] is False
    assert snapshot["moc"] is None
    assert snapshot["uda_applicable"] is True
    assert snapshot["uda"] == pytest.approx(1.0)

    assert action["counting_subtype"] == "E1-Action"
    assert action["moc_applicable"] is True
    assert action["moc"] == pytest.approx(1.0)
    assert action["uda_applicable"] is True
    assert action["uda"] == pytest.approx(1.0)

    assert unknown["moc_applicable"] is False
    assert unknown["moc"] is None
    assert unknown["uda_applicable"] is False
    assert unknown["uda"] is None


def test_parse_failure_stays_aligned_and_counts_as_incorrect():
    metric = _trajectory_metrics_strict([0, None, 2], [0, 1, 2], "E1-Action")
    assert metric["n_queries"] == 3
    assert metric["n_valid"] == 2
    assert metric["n_invalid"] == 1
    assert metric["coverage"] == pytest.approx(2 / 3)
    assert metric["exact_accuracy_all_queries"] == pytest.approx(2 / 3)
    assert metric["official_complete_trajectory"] is False
    assert metric["gpa"] is None
    assert metric["moc"] is None
    assert metric["uda"] is None
    assert metric["gpa_failure_zero"] == pytest.approx(2 / 3)

    rows, annotation, selection = _matched_fixture()
    failed = next(
        row for row in rows
        if row["condition"] == "spatial" and row["q_id"] == "p0"
    )
    failed.update({
        "answer": "unable to determine",
        "prediction": None,
        "correct": False,
        "status": "parse_error",
        "error": None,
    })
    summary = _validate_and_summarize(rows, annotation, selection)
    spatial = summary["by_condition"]["spatial"]
    trajectory = summary["trajectories"]["spatial"]["q1"]

    assert spatial["n_queries"] == 2
    assert spatial["n_parse_or_runtime_failures"] == 1
    assert spatial["coverage_query_micro"] == pytest.approx(0.5)
    assert spatial["exact_accuracy_query_micro"] == pytest.approx(0.5)
    assert trajectory["n_invalid"] == 1
    assert trajectory["gpa"] is None
    assert trajectory["gpa_failure_zero"] == pytest.approx(0.5)
    calibration = spatial["count_calibration"]
    assert calibration["n_valid"] == 1
    assert calibration["n_invalid"] == 1
    assert calibration["coverage"] == pytest.approx(0.5)
    assert calibration["mae"] is None
    assert calibration["mae_valid_only_diagnostic"] == pytest.approx(0.0)
    assert calibration["prediction_min"] == 2
    assert calibration["prediction_max"] == 2
    assert calibration["prediction_histogram"] == {"2": 1}


def test_count_calibration_reports_complete_arm_mae_range_and_histogram(tmp_path):
    rows, annotation, selection = _matched_fixture()
    changed = next(
        row for row in rows
        if row["condition"] == "blind" and row["q_id"] == "p0"
    )
    changed.update({"answer": "3", "prediction": 3, "correct": False})

    summary = _validate_and_summarize(rows, annotation, selection)
    calibration = summary["by_condition"]["blind"]["count_calibration"]

    assert calibration == {
        "n_queries": 2,
        "n_valid": 2,
        "n_invalid": 0,
        "coverage": 1.0,
        "mae": 1.5,
        "mae_requires_complete_coverage": True,
        "mae_valid_only_diagnostic": 1.5,
        "prediction_statistics_scope": "valid_predictions_only",
        "prediction_min": 2,
        "prediction_max": 3,
        "prediction_histogram": {"2": 1, "3": 1},
    }
    assert summary["by_condition"]["blind"]["exact_accuracy_query_micro"] == pytest.approx(0.5)

    report = tmp_path / "calibration.md"
    write_report(report, summary)
    text = report.read_text(encoding="utf-8")
    assert "## Count calibration (descriptive)" in text
    assert "| blind | 2 / 2 | 1.0000 | 1.5000 | 2 | 3 | `{" in text
    assert "mae_valid_only_diagnostic" in text


def test_summary_accepts_the_runners_icl_encode_error_budget():
    rows, annotation, selection = _matched_fixture()
    failed = next(
        row for row in rows
        if row["condition"] == "icl" and row["q_id"] == "p0"
    )
    failed.update({
        "answer": "",
        "prediction": None,
        "correct": False,
        "status": "error",
        "error": "RuntimeError: visual encoding failed",
        "visual_frames_encoded": None,
        "visual_padding_frames": None,
        "visual_encode_passes": 0,
        "visual_encode_attempted": True,
    })

    summary = _validate_and_summarize(rows, annotation, selection)
    assert summary["by_condition"]["icl"]["n_runtime_errors"] == 1
    assert summary["by_condition"]["icl"]["exact_accuracy_query_micro"] == pytest.approx(0.5)

    failed["visual_frames_encoded"] = 4
    with pytest.raises(ValidationError, match="visual_frames_encoded"):
        _validate_and_summarize(rows, annotation, selection)


def test_summary_audits_failed_spatial_writes_as_one_shared_budget():
    rows, annotation, selection = _matched_fixture()
    protocol = rows[0]["run_protocol"]
    protocol["sampling"]["max_frames"] = 32
    protocol["sampling"]["write_chunk_frames"] = 8
    _refresh_comparison_fingerprint(protocol)
    timestamps = [4.9 * index / 18 for index in range(19)]
    for row in rows:
        if row["q_id"] != "p0":
            continue
        row.update({
            "sampled_prefix_frame_count": 19,
            "selected_frame_count": 19,
            "selected_sample_indices": list(range(19)),
            "frame_timestamps": timestamps,
            "max_decoded_timestamp": timestamps[-1],
        })
        if row["condition"] == "icl":
            row.update({
                "visual_frames_provided_to_answer": 19,
                "visual_frames_encoded": 20,
                "visual_padding_frames": 1,
            })

    failed_rows = [
        row for row in rows
        if row["condition"].startswith("spatial") and row["q_id"] == "p0"
    ]
    for row in failed_rows:
        row.update({
            "answer": "",
            "prediction": None,
            "correct": False,
            "status": "error",
            "error": "RuntimeError: spatial write failed",
            "spatial_frames_ingested": 8,
            "spatial_frames_encoded": 8,
            "spatial_padding_frames": 0,
            "spatial_write_calls": 1,
            "spatial_write_calls_planned": 3,
            "shared_spatial_write": True,
        })

    summary = _validate_and_summarize(rows, annotation, selection)
    for condition in ("spatial", "spatial_slow_only", "spatial_base"):
        assert summary["by_condition"][condition]["n_runtime_errors"] == 1

    failed_rows[-1].update({
        "spatial_frames_ingested": 16,
        "spatial_frames_encoded": 16,
        "spatial_padding_frames": 0,
        "spatial_write_calls": 2,
        "shared_spatial_write": True,
    })
    with pytest.raises(ValidationError, match="do not share the same completed write budget"):
        _validate_and_summarize(rows, annotation, selection)


def test_summary_preserves_official_order_and_reports_chronological_uda_sensitivity(
    tmp_path,
):
    rows, annotation, selection = _matched_fixture()
    point = {"q_id": "p2", "query_index": 2, "query_time": 7.5, "count": 1}
    annotation[0]["query_points"].append(point)
    selection["n_query_points"] = 3
    selection["videos"][0]["question_point_counts"]["q1"] = 3
    selection["source_type"] = "mixed-source real-image video fixture"

    for row in rows:
        if row["q_id"] == "p1":
            row.update({"answer": "0", "prediction": 0, "correct": False})

    template_rows = [row for row in rows if row["q_id"] == "p1"]
    for template in template_rows:
        row = deepcopy(template)
        row.update({
            "episode_id": "Ego4D/fixture.mp4::q1::p2",
            "q_id": "p2",
            "query_index": 2,
            "query_time": 7.5,
            "cutoff_time": 7.5,
            "gold": 1,
            "prompt": _timed_question(
                annotation[0]["question"], query_time=7.5, cutoff_time=7.5,
            ),
            "frame_timestamps": [0.0, 6.5],
            "frame_bundle_sha256": "2" * 64,
            "max_decoded_timestamp": 6.5,
            "answer": "1",
            "prediction": 1,
            "correct": True,
        })
        rows.append(row)

    summary = _validate_and_summarize(rows, annotation, selection)
    sensitivity = summary["trajectory_order_sensitivity"]

    assert summary["by_condition"]["spatial"]["official_metrics"][
        "uda_question_macro"
    ] == pytest.approx(0.0)
    assert sensitivity["primary_metrics_unchanged"] is True
    assert sensitivity["n_order_mismatch_trajectories"] == 1
    assert sensitivity["order_mismatch_question_ids"] == ["q1"]
    mismatch = sensitivity["order_mismatches"]["q1"]
    assert [point["query_time"] for point in mismatch["official_query_index_order"]] == [
        5.0, 10.0, 7.5,
    ]
    assert [point["query_time"] for point in mismatch["chronological_query_time_order"]] == [
        5.0, 7.5, 10.0,
    ]
    for condition in CONDITIONS:
        metric = sensitivity["by_condition"][condition]
        assert metric["official_uda_question_macro"] == pytest.approx(0.0)
        assert metric["chronological_uda_question_macro"] == pytest.approx(0.5)
        assert metric["chronological_minus_official"] == pytest.approx(0.5)
        assert metric["order_mismatch_details"]["q1"] == {
            "official_uda": 0.0,
            "chronological_uda": 0.5,
            "chronological_minus_official": 0.5,
        }

    report = tmp_path / "report.md"
    write_report(report, summary)
    text = report.read_text(encoding="utf-8")
    assert "real Ego4D videos" not in text
    assert "mixed-source real-image video fixture" in text
    assert "Trajectories whose two orders differ: **1**" in text
    assert "## Count calibration (descriptive)" in text
    assert (
        "| q1 | Ego4D/fixture.mp4 | 5.00, 10.00, 7.50 | 5.00, 7.50, 10.00 |"
        in text
    )
