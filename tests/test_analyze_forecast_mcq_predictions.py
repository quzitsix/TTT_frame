"""Prediction-only evidence audits, independent of benchmark answers or media."""

from copy import deepcopy
import json
import math

import pytest

from scripts.analyze_forecast_mcq_predictions import ARMS, analyze, main


def rows_for_case(case_id="egolife_001"):
    audit = {"decoded": True, "audio_used": False, "sample_id": case_id,
             "frame_plan_sha256": "a" * 64, "frame_bundle_sha256": "b" * 64,
             "context_frame_bundle_sha256": "c" * 64, "recent_frame_bundle_sha256": "d" * 64}
    write = {"stats": {"frames": 90, "visual_tokens": 2205}, "ttt_layers": [2, 3],
             "updates_per_layer": {"2": 12, "3": 12}, "parameter_only": True,
             "retained_video_frames": 0, "retained_text_records": 0,
             "trained_checkpoint_loaded": True}
    rows = []
    for arm in ARMS:
        logits = [1, 0, 0, 0, 0, 0]
        if arm == "spatial":
            logits = [0, 2, 0, 0, 0, 0]
        rows.append({"schema": "forecast_mcq_prediction_v1", "id": case_id, "arm": arm,
                     "status": "ok", "protocol_fingerprint": "f" * 64,
                     "sampling": deepcopy(audit), "label_logits": logits,
                     "prediction": "B" if arm == "spatial" else "A",
                     "write_report": deepcopy(write) if arm.startswith("spatial") else None})
    return rows


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_logit_differences_and_write_audits_without_any_dataset(tmp_path):
    rows = rows_for_case()
    path = tmp_path / "only_predictions.jsonl"
    write(path, rows)
    report = analyze([path], expected_cases=1)
    assert report["all_checks_passed"] and report["complete"]
    assert report["answers_read"] is False
    assert report["frame_bundle_checks"]["passed_cases"] == 1
    assert report["spatial_write_checks"]["passed_cases"] == 1
    for comparison in report["spatial_logit_comparisons"].values():
        assert comparison["compared_cases"] == 1 and comparison["label_flips"] == 1
        assert comparison["max_abs_logit_difference_max"] == 2
        assert comparison["l2_logit_difference_max"] == pytest.approx(math.sqrt(5))
        assert comparison["nonzero_logit_difference_cases"] == 1


def test_changed_logits_without_label_flip_is_reported(tmp_path):
    rows = rows_for_case()
    fast = next(row for row in rows if row["arm"] == "spatial")
    fast["label_logits"], fast["prediction"] = [1.5, 0, 0, 0, 0, 0], "A"
    path = tmp_path / "pred.jsonl"
    write(path, rows)
    report = analyze([path], expected_cases=1)
    comparison = report["spatial_logit_comparisons"]["spatial_slow_only"]
    assert comparison["label_flips"] == 0
    assert comparison["nonzero_logit_difference_cases"] == 1
    assert comparison["max_abs_logit_difference_mean"] == 0.5
    assert report["all_checks_passed"]


def test_hash_mismatch_between_families_fails_audit(tmp_path):
    rows = rows_for_case()
    for row in rows:
        if row["arm"].startswith("qwen_"):
            row["sampling"]["frame_bundle_sha256"] = "e" * 64
    path = tmp_path / "pred.jsonl"
    write(path, rows)
    report = analyze([path], expected_cases=1)
    assert report["complete"] and not report["all_checks_passed"]
    assert "frame hash mismatch: frame_bundle_sha256" in report["frame_bundle_checks"]["cases"][0]["reasons"]


@pytest.mark.parametrize("mutation, reason", [
    ("frames", "90 real frames"), ("tokens", "2205 visual tokens"),
    ("updates", "positive updates"), ("mismatch", "same completed write report"),
])
def test_write_budget_and_updates_checked(tmp_path, mutation, reason):
    rows = rows_for_case()
    for row in rows:
        if row["write_report"] is None:
            continue
        if mutation == "frames":
            row["write_report"]["stats"]["frames"] = 88
        elif mutation == "tokens":
            row["write_report"]["stats"]["visual_tokens"] = 2100
        elif mutation == "updates":
            row["write_report"]["updates_per_layer"]["2"] = 0
        elif row["arm"] == "spatial_base":
            row["write_report"]["updates_per_layer"]["2"] = 13
    path = tmp_path / "pred.jsonl"
    write(path, rows)
    report = analyze([path], expected_cases=1)
    assert not report["all_checks_passed"]
    assert any(reason in message for message in report["spatial_write_checks"]["cases"][0]["reasons"])


def test_partial_coverage_not_passed_and_cli_writes_diagnostic_failure(tmp_path):
    path = tmp_path / "pred.jsonl"
    write(path, rows_for_case()[:-1])
    output = tmp_path / "diagnostics.json"
    assert main(["--predictions", str(path), "--output", str(output), "--expected-cases", "1"]) == 2
    report = json.loads(output.read_text())
    assert not report["complete"] and not report["all_checks_passed"]
    assert report["missing_arms_by_observed_id"] == {"egolife_001": ["spatial_base"]}
    with pytest.raises(SystemExit):
        main(["--predictions", str(path), "--output", str(output)])


def test_duplicate_and_mixed_protocol_rejected(tmp_path):
    path = tmp_path / "pred.jsonl"
    rows = rows_for_case()
    write(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        analyze([path], expected_cases=1)
    rows[0]["protocol_fingerprint"] = "e" * 64
    write(path, rows)
    with pytest.raises(ValueError, match="consistent protocol"):
        analyze([path], expected_cases=1)


def test_recorded_label_must_match_logits_and_invalid_logits_excluded(tmp_path):
    rows = rows_for_case()
    next(row for row in rows if row["arm"] == "spatial")["prediction"] = "A"
    next(row for row in rows if row["arm"] == "spatial_base")["label_logits"] = [float("nan")] * 6
    path = tmp_path / "pred.jsonl"
    write(path, rows)
    report = analyze([path], expected_cases=1)
    assert not report["all_checks_passed"]
    assert report["spatial_logit_comparisons"]["spatial_slow_only"]["argmax_prediction_mismatch_ids"] == ["egolife_001"]
    assert report["spatial_logit_comparisons"]["spatial_base"]["missing_or_invalid_logit_ids"] == ["egolife_001"]
