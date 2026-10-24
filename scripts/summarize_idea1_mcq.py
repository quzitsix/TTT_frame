#!/usr/bin/env python3
"""Post-hoc Idea 01 scoring, paired controls and frozen VLM reference results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.summarize_forecast_mcq import summarize, _paired_scores, load_predictions
from scripts.evaluate_idea1_mcq import digest_json
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.idea1_mcq_memory import ARMS


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path,
                        default=Path("data/bench_v0_baseline_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists")
    completion = json.loads((args.run_dir / "completion.json").read_text())
    run_record = json.loads((args.run_dir / "protocol.json").read_text())
    protocol = run_record["protocol"]
    if (run_record["protocol_fingerprint"] != digest_json(protocol)
            or completion["protocol_fingerprint"] != run_record["protocol_fingerprint"]
            or protocol["split_sha256"] != file_sha(args.split_manifest)
            or protocol["input_manifest_sha256"] != file_sha(args.root / "inputs_mcq.jsonl")
            or run_record["shared_readout_sha256"] != file_sha(args.run_dir / "shared_readout.pt")):
        raise ValueError("training protocol, split, model or completion mismatch")
    predictions = args.run_dir / "predictions.jsonl"
    if (completion.get("complete") is not True
            or completion["predictions_sha256"] != file_sha(predictions)):
        raise ValueError("freeze every prediction before loading private test answers")
    for line in predictions.read_text().splitlines():
        if line and json.loads(line)["protocol_fingerprint"] != run_record["protocol_fingerprint"]:
            raise ValueError("prediction/training protocol mismatch")
    arms = list(ARMS) + ["text_prior"]
    if protocol.get("arms") != arms or completion.get("test_answers_loaded") is not False:
        raise ValueError("unexpected arms or test-answer use in prediction protocol")
    public_ids = [case.id for case in load_forecast_cases(args.root)]
    checked_rows, fingerprint = load_predictions([predictions], public_ids, arms)
    expected_keys = {(sid, arm) for sid in public_ids for arm in arms}
    if (set(checked_rows) != expected_keys or completion["expected_rows"] != len(expected_keys)
            or completion["rows"] != len(expected_keys)
            or fingerprint != run_record["protocol_fingerprint"]):
        raise ValueError("incomplete prediction set before loading private test answers")
    for row in checked_rows.values():
        scores = row.get("candidate_scores", {})
        if (set(scores) != set("ABCDEF") or any(
                isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) for value in scores.values())):
            raise ValueError("candidate scores must contain six finite values")
        best = max(scores.values())
        ties = sum(value == best for value in scores.values())
        if (type(row.get("tie_count")) is not int or row["tie_count"] != ties
                or row["prediction"] not in scores or scores[row["prediction"]] != best):
            raise ValueError("prediction or tie count inconsistent with candidate scores")
    report, exports = summarize(args.root, [predictions], arms, args.split_manifest)
    if not report["complete"] or report["inference_rows"] != completion["expected_rows"]:
        raise ValueError("incomplete prediction set")
    paired_controls = {}
    for arm in arms:
        paired_controls[arm] = {}
        for control in arms:
            if arm == control:
                continue
            pairs = list(zip(report["by_arm"][arm]["cases"],
                             report["by_arm"][control]["cases"]))
            paired_controls[arm][control] = {
                role: _paired_scores([(a, b) for a, b in pairs if a["research_split"] == role])
                for role in ("train", "validation", "test")
            }
    diagnostics = {}
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    by_key = {(row["arm"], row["id"]): row for row in rows}
    test_cases = [row for row in report["by_arm"]["residual"]["cases"]
                  if row["research_split"] == "test"]
    test_ids = [row["id"] for row in test_cases]
    for arm in arms:
        differences = [max(abs(by_key[arm, sample_id]["candidate_scores"][label]
                               - by_key["recent_only", sample_id]["candidate_scores"][label])
                           for label in "ABCDEF") for sample_id in test_ids]
        diagnostics[arm] = {
            "test_cases": len(test_ids),
            "max_score_delta_from_recent_only": max(differences),
            "mean_max_score_delta_from_recent_only": sum(differences) / len(differences),
            "test_tied_maxima": sum(by_key[arm, sid]["tie_count"] > 1 for sid in test_ids),
        }
    baseline_metrics = json.loads((args.baseline_archive / "aggregate_metrics.json").read_text())
    if baseline_metrics["inputs_sha256"] != report["inputs_sha256"]:
        raise ValueError("VLM baseline/public manifest mismatch")
    baseline_split = json.loads((args.baseline_archive / "research_day_split.json").read_text())
    split = json.loads(args.split_manifest.read_text())
    if baseline_split["splits"] != split["splits"]:
        raise ValueError("VLM baseline and Idea 01 use different research splits")
    vlm_comparisons = {}
    for baseline, values in baseline_metrics["by_arm"].items():
        path = args.baseline_archive / f"{baseline}_predictions.jsonl"
        candidates = [json.loads(line) for line in path.read_text().splitlines() if line]
        frozen = {row["id"]: row["prediction"] for row in candidates}
        if (len(candidates) != report["sample_count"] or len(frozen) != len(candidates)
                or set(frozen) != set(public_ids)):
            raise ValueError("VLM reference predictions must be complete and unique")
        records = [{**row, "prediction": frozen[row["id"]],
                    "correct": frozen[row["id"]] == row["correct_answer"]} for row in test_cases]
        archived_test = values["by_research_split"]["test"]
        correct = sum(row["correct"] for row in records)
        if (type(archived_test["n"]) is not int or type(archived_test["correct"]) is not int
                or not isinstance(archived_test["accuracy_all"], (int, float))
                or isinstance(archived_test["accuracy_all"], bool)
                or not math.isfinite(archived_test["accuracy_all"])
                or archived_test["n"] != len(records) or archived_test["correct"] != correct
                or abs(archived_test["accuracy_all"] - correct / len(records)) > 1e-12):
            raise ValueError("VLM archived test metrics disagree with frozen predictions")
        vlm_comparisons[baseline] = {
            "test": values["by_research_split"]["test"],
            "residual_vs_reference_test": _paired_scores(list(zip(test_cases, records))),
            "prediction_sha256": file_sha(path),
            "matched_input_budget_or_training": False,
        }
    report.update({
        "schema": "idea01_mcq_results_v1", "primary_split": "test",
        "full_200_accuracy_is_held_out": False,
        "paired_controls": paired_controls, "readout_score_diagnostics": diagnostics,
        "vlm_reference_results": vlm_comparisons,
        "vlm_comparability": "same fixed 57 IDs; different evidence budgets and supervision",
        "uncertainty": "57 test queries from only two recording days; overlaps invalidate IID confidence claims",
    })
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "summary.json", report)
    # The archive omits per-case correct answers and video/question contents.
    aggregate = {**report, "by_arm": {
        arm: {key: value for key, value in record.items() if key != "cases"}
        for arm, record in report["by_arm"].items()
    }}
    write_json(args.output_dir / "aggregate_metrics.json", aggregate)
    for arm, rows in exports.items():
        (args.output_dir / f"{arm}_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    lines = ["# Idea 01 forecasting MCQ results", "",
             "Primary result: fixed 57-case held-out test. Train/validation are diagnostics.", "",
             "| Arm | Train (111) | Validation (32) | Test (57) |",
             "|---|---:|---:|---:|"]
    for arm, record in report["by_arm"].items():
        cells = [f"{record['by_research_split'][role]['correct']}/{record['by_research_split'][role]['n']} "
                 f"({record['by_research_split'][role]['accuracy_all']:.2%})"
                 for role in ("train", "validation", "test")]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "VLM baselines use 90 frames and no supervised semantic readout training; "
              "the dense prototype uses 3660 frames. This table does not establish a matched-budget winner.", "",
              "| Frozen VLM reference | Same 57-case test |", "|---|---:|"]
    for arm, record in vlm_comparisons.items():
        row = record["test"]
        lines.append(f"| {arm} | {row['correct']}/{row['n']} ({row['accuracy_all']:.2%}) |")
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
