#!/usr/bin/env python3
"""Score completed or partial forecasting baseline runs, separately from inference.

Only this scorer reads the private answer key and audit metadata. Missing,
invalid and failed responses count as incorrect against the entire public
benchmark, never just the completed subset. The exported per-arm predictions
can also be scored by bench_v0/code/evaluate_mcq.py on its required cluster.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttt_frame.forecast_bench import _audit_metadata, _jsonl, load_forecast_cases


SCHEMA = "forecast_mcq_prediction_v1"
DEFAULT_ARMS = (
    "qwen_blind", "qwen_recent_icl", "qwen_full_icl", "blind", "recent_icl",
    "full_icl", "spatial", "spatial_slow_only", "spatial_base",
)
BASELINES = (
    "qwen_blind", "qwen_recent_icl", "qwen_full_icl", "blind", "recent_icl",
    "full_icl", "spatial_slow_only",
)


def parse_choice(raw: Any, options: dict[str, str]) -> str | None:
    """Match the bundled scorer: one letter or normalized exact option text.

    Explanations, embedded letters and ambiguous answers are not extracted.
    Trailing punctuation accepted by the release scorer is preserved here.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().strip("。.)） \n\t")
    if re.fullmatch("[A-Fa-f]", candidate):
        return candidate.upper()
    normalized = re.sub(r"[。.!！\s]+$", "", raw.strip())
    matched = [label for label, text in options.items() if normalized == text]
    return matched[0] if len(matched) == 1 else None


def _unique_by_id(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or sample_id in result:
            raise ValueError(f"invalid or duplicated {name} ID: {sample_id!r}")
        result[sample_id] = row
    return result


def load_references(root: Path) -> tuple[list[str], dict[str, dict]]:
    """Validate IDs, labels and public option text before post-hoc scoring."""
    cases = load_forecast_cases(root)
    metadata = _unique_by_id(_audit_metadata(root, cases), "metadata")
    answers = _unique_by_id(_jsonl(root / "answers.jsonl"), "answer")
    ids = [case.id for case in cases]
    if set(answers) != set(ids):
        raise ValueError("answer IDs do not match public input IDs")
    references = {}
    for case in cases:
        answer, audit = answers[case.id], metadata[case.id]
        options = {option.label: option.text for option in case.options}
        gold = answer.get("correct_answer")
        if gold not in options or gold != audit["output"]["correct_answer"]:
            raise ValueError(f"answer key and metadata disagree: {case.id}")
        for source in (answer, audit["output"]):
            private_options = source.get("options", [])
            if (len(private_options) != 6
                    or {item["label"]: item["text"] for item in private_options} != options):
                raise ValueError(f"private options disagree with public input: {case.id}")
        counts = answer["option_type_counts"]
        mix = (f"{counts.get('plausible_not_observed', 0)} plausible + "
               f"{counts.get('implausible_not_observed', 0)} unlikely")
        references[case.id] = {
            "gold": gold, "options": options,
            "day": str(audit["metadata"]["day"]),
            "participant": audit["metadata"]["participant"],
            "gt_consistency": audit["verification"]["gt_consistency"],
            "distractor_mix": mix,
            "human_verified": bool(audit["verification"].get("human_verified", False)),
        }
    return ids, references


def load_predictions(paths: list[Path], ids: list[str], arms: list[str]) -> tuple[dict, str | None]:
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("expected arms must be nonempty and distinct")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", arm) for arm in arms):
        raise ValueError("arm names must be lowercase identifiers")
    allowed_ids, allowed_arms = set(ids), set(arms)
    predictions: dict[tuple[str, str], dict] = {}
    fingerprints = set()
    for path in paths:
        for row in _jsonl(path):
            if row.get("schema") != SCHEMA:
                raise ValueError(f"unsupported prediction schema in {path}")
            sample_id, arm = row.get("id"), row.get("arm")
            if sample_id not in allowed_ids or arm not in allowed_arms:
                raise ValueError(f"unknown prediction ID or arm: {(sample_id, arm)}")
            key = (sample_id, arm)
            if key in predictions:
                raise ValueError(f"duplicate prediction: {key}")
            fingerprint = row.get("protocol_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("prediction protocol_fingerprint is missing")
            fingerprints.add(fingerprint)
            elapsed = row.get("elapsed_seconds")
            if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or elapsed < 0):
                raise ValueError(f"invalid prediction elapsed_seconds: {key}")
            if row.get("status") not in ("ok", "parse_error", "invalid", "error"):
                raise ValueError(f"unknown prediction status: {key}")
            if "prediction" not in row or not isinstance(row.get("raw_answer"), str):
                raise ValueError(f"missing prediction or raw_answer: {key}")
            mass = row.get("label_probability_mass")
            if mass is not None and (
                isinstance(mass, bool) or not isinstance(mass, (int, float))
                or not math.isfinite(mass) or not 0 <= mass <= 1
            ):
                raise ValueError(f"invalid label_probability_mass: {key}")
            top_token = row.get("unrestricted_top_token")
            if top_token is not None and not isinstance(top_token, str):
                raise ValueError(f"invalid unrestricted_top_token: {key}")
            predictions[key] = row
    if len(fingerprints) > 1:
        raise ValueError("inconsistent protocol fingerprints; do not merge different runs")
    return predictions, next(iter(fingerprints), None)


def _counts(records: list[dict]) -> dict:
    n = len(records)
    correct = sum(row["correct"] for row in records)
    return {
        "n": n, "correct": correct, "accuracy_all": correct / n if n else None,
        "present": sum(not row["missing"] for row in records),
        "valid": sum(row["prediction"] is not None for row in records),
        "missing": sum(row["missing"] for row in records),
        "invalid": sum(row["invalid"] for row in records),
        "errors": sum(row["error"] for row in records),
    }


def _group_scores(records: list[dict], key: str) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for row in records:
        groups[row[key]].append(row)
    return {name: _counts(group) for name, group in sorted(groups.items())}


def _readout_diagnostics(records: list[dict]) -> dict:
    """Report constrained-readout health without changing the selected answer."""
    valid = [row for row in records if row["prediction"] is not None]
    masses = [row["label_probability_mass"] for row in valid
              if row.get("label_probability_mass") is not None]
    tops = [row for row in valid if row.get("unrestricted_top_token") is not None]
    matches = sum(row["unrestricted_top_token"] == row["prediction"] for row in tops)
    return {
        "scope": "successful valid predictions with each optional diagnostic present",
        "label_probability_mass_count": len(masses),
        "label_probability_mass_mean": sum(masses) / len(masses) if masses else None,
        "label_probability_mass_min": min(masses) if masses else None,
        "unrestricted_top_token_count": len(tops),
        "unrestricted_top_token_matches_prediction": matches,
        "unrestricted_top_token_prediction_agreement": matches / len(tops) if tops else None,
        "top_token_match_rule": "exact decoded single-token string equals selected uppercase A-F label",
    }


def load_research_split(path: Path, ids: list[str], references: dict,
                        inputs_sha256: str) -> tuple[dict, dict[str, str]]:
    """Validate a fixed, whole-recording-day research split for subgroup scores."""
    split = json.loads(path.read_text(encoding="utf-8"))
    if (split.get("schema") != "forecast_bench_research_day_split_v1"
            or split.get("official_split") is not False
            or split.get("grouping") != "recording_day_all_participants"):
        raise ValueError("expected unofficial research split grouped by recording day")
    if split.get("inputs_manifest_sha256") != inputs_sha256:
        raise ValueError("research split inputs SHA does not match benchmark")
    if set(split.get("id_to_day", {})) != set(ids):
        raise ValueError("research split id_to_day does not match benchmark IDs")
    memberships, seen_days = {}, set()
    for role in ("train", "validation", "test"):
        selected = split.get("splits", {}).get(role, [])
        days = split.get("days", {}).get(role, [])
        if (not selected or not days or len(set(selected)) != len(selected)
                or len(set(days)) != len(days) or seen_days.intersection(days)):
            raise ValueError("research split has empty, duplicated or overlapping days/IDs")
        if any(type(day) is not int or not 1 <= day <= 7 for day in days):
            raise ValueError("invalid research split recording day")
        seen_days.update(days)
        for sample_id in selected:
            if sample_id not in references or sample_id in memberships:
                raise ValueError("research split has unknown or overlapping IDs")
            day = split["id_to_day"][sample_id]
            if (type(day) is not int or day not in days
                    or str(day) != references[sample_id]["day"]):
                raise ValueError("research split ID/day assignment disagrees with metadata")
            memberships[sample_id] = role
        if role in split.get("counts", {}) and split["counts"][role] != len(selected):
            raise ValueError("research split count disagrees with IDs")
    if set(memberships) != set(ids):
        raise ValueError("research split does not cover all benchmark IDs")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "official_split": False, "grouping": split["grouping"],
        "days": split["days"],
        "counts": {role: len(split["splits"][role]) for role in ("train", "validation", "test")},
        "purpose": "Fixed descriptive subgroups; this scorer does not fit or select models",
    }, memberships


def _paired_scores(pairs: list[tuple[dict, dict]]) -> dict:
    n = len(pairs)
    return {
        "n": n,
        "both_correct": sum(a["correct"] and b["correct"] for a, b in pairs),
        "arm_only_correct": sum(a["correct"] and not b["correct"] for a, b in pairs),
        "baseline_only_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
        "both_incorrect": sum(not a["correct"] and not b["correct"] for a, b in pairs),
        "both_valid": sum(a["prediction"] is not None and b["prediction"] is not None
                          for a, b in pairs),
        "valid_prediction_disagreements": sum(
            a["prediction"] is not None and b["prediction"] is not None
            and a["prediction"] != b["prediction"] for a, b in pairs
        ),
        "accuracy_delta_all": sum(int(a["correct"]) - int(b["correct"])
                                  for a, b in pairs) / n if n else None,
    }


def summarize(root: Path, paths: list[Path], arms: list[str],
              split_manifest: Path | None = None) -> tuple[dict, dict]:
    ids, references = load_references(root)
    predictions, fingerprint = load_predictions(paths, ids, arms)
    inputs_sha256 = hashlib.sha256((root / "inputs_mcq.jsonl").read_bytes()).hexdigest()
    split_info, memberships = None, {}
    if split_manifest is not None:
        split_info, memberships = load_research_split(split_manifest, ids, references, inputs_sha256)
    by_arm, official = {}, {}
    for arm in arms:
        records, exports = [], []
        for sample_id in ids:
            reference = references[sample_id]
            row = predictions.get((sample_id, arm))
            parsed = parse_choice(row["prediction"], reference["options"]) if row else None
            successful = row is not None and row["status"] == "ok"
            choice = parsed if successful else None
            error = row is not None and row["status"] == "error"
            record = {
                "id": sample_id, "prediction": choice,
                "correct_answer": reference["gold"],
                "correct": choice == reference["gold"], "missing": row is None,
                "invalid": row is not None and not error and choice is None,
                "error": error, "status": row["status"] if row else "missing",
                "raw_answer": row["raw_answer"] if row else None,
                "elapsed_seconds": row["elapsed_seconds"] if row else 0,
                "label_probability_mass": row.get("label_probability_mass") if row else None,
                "unrestricted_top_token": row.get("unrestricted_top_token") if row else None,
                **{key: reference[key] for key in (
                    "day", "participant", "gt_consistency", "distractor_mix",
                )},
            }
            if memberships:
                record["research_split"] = memberships[sample_id]
            records.append(record)
            if row is not None:
                exports.append({"id": sample_id, "prediction": choice})
        day_scores = _group_scores(records, "day")
        predicted_counts = Counter(row["prediction"] for row in records
                                   if row["prediction"] is not None)
        gold_counts = Counter(row["correct_answer"] for row in records)
        by_arm[arm] = {
            **_counts(records),
            "complete": all(not row["missing"] for row in records),
            "day_macro_accuracy": sum(group["accuracy_all"] for group in day_scores.values())
                / len(day_scores),
            "elapsed_seconds_sum": sum(row["elapsed_seconds"] for row in records),
            "prediction_counts": dict(sorted(predicted_counts.items())),
            "predicted_label_counts": {label: predicted_counts[label] for label in "ABCDEF"},
            "gold_label_counts": {label: gold_counts[label] for label in "ABCDEF"},
            "by_gold_label": {label: _counts([row for row in records
                                              if row["correct_answer"] == label])
                              for label in "ABCDEF"},
            "readout_diagnostics": _readout_diagnostics(records),
            "by_day": day_scores,
            "by_participant": _group_scores(records, "participant"),
            "by_gt_consistency": _group_scores(records, "gt_consistency"),
            "by_distractor_mix": _group_scores(records, "distractor_mix"),
            "cases": records,
        }
        if memberships:
            by_arm[arm]["by_research_split"] = _group_scores(records, "research_split")
        official[arm] = exports
    paired = {}
    for arm in arms:
        paired[arm] = {}
        for baseline in BASELINES:
            if baseline not in by_arm or baseline == arm:
                continue
            pairs = list(zip(by_arm[arm]["cases"], by_arm[baseline]["cases"]))
            paired[arm][baseline] = _paired_scores(pairs)
            if memberships:
                paired[arm][baseline]["by_research_split"] = {
                    role: _paired_scores([(a, b) for a, b in pairs
                                          if a["research_split"] == role])
                    for role in ("train", "validation", "test")
                }
    report = {
        "schema": "forecast_mcq_summary_v1", "sample_count": len(ids),
        "protocol_fingerprint": fingerprint,
        "complete": all(values["complete"] for values in by_arm.values()),
        "inference_rows": len(predictions), "expected_rows": len(ids) * len(arms),
        "inputs_sha256": inputs_sha256,
        "answers_sha256": hashlib.sha256((root / "answers.jsonl").read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256((root / "metadata.jsonl").read_bytes()).hexdigest(),
        "chance_accuracy": 1 / 6,
        "ground_truth_status": "Automatically screened silver labels; not strict human verification",
        "human_verified_count": sum(row["human_verified"] for row in references.values()),
        "metric_policy": "All public cases in denominator; missing, invalid and errors are incorrect",
        "uncertainty_policy": "Day macro is descriptive; overlapping cases are not IID",
        "score_only_private_inputs": ["answers.jsonl", "metadata.jsonl"],
        "prediction_files": [str(path.resolve()) for path in paths],
        "research_split": split_info,
        "by_arm": by_arm, "paired_vs_baselines": paired,
    }
    return report, official


def markdown_report(report: dict) -> str:
    lines = [
        "# Forecasting MCQ baseline results", "",
        f"Cases per arm: {report['sample_count']}. Complete: {report['complete']}.", "",
        "| Arm | Correct / all | Accuracy | Day macro | Missing | Invalid | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in report["by_arm"].items():
        lines.append(f"| {arm} | {row['correct']}/{row['n']} | {row['accuracy_all']:.2%} | "
                     f"{row['day_macro_accuracy']:.2%} | {row['missing']} | "
                     f"{row['invalid']} | {row['errors']} |")
    if report.get("research_split"):
        lines.extend(["", "## Fixed research split", "",
                      "This is an unofficial day-grouped split, reported separately from the full benchmark.",
                      "", "| Arm | Train | Validation | Test |",
                      "|---|---:|---:|---:|"])
        for arm, row in report["by_arm"].items():
            cells = []
            for role in ("train", "validation", "test"):
                group = row["by_research_split"][role]
                cells.append(f"{group['correct']}/{group['n']} ({group['accuracy_all']:.2%})")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Option-position and readout diagnostics", "",
                  "A-F counts below describe valid predictions; the gold distribution uses all cases.",
                  "", "| Arm | Predicted A/B/C/D/E/F | Gold A/B/C/D/E/F | Label mass mean / min | Unrestricted token agreement |",
                  "|---|---|---|---:|---:|"])
    for arm, row in report["by_arm"].items():
        predicted = "/".join(str(row["predicted_label_counts"][label]) for label in "ABCDEF")
        gold = "/".join(str(row["gold_label_counts"][label]) for label in "ABCDEF")
        diagnostic = row["readout_diagnostics"]
        mass = "n/a"
        if diagnostic["label_probability_mass_mean"] is not None:
            mass = (f"{diagnostic['label_probability_mass_mean']:.4g} / "
                    f"{diagnostic['label_probability_mass_min']:.4g} "
                    f"(n={diagnostic['label_probability_mass_count']})")
        agreement = "n/a"
        if diagnostic["unrestricted_top_token_prediction_agreement"] is not None:
            agreement = (f"{diagnostic['unrestricted_top_token_prediction_agreement']:.2%} "
                         f"(n={diagnostic['unrestricted_top_token_count']})")
        lines.append(f"| {arm} | {predicted} | {gold} | {mass} | {agreement} |")
    lines.extend(["", "| Arm | Gold A accuracy | Gold B | Gold C | Gold D | Gold E | Gold F |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for arm, row in report["by_arm"].items():
        cells = []
        for label in "ABCDEF":
            group = row["by_gold_label"][label]
            cells.append(f"{group['correct']}/{group['n']} "
                         + (f"({group['accuracy_all']:.2%})" if group["n"] else "(n/a)"))
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.extend(["", "Missing, invalid and failed responses count as incorrect.", "",
                  "Labels are automatically screened silver labels. Overlapping histories and "
                  "shared scenes make case-level IID uncertainty inappropriate. These are "
                  "benchmark descriptive scores, not independent-trial significance claims.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--split-manifest", type=Path,
                        help="optional fixed day split for train/validation/test subgroup scores")
    args = parser.parse_args(argv)
    if args.output_dir.resolve().is_relative_to(args.root.resolve()):
        parser.error("output directory must be outside the source benchmark")
    if args.output_dir.exists():
        parser.error("output directory exists; choose a new report directory")
    report, exports = summarize(args.root, args.predictions, args.expected_arms, args.split_manifest)
    args.output_dir.mkdir(parents=True)
    for arm, rows in exports.items():
        (args.output_dir / f"{arm}_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    rendered = markdown_report(report)
    (args.output_dir / "results.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
