#!/usr/bin/env python3
"""Audit memory effects using prediction artifacts only, never answer keys.

Checks decoded-frame hashes across both model families, Spatial write exposure
and per-layer updates, and the six A-F logit differences between fast memory
and its slow/base controls. Logit changes and label flips measure influence,
not correctness. The default completeness check expects this benchmark's 200
cases and nine baseline arms. No protocol, model, media or private metadata is
opened by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ARMS = ("qwen_blind", "qwen_recent_icl", "qwen_full_icl", "blind", "recent_icl",
        "full_icl", "spatial", "spatial_slow_only", "spatial_base")
HASH_KEYS = ("frame_plan_sha256", "frame_bundle_sha256", "context_frame_bundle_sha256",
             "recent_frame_bundle_sha256")


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_predictions(paths: list[Path]) -> tuple[dict, str, list[dict]]:
    rows, fingerprints, sources = {}, set(), []
    for path in paths:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"prediction file changed while reading; wait for completion: {path}")
        sources.append({"path": str(path.resolve()), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
        for index, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid prediction JSON at {path}:{index}") from exc
            if (not isinstance(row, dict) or row.get("schema") != "forecast_mcq_prediction_v1"
                    or not isinstance(row.get("id"), str) or row.get("arm") not in ARMS):
                raise ValueError(f"invalid prediction row at {path}:{index}")
            key = (row["id"], row["arm"])
            if key in rows:
                raise ValueError(f"duplicate prediction row: {key}")
            if not _sha256(row.get("protocol_fingerprint")):
                raise ValueError(f"invalid protocol fingerprint: {key}")
            fingerprints.add(row["protocol_fingerprint"])
            rows[key] = row
    if len(fingerprints) != 1:
        raise ValueError("require a nonempty prediction set with one consistent protocol fingerprint")
    return rows, next(iter(fingerprints)), sources


def _logits(row: dict | None) -> list[float] | None:
    if row is None or row.get("status") != "ok":
        return None
    values = row.get("label_logits")
    if (not isinstance(values, list) or len(values) != 6
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in values)):
        return None
    return [float(value) for value in values]


def _frame_check(case_id: str, selected: list[dict]) -> dict:
    reasons = []
    hashes = {key: set() for key in HASH_KEYS}
    family_hashes = {family: {key: set() for key in HASH_KEYS} for family in ("qwen", "spatial")}
    present_arms = set()
    for row in selected:
        present_arms.add(row["arm"])
        family = "qwen" if row["arm"].startswith("qwen_") else "spatial"
        sampling = row.get("sampling")
        if (not isinstance(sampling, dict) or sampling.get("decoded") is not True
                or sampling.get("sample_id") != case_id or sampling.get("audio_used") is not False):
            reasons.append(f"{row['arm']}: missing decoded past-only frame audit")
            continue
        for key in HASH_KEYS:
            value = sampling.get(key)
            if not _sha256(value):
                reasons.append(f"{row['arm']}: missing/invalid {key}")
                continue
            hashes[key].add(value)
            family_hashes[family][key].add(value)
    missing = sorted(set(ARMS) - present_arms)
    if missing:
        reasons.append("missing arms: " + ", ".join(missing))
    for key, values in hashes.items():
        if len(values) > 1:
            reasons.append(f"frame hash mismatch: {key}")
    both = all(family_hashes[family]["frame_bundle_sha256"] for family in ("qwen", "spatial"))
    if not both:
        reasons.append("both model-family decoded bundles are required")
    return {"id": case_id, "passed": not reasons, "reasons": reasons,
            "frame_hashes": {key: sorted(values) for key, values in hashes.items()},
            "both_families_present": bool(both)}


def _write_check(case_id: str, rows: dict, expected_frames: int, expected_tokens: int) -> dict:
    reasons, reports = [], []
    for arm in ("spatial", "spatial_slow_only", "spatial_base"):
        row = rows.get((case_id, arm))
        report = row.get("write_report") if row else None
        if not isinstance(report, dict):
            reasons.append(f"{arm}: missing write_report")
        else:
            reports.append(report)
    if not reports:
        return {"id": case_id, "passed": False, "reasons": reasons}
    if any(report != reports[0] for report in reports[1:]):
        reasons.append("spatial controls do not share the same completed write report")
    report = reports[0]
    stats = report.get("stats", {})
    if stats.get("frames") != expected_frames:
        reasons.append(f"expected {expected_frames} real frames, got {stats.get('frames')}")
    if stats.get("visual_tokens") != expected_tokens:
        reasons.append(f"expected {expected_tokens} visual tokens, got {stats.get('visual_tokens')}")
    layers, updates = report.get("ttt_layers", []), report.get("updates_per_layer", {})
    if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
            or not isinstance(updates, dict) or set(updates) != {str(layer) for layer in layers}
            or any(type(value) is not int or value <= 0 for value in updates.values())):
        reasons.append("expected strictly positive updates for every declared TTT layer")
    if report.get("trained_checkpoint_loaded") is not True:
        reasons.append("official trained checkpoint is not confirmed loaded")
    if (report.get("parameter_only") is not True or report.get("retained_video_frames") != 0
            or report.get("retained_text_records") != 0):
        reasons.append("write report does not confirm parameter-only memory")
    return {"id": case_id, "passed": not reasons, "reasons": reasons,
            "real_frames": stats.get("frames"), "visual_tokens": stats.get("visual_tokens"),
            "ttt_layers": layers, "updates_per_layer": updates}


def _compare(case_ids: list[str], rows: dict, control: str) -> dict:
    cases, missing = [], []
    for case_id in case_ids:
        fast_row, control_row = rows.get((case_id, "spatial")), rows.get((case_id, control))
        fast, slow = _logits(fast_row), _logits(control_row)
        if fast is None or slow is None:
            missing.append(case_id)
            continue
        difference = [a - b for a, b in zip(fast, slow)]
        max_abs = max(abs(value) for value in difference)
        fast_label = "ABCDEF"[max(range(6), key=lambda i: fast[i])]
        control_label = "ABCDEF"[max(range(6), key=lambda i: slow[i])]
        cases.append({
            "id": case_id, "max_abs_logit_difference": max_abs,
            "l2_logit_difference": math.sqrt(sum(value * value for value in difference)),
            "label_flipped": fast_label != control_label,
            "spatial_prediction": fast_row.get("prediction"),
            "control_prediction": control_row.get("prediction"),
            "logit_argmax_matches_recorded_predictions": fast_row.get("prediction") == fast_label
                and control_row.get("prediction") == control_label,
            "spatial_minus_control_A_to_F": difference,
        })
    n = len(cases)
    return {
        "control": control, "compared_cases": n, "missing_or_invalid_logit_ids": missing,
        "label_flips": sum(case["label_flipped"] for case in cases),
        "label_flip_rate": sum(case["label_flipped"] for case in cases) / n if n else None,
        "nonzero_logit_difference_cases": sum(case["max_abs_logit_difference"] > 0 for case in cases),
        "logit_difference_above_1e_minus_6_cases": sum(
            case["max_abs_logit_difference"] > 1e-6 for case in cases),
        "max_abs_logit_difference_mean": sum(case["max_abs_logit_difference"] for case in cases)
            / n if n else None,
        "max_abs_logit_difference_max": max((case["max_abs_logit_difference"] for case in cases),
                                            default=None),
        "l2_logit_difference_mean": sum(case["l2_logit_difference"] for case in cases) / n if n else None,
        "l2_logit_difference_max": max((case["l2_logit_difference"] for case in cases), default=None),
        "argmax_prediction_mismatch_ids": [case["id"] for case in cases
                                           if not case["logit_argmax_matches_recorded_predictions"]],
        "cases": cases,
    }


def analyze(paths: list[Path], *, expected_cases: int = 200,
            expected_frames: int = 90, expected_visual_tokens: int = 2205) -> dict:
    if min(expected_cases, expected_frames, expected_visual_tokens) < 1:
        raise ValueError("expected case, frame and token counts must be positive")
    rows, fingerprint, sources = read_predictions(paths)
    case_ids = sorted({sample_id for sample_id, _ in rows})
    missing = {case_id: [arm for arm in ARMS if (case_id, arm) not in rows] for case_id in case_ids}
    missing = {case_id: arms for case_id, arms in missing.items() if arms}
    failed = [{"id": row["id"], "arm": row["arm"], "status": row.get("status")}
              for row in rows.values() if row.get("status") != "ok"]
    frames = [_frame_check(case_id, [row for (sample_id, _), row in rows.items()
                                   if sample_id == case_id]) for case_id in case_ids]
    writes = [_write_check(case_id, rows, expected_frames, expected_visual_tokens) for case_id in case_ids]
    comparisons = {control: _compare(case_ids, rows, control)
                   for control in ("spatial_slow_only", "spatial_base")}
    complete = len(case_ids) == expected_cases and not missing and not failed
    passed = (complete and all(row["passed"] for row in frames + writes)
              and all(not row["missing_or_invalid_logit_ids"]
                      and not row["argmax_prediction_mismatch_ids"] for row in comparisons.values()))
    return {
        "schema": "forecast_mcq_prediction_diagnostics_v1", "answers_read": False,
        "interpretation": "Logit differences and flips establish influence, not accuracy or useful memory",
        "protocol_fingerprint": fingerprint, "prediction_sources": sources,
        "expected_case_count": expected_cases, "observed_case_count": len(case_ids),
        "expected_rows": expected_cases * len(ARMS), "observed_rows": len(rows),
        "observed_rows_by_arm": dict(sorted(Counter(arm for _, arm in rows).items())),
        "missing_arms_by_observed_id": missing, "failed_prediction_rows": failed,
        "coverage_note": "Unseen case IDs cannot be identified from prediction artifacts alone",
        "complete": complete, "all_checks_passed": passed,
        "frame_bundle_checks": {"passed_cases": sum(row["passed"] for row in frames), "cases": frames},
        "spatial_write_checks": {"expected_real_frames": expected_frames,
                                 "expected_visual_tokens": expected_visual_tokens,
                                 "passed_cases": sum(row["passed"] for row in writes), "cases": writes},
        "spatial_logit_comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--expected-frames", type=int, default=90)
    parser.add_argument("--expected-visual-tokens", type=int, default=2205)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new diagnostics file")
    if any(args.output.resolve() == path.resolve() for path in args.predictions):
        parser.error("output cannot replace a prediction file")
    report = analyze(args.predictions, expected_cases=args.expected_cases,
                     expected_frames=args.expected_frames, expected_visual_tokens=args.expected_visual_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps({key: report[key] for key in (
        "observed_case_count", "observed_rows", "complete", "all_checks_passed",
    )}))
    return 0 if report["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
