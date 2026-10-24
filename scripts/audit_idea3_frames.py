#!/usr/bin/env python3
"""Audit Idea 03 pixels/timestamps against original Qwen full-ICL raw logs.

No answer files or predictions are used. Complete coverage is required unless
--allow-subset explicitly requests a partial smoke audit. Runtime sums are
recorded process time, not elapsed wall time or isolated GPU benchmarking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
ROLES = ("train", "validation", "test")
SEARCH_KEYS = tuple(prefix + radius for prefix in ("search_", "wrong_search_")
                    for radius in ("r01", "r05", "r10"))
HASH_KEYS = ("frame_plan_sha256", "frame_bundle_sha256", "context_frame_bundle_sha256",
             "recent_frame_bundle_sha256")
TIME_KEYS = tuple(f"{role}_{kind}_seconds" for role in ("context", "recent")
                  for kind in ("requested", "actual"))
VISUAL_KEYS = ("visual_key", "text_key", "inverse_text_key", "inverse_text_objective")
PROBE_TIME_KEYS = ("visual_probe_seconds", "reset_probe_seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_bundle(bundle: dict, sid: str, sampling: dict) -> None:
    if (bundle.get("sample_id") != sid or bundle.get("decoded") is not True
            or bundle.get("audio_used") is not False):
        raise ValueError(f"{sid}: frame bundle must contain decoded, silent recipient frames")
    config = bundle["sampling"]
    for key in ("context_frames", "recent_frames", "max_side"):
        if config[key] != sampling[key]:
            raise ValueError(f"{sid}: frame sampling differs from baseline protocol: {key}")
    if any(not digest(bundle.get(key)) for key in HASH_KEYS):
        raise ValueError(f"{sid}: missing or invalid frame hash")
    cutoff = finite(config["cutoff_prefix_seconds"], f"{sid} cutoff", nonnegative=True)
    for role in ("context", "recent"):
        requested, actual = (bundle[f"{role}_{kind}_seconds"] for kind in ("requested", "actual"))
        if len(requested) != sampling[f"{role}_frames"] or len(actual) != len(requested):
            raise ValueError(f"{sid}: invalid {role} timestamp count")
        for values in (requested, actual):
            for value in values:
                finite(value, f"{sid} {role} timestamp", nonnegative=True)
            if any(left > right for left, right in zip(values, values[1:])):
                raise ValueError(f"{sid}: nonmonotonic {role} timestamps")
        if any(observed > target + 1e-9 or target >= cutoff for observed, target in zip(actual, requested)):
            raise ValueError(f"{sid}: noncausal {role} timestamps")


def number_summary(values: list[float]) -> dict:
    return {"n": len(values), "sum": math.fsum(values), "mean": statistics.fmean(values),
            "min": min(values), "max": max(values)}


def runtime_summary(records: list[dict]) -> dict:
    result = {}
    for key in ("elapsed_seconds", "decode_seconds", "extraction_seconds", "cuda_peak_allocated_bytes"):
        result[key] = number_summary([finite(row[key], key, nonnegative=True) for row in records])
    probe_presence = [key in row for row in records for key in PROBE_TIME_KEYS]
    if any(probe_presence):
        if not all(probe_presence):
            raise ValueError("visual/reset probe timings must both be present in all rows or absent in all rows")
        for key in PROBE_TIME_KEYS:
            result[key] = number_summary([finite(row[key], key, nonnegative=True) for row in records])
    result["ingest_seconds"] = number_summary([
        finite(row["write_report"]["stats"]["ingest_seconds"], "ingest_seconds", nonnegative=True)
        for row in records])
    for row in records:
        if set(row["search_seconds"]) != set(SEARCH_KEYS):
            raise ValueError(f"{row['id']}: expected three current and three wrong-memory search timings")
    result["search_seconds"] = {key: number_summary([
        finite(row["search_seconds"][key], key, nonnegative=True) for row in records]) for key in SEARCH_KEYS}
    result["all_search_seconds"] = number_summary([
        math.fsum(row["search_seconds"].values()) for row in records])
    result["current_search_seconds"] = number_summary([
        math.fsum(row["search_seconds"][key] for key in SEARCH_KEYS if key.startswith("search_"))
        for row in records])
    result["wrong_search_seconds"] = number_summary([
        math.fsum(row["search_seconds"][key] for key in SEARCH_KEYS if key.startswith("wrong_"))
        for row in records])
    result["devices"] = sorted({row["device"] for row in records})
    return result


def visual_summary(records: list[dict]) -> dict:
    values = {key: [finite(row["visual_key_diagnostic"][key], key) for row in records]
              for key in VISUAL_KEYS}
    pairs = {"visual_minus_text": ("visual_key", "text_key"),
             "inverse_minus_text": ("inverse_text_key", "text_key"),
             "visual_minus_inverse": ("visual_key", "inverse_text_key")}
    result = {key: number_summary(items) for key, items in values.items()}
    result["gaps"] = {name: number_summary([left - right for left, right in zip(values[a], values[b])])
                      for name, (a, b) in pairs.items()}
    result["visual_gt_text_cases"] = sum(a > b for a, b in zip(values["visual_key"], values["text_key"]))
    return result


def audit(scores: Path, baseline_dir: Path, output: Path, *, allow_subset: bool = False,
          donor_audits: tuple[Path, ...] = ()) -> dict:
    if output.exists():
        raise FileExistsError("audit output must be new")
    baseline_protocol_path = baseline_dir / "protocol.json"
    protocol = json.loads(baseline_protocol_path.read_text(encoding="utf-8"))
    payload, fingerprint = protocol["payload"], protocol["protocol_fingerprint"]
    if canonical_digest(payload) != fingerprint:
        raise ValueError("baseline protocol fingerprint mismatch")
    if output.resolve().is_relative_to(Path(payload["root"]).resolve()):
        raise ValueError("output must be outside source benchmark")
    expected = set(payload["case_ids"])
    if not expected or len(expected) != len(payload["case_ids"]):
        raise ValueError("baseline protocol case IDs must be unique and nonempty")
    files = sorted((baseline_dir / "eval").glob("qwen_shard*.jsonl"))
    if not files:
        raise ValueError("original Qwen raw shards are missing")
    baseline, shared_bundles = {}, {}
    for path in files:
        for row in rows(path):
            sid = row["id"]
            if row.get("protocol_fingerprint") != fingerprint or sid not in expected:
                raise ValueError("mixed baseline protocols or unexpected IDs")
            if not row.get("arm", "").startswith("qwen_"):
                raise ValueError("non-Qwen row in Qwen shard")
            bundle = row["sampling"]
            validate_bundle(bundle, sid, payload["sampling"])
            evidence = {key: bundle[key] for key in (*HASH_KEYS, *TIME_KEYS, "sampling")}
            if sid in shared_bundles and evidence != shared_bundles[sid]:
                raise ValueError(f"{sid}: baseline arms/shards disagree on decoded evidence")
            shared_bundles[sid] = evidence
            if row["arm"] != "qwen_full_icl":
                continue
            if sid in baseline or row.get("status") != "ok":
                raise ValueError("duplicate or failed original full-ICL record")
            baseline[sid] = row
    if set(baseline) != expected:
        raise ValueError("original full-ICL raw logs do not cover baseline protocol IDs")
    records = rows(scores)
    seen, inference_fingerprints, matched = set(), set(), []
    for row in records:
        sid = row["id"]
        if (sid in seen or sid not in expected or row.get("status") != "ok"
                or row.get("answer_keys_used") is not False or row.get("research_split") not in ROLES):
            raise ValueError("duplicate, unexpected, failed, or invalid Idea3 record")
        seen.add(sid)
        if not digest(row.get("protocol_fingerprint")):
            raise ValueError("missing Idea3 inference fingerprint")
        inference_fingerprints.add(row["protocol_fingerprint"])
        bundle, reference = row["sampling"], baseline[sid]["sampling"]
        validate_bundle(bundle, sid, payload["sampling"])
        for key in (*HASH_KEYS, *TIME_KEYS, "sampling"):
            if bundle[key] != reference[key]:
                raise ValueError(f"{sid}: evidence mismatch: {key}")
        matched.append({"id": sid, "research_split": row["research_split"],
                        "frame_bundle_sha256": bundle["frame_bundle_sha256"],
                        "frame_plan_sha256": bundle["frame_plan_sha256"],
                        "context_frames": len(bundle["context_actual_seconds"]),
                        "recent_frames": len(bundle["recent_actual_seconds"]),
                        "last_observed_prefix_seconds": bundle["recent_actual_seconds"][-1],
                        "hashes_and_requested_actual_timestamps_equal": True})
    if not seen or len(inference_fingerprints) != 1:
        raise ValueError("Idea3 scores must be nonempty and share one inference protocol")
    if not allow_subset and seen != expected:
        raise ValueError("Idea3 scores missing baseline cases; use --allow-subset only for a partial smoke audit")
    runtime = {"all": runtime_summary(records)}
    visual = {"all": visual_summary(records)}
    for role in ROLES:
        group = [row for row in records if row["research_split"] == role]
        if group:
            runtime[role], visual[role] = runtime_summary(group), visual_summary(group)
    donor_records = []
    for path in donor_audits:
        values = json.loads(path.read_text(encoding="utf-8"))
        for sid, entry in values.items():
            donor_records.append({"file": str(path.resolve()), "id": sid,
                "decode_seconds": finite(entry["decode_seconds"], "donor decode", nonnegative=True),
                "ingest_seconds": finite(entry["write"]["stats"]["ingest_seconds"],
                                          "donor ingest", nonnegative=True)})
    report = {"schema": "idea03_frame_and_runtime_audit_v1", "audit_passed": True,
        "complete_benchmark_coverage": seen == expected, "allow_subset": allow_subset,
        "expected_cases": len(expected), "matched_cases": len(seen), "missing_ids": sorted(expected - seen),
        "baseline_arm": "qwen_full_icl", "baseline_protocol_fingerprint": fingerprint,
        "inference_protocol_fingerprint": next(iter(inference_fingerprints)),
        "score_file": file_record(scores), "baseline_protocol": file_record(baseline_protocol_path),
        "baseline_raw_shards": [file_record(path) for path in files],
        "audit_source": file_record(Path(__file__)), "answer_files_read": False,
        "evidence_checked": list((*HASH_KEYS, *TIME_KEYS, "sampling")),
        "input_budget": payload["sampling"], "per_case": matched,
        "runtime": runtime, "visual_key_diagnostic": visual,
        "donor_audit_files": [file_record(path) for path in donor_audits], "donor_ingestion": donor_records,
        "runtime_caveats": [
            "Sums are recorded per-case process durations under shared-device concurrency, not wall time or GPU-hours.",
            "Per-case elapsed includes recipient decode, writing, seven text forwards, three current and three wrong radii, reset and visual diagnostics, and overhead.",
            "Seven text forwards comprise one question and six candidates. Current-radius search timings include direct reads and all 21 layers/six candidates.",
            "Donor writes/model loading/startup are outside per-case elapsed. Supplied donor sidecars report decode/ingest only; repeated files/process work is counted separately.",
            "CUDA peak allocated bytes is the process high-water mark at each row, not independently reset per-case total GPU memory.",
            "Visual/reset probe fields, when present, are separately recorded wall-clock intervals; old smoke records omit both fields.",
            "Scalar conversions within probes synchronize relevant GPU work. Timings still include Python overhead and shared-device contention; sums or differences do not establish isolated inference latency or GPU-hours.",
        ],
        "visual_diagnostic_caveat": (
            "Final observed visual K and future-action question Q are compared against the final visual V. "
            "They are not descriptions of the same fact. Positive gaps establish neither a semantic "
            "text-address bottleneck nor retention of older facts, and are not future-action accuracy."),
        "comparison_limit": "Equal decoded pixels/timestamps do not match checkpoint, training, tokenization, temporal packaging, readout or compute."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path,
                        default=REPO / "runs/forecast_bench_v0/mcq_baseline_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true", help="Explicitly label a partial smoke audit")
    parser.add_argument("--donor-audits", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    result = audit(args.scores, args.baseline_dir, args.output, allow_subset=args.allow_subset,
                   donor_audits=tuple(args.donor_audits))
    print(json.dumps({key: result[key] for key in
                      ("audit_passed", "complete_benchmark_coverage", "matched_cases", "expected_cases")}))


if __name__ == "__main__":
    main()
