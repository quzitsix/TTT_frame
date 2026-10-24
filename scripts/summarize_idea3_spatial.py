#!/usr/bin/env python3
"""Create a compact checked Idea 03 archive from frozen predictions and scoring."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.evaluate_idea3_spatial import (
    ROLES, canonical_sha256, read_json, sha256_file, verify_predictions, write_json,
)


def markdown(summary: dict) -> str:
    budget = summary["input_budget"]
    rows = ["# Idea 03: Spatial-TTT address-search forecasting", "",
        "Exploratory comparison on the fixed bench_v0 day split; silver labels, overlapping windows,",
        "and only two test recording days. Prediction files were fixed before test answers were scored.", "",
        "| Method | Validation | Test | Test accuracy | vs full ICL gains / losses |",
        "|---|---:|---:|---:|---:|"]
    for arm, groups in summary["metrics"].items():
        val, test = groups["validation"], groups["test"]
        pair = summary["paired"].get(arm, {}).get("full_icl", {}).get("test")
        comparison = "—" if pair is None else f"{pair['gains']} / {pair['losses']}"
        rows.append(f"| {arm} | {val['correct']}/{val['n']} | {test['correct']}/{test['n']} | "
                    f"{test['accuracy']:.2%} | {comparison} |")
    selection = summary["selection"]
    rows += ["", f"Validation selected direct beta={selection['direct']['beta']}; search "
             f"radius={selection['search']['radius']}, beta={selection['search']['beta']}.", "",
             "`direct` chooses its own beta on validation. `direct_matched` uses the search beta,",
             "so its paired comparison with `search` changes only the read procedure. Reset and",
             "frozen initialization share the same native candidate-likelihood prior; reset association",
             "deltas are exactly zero. Wrong-video controls use the corresponding selected coefficients.", "",
             "`*_raw` are memory-only diagnostics at the radius selected for the mixed search score;",
             "they are not independently tuned competitors. All score normalization is within question",
             "and permutation invariant. Beta=0 permits validation to reject the memory contribution.", "",
             f"Both full ICL and Spatial use {budget['context_frames']} historical plus {budget['recent_frames']} recent",
             f"frames at max side {budget['max_side']}, spanning {budget['context_seconds']}+{budget['recent_seconds']} seconds.",
             "Model checkpoint, readout, and query compute differ.",
             "This does not establish an architecture-level win. The official Spatial checkpoint is",
             "used through this repository's parameter-only path; this is not the complete official",
             "hybrid inference workflow or the proposal's outer-trained semantic classifier.",
             "The native text-V candidate target is an untrained semantic surrogate. A last-observed",
             "visual key versus a future-question query is not a same-fact recall comparison; the visual",
             "probe cannot establish that the desired future information was written but unreadable.", "",
             "Per-case predictions/correctness: `per_case.jsonl` and `per_case.csv`. Raw candidate",
             "regularized association objectives and prior scores: `scores.jsonl`. Pairwise gain/loss IDs and day counts:",
             "`summary.json`. Validation grid and frozen evaluation protocol: `protocol.json`.",
             "Inference model/source identities: `inference_protocol.json`; fixed days: `split_manifest.json`.", "",
             "No test labels were used for query search or configuration selection. The archiver does",
             "not access answer files. Actual ingestion/query runtime and source/model identities,",
             "when emitted by the scorer, remain in `scores.jsonl`; fixed budget is in `protocol.json`.", ""]
    return "\n".join(rows)


def archive(run_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("archive directory must be new")
    protocol, predictions = verify_predictions(run_dir)
    if output.resolve().is_relative_to(Path(protocol["benchmark_root"]).resolve()):
        raise ValueError("archive output must be outside source benchmark")
    report = run_dir / "report"
    manifest = read_json(report / "manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError("incomplete scoring report")
    expected_files = {"summary.json", "per_case.jsonl", "per_case.csv"}
    if set(manifest["files"]) != expected_files:
        raise ValueError("scoring manifest does not cover expected files")
    for name, metadata in manifest["files"].items():
        path = report / name
        if path.stat().st_size != metadata["bytes"] or sha256_file(path) != metadata["sha256"]:
            raise ValueError("scoring report checksum mismatch")
    if read_json(run_dir / "validation_selection.json") != protocol["selection"]:
        raise ValueError("selection sidecar disagrees with frozen evaluation protocol")
    summary = read_json(report / "summary.json")
    if (summary["protocol_fingerprint"] != canonical_sha256(protocol)
            or summary["predictions_sha256"] != sha256_file(run_dir / "predictions.jsonl")):
        raise ValueError("scoring result belongs to different predictions")
    for role in ROLES:
        n = len(protocol["splits"][role])
        if any(groups[role]["n"] != n for groups in summary["metrics"].values()):
            raise ValueError("scoring denominators disagree with split")
    scores = Path(protocol["scores_file"])
    if sha256_file(scores) != protocol["scores_sha256"]:
        raise ValueError("raw scores changed after predictions")
    output.mkdir(parents=True)
    for name in ("protocol.json", "inference_protocol.json", "split_manifest.json", "validation_selection.json",
                 "predictions.jsonl", "completion.json"):
        shutil.copy2(run_dir / name, output / name)
    for name in expected_files:
        shutil.copy2(report / name, output / name)
    shutil.copy2(scores, output / "scores.jsonl")
    (output / "results.md").write_text(markdown(summary), encoding="utf-8")
    installed = {"schema": "idea03_spatial_archive_v1", "source_run": str(run_dir.resolve()),
                 "protocol_fingerprint": canonical_sha256(protocol), "cases": len(predictions),
                 "answer_files_read_by_archiver": False,
                 "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                           for path in sorted(output.iterdir()) if path.is_file()}}
    write_json(output / "manifest.json", installed)
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="evaluator output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = archive(args.run_dir, args.output_dir)
    print(f"Archived {result['cases']} cases and {len(result['files'])} checksummed files to {args.output_dir}")


if __name__ == "__main__":
    main()
