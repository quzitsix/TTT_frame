#!/usr/bin/env python3
"""Write a compact, reproducible report for experiment B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [row for row in rows if row.get("status", "ok") == "ok"]
    scores = [row.get("score", {}) for row in good]
    return {
        "n": len(rows),
        "n_ok": len(good),
        "mean_token_f1": round(mean(float(s.get("token_f1", 0.0)) for s in scores), 4) if scores else 0.0,
        "exact": sum(bool(s.get("exact_normalized")) for s in scores),
        "gold_substring": sum(bool(s.get("gold_substring")) for s in scores),
    }


def home_section(icl_path: Path, spatial_path: Path, full_icl_path: Path | None = None) -> str:
    icl = _read_jsonl(icl_path)
    if not icl:
        raise ValueError(f"empty HomeSentinel ICL file: {icl_path}")
    wanted = {str(row["query_id"]) for row in icl}
    spatial = [row for row in _read_jsonl(spatial_path) if str(row.get("query_id")) in wanted]
    arms = {
        "ICL visual (quick)": icl,
        "Spatial-TTT memory": [row for row in spatial if row.get("arm") == "memory"],
        "Frozen readout": [row for row in spatial if row.get("arm") == "without_memory"],
    }
    if full_icl_path and full_icl_path.is_file():
        arms["ICL visual (full-video)"] = _read_jsonl(full_icl_path)
    lines = [
        "## HomeSentinel quick B", "",
        f"The ICL file contains **{len(icl)}** event/home queries (IDs shared with the Spatial-TTT run: **{len(wanted)}**).",
        "Scores are the existing open-ended lexical diagnostics (token F1, exact normalized match, and gold-substring count). They are not human-validated semantic accuracy.", "",
        "| condition | n | mean token F1 | exact | gold substring |", "|---|---:|---:|---:|---:|",
    ]
    for label, rows in arms.items():
        m = _metric(rows)
        lines.append(f"| {label} | {m['n_ok']}/{m['n']} | {m['mean_token_f1']:.4f} | {m['exact']} | {m['gold_substring']} |")
    lines += [
        "",
        "The quick ICL control samples **2 frames from the first 60 seconds of each observed video**, then caps the history at **32 frames**. The optional full-video ICL row uses the same frame count but samples over each complete observed video. The Spatial-TTT numbers come from the existing `homesentinel_asuka_spatial_60s4f_run6` run, whose full run uses 60-second/4-frame chunks over the 52-video stream. Therefore the quick row is a diagnostic comparison; the full-video row is the closer temporal control.", "",
        "| query | ICL quick F1 | Spatial F1 | frozen F1 | ICL full-video F1 |", "|---|---:|---:|---:|---:|",
    ]
    maps = {label: {str(row["query_id"]): row for row in rows} for label, rows in arms.items()}
    for row in icl:
        qid = str(row["query_id"])
        labels = ("ICL visual (quick)", "Spatial-TTT memory", "Frozen readout", "ICL visual (full-video)")
        vals = [maps.get(label, {}).get(qid, {}).get("score", {}).get("token_f1") for label in labels]
        rendered = ["—" if value is None else f"{float(value):.4f}" for value in vals]
        lines.append(f"| `{qid}` | " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _svc_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| condition | questions | valid/queries | coverage | exact count accuracy | GPA | MoC | UDA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, label in (("icl", "ICL recent window"), ("spatial", "Spatial-TTT full prefix")):
        m = summary.get(condition, {})
        def fmt(value):
            return "—" if value is None else f"{float(value):.4f}"
        lines.append(f"| {label} | {m.get('n_questions', 0)} | {m.get('n_valid', 0)}/{m.get('n_queries', 0)} | {fmt(m.get('coverage'))} | {fmt(m.get('accuracy'))} | {fmt(m.get('gpa'))} | {fmt(m.get('moc'))} | {fmt(m.get('uda'))} |")
    return lines


def svc_section(summary_path: Path, quick_summary_path: Path | None = None) -> str:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}).get("by_condition", {})
    lines = [
        "## SVCBench structured counting", "",
        "This subset contains one generated video (`simulated/0030.mp4`), two questions, and 47 raw ordered query points. The reported smoke keeps every fourth point (13 evaluated points: 9 O2-Unique and 4 E1-Action) to make the two-condition run tractable. At each evaluated point, the runner enforces a strict temporal prefix (`timestamp <= query_time`). ICL receives a bounded recent 8-second/8-frame window; Spatial-TTT ingests all sampled prefix frames incrementally and answers from fast weights.", "",
    ]
    lines.extend(_svc_table(summary))
    if quick_summary_path and quick_summary_path.is_file():
        quick_payload = json.loads(quick_summary_path.read_text(encoding="utf-8"))
        quick = quick_payload.get("summary", {}).get("by_condition", {})
        lines += ["", "The preliminary stride-4/max-side-224 smoke (13 points) was:", ""]
        lines.extend(_svc_table(quick))
    lines += ["", "The metrics are a small-sample trajectory smoke test rather than a claim about the full SVCBench benchmark. Exact count accuracy is the direct pointwise measure; GPA rewards numerical closeness, MoC measures monotonicity of predictions, and UDA measures agreement of change direction."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--homesentinel-icl", type=Path, required=True)
    parser.add_argument("--homesentinel-icl-full", type=Path, default=None)
    parser.add_argument("--homesentinel-spatial", type=Path, required=True)
    parser.add_argument("--svc-summary", type=Path, required=True)
    parser.add_argument("--svc-quick-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = "# Experiment B: long-history and structured counting\n\n"
    document += home_section(args.homesentinel_icl, args.homesentinel_spatial, args.homesentinel_icl_full)
    document += "\n\n" + svc_section(args.svc_summary, args.svc_quick_summary)
    document += r"""

## Reproduction

```bash
# HomeSentinel quick visual ICL (15 event/home queries in the reported run)
conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 \
  --sample-seconds 60 --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10.jsonl --overwrite

# Same 15 queries, sampling over each complete observed video
conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 \
  --sample-seconds 0 --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10_fullvideo.jsonl --overwrite

# Selective SVCBench data (one video, no repository-wide download)
SVC_ROOT=/data/quzitsix/svcbench_subset bash scripts/download_svcbench_subset.sh
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n meowbench python scripts/evaluate_svcbench.py \
  --annotations /data/quzitsix/svcbench_subset/subset/svcbench_subset.json \
  --video-root /data/quzitsix/svcbench_subset \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:0 --dtype bfloat16 --max-side 224 --query-stride 4 \
  --output runs/experiment_b_svcbench_0030_stride4.jsonl
```

The SVCBench annotations and selective-download layout follow the [official SVCBench repository](https://github.com/buaa-colalab/SVCBench) and [dataset card](https://huggingface.co/datasets/buaaplay/SVCBench).
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
