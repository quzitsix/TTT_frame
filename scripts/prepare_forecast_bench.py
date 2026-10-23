#!/usr/bin/env python3
"""Audit forecasting bench_v0 and optionally create a nonofficial day split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt_frame.forecast_bench import audit_forecast_bench, make_research_day_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/forecast_bench_v0"))
    parser.add_argument("--make-research-split", action="store_true")
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(root):
        raise SystemExit("output directory must be outside the source benchmark")
    audit = audit_forecast_bench(root)
    split = make_research_day_split(root) if args.make_research_split else None
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if split is not None:
        (output / "research_day_split.json").write_text(
            json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "samples": audit["sample_count"],
        "human_verified": audit["human_verified"],
        "audit": str(output / "audit.json"),
        "research_split_counts": None if split is None else split["counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
