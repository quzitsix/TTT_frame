#!/usr/bin/env python3
"""Cache causal still-image Qwen visual features from public bench_v0 past inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import torch

from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_features import FeatureConfig, QwenImageFeatures, cache_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/share/raw/bench_v0"))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/data/quzitsix/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--context-tail-seconds", type=float)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.output_dir.resolve().is_relative_to(args.root.resolve()):
        parser.error("output directory must be outside the source benchmark")
    torch.set_num_threads(args.threads)
    config = FeatureConfig(
        sample_fps=args.sample_fps, max_side=args.max_side, dtype=args.dtype,
        projection_seed=args.projection_seed, context_tail_seconds=args.context_tail_seconds,
        limit_frames=args.limit_frames,
    )
    cases = load_forecast_cases(args.root, ids=args.ids)
    for case in cases:
        if (args.output_dir / f"{case.id}.pt").exists():
            parser.error(f"cache already exists for {case.id}; choose a new output directory")
    manifest_sha = hashlib.sha256((args.root / "inputs_mcq.jsonl").read_bytes()).hexdigest()
    print(f"Loading visual tower for {len(cases)} case(s)", flush=True)
    encoder = QwenImageFeatures(args.model_path, config, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        start = time.perf_counter()
        print(f"Encoding {case.id}", flush=True)
        last_progress = 0

        def report_progress(done: int, total: int) -> None:
            nonlocal last_progress
            if done - last_progress >= 256 or done == total:
                print(f"  {case.id}: {done}/{total} frames", flush=True)
                last_progress = done

        cache = cache_case(case, encoder, input_manifest_sha256=manifest_sha,
                           batch_size=args.batch_size, progress=report_progress)
        cache["extraction_seconds"] = time.perf_counter() - start
        destination = args.output_dir / f"{case.id}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(cache, temporary)
        temporary.replace(destination)
        print(f"Saved {destination}: {len(cache['features'])} frames, "
              f"complete_prefix={cache['complete_prefix']}, "
              f"seconds={time.perf_counter() - start:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
