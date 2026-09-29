#!/usr/bin/env python3
"""Evaluate Codex-derived MCQs with the three project memory methods.

The default methods are the two LoRA memories trained from the local and
Codex teachers, plus the dense 4s/8f Spatial-TTT memory.  A worker loads one
model once and answers the whole JSONL set, so the three methods can run in
parallel without reloading a model for every question.

Example::

    python scripts/evaluate_codex_mcq.py \
      --items-path data/q9_codex_mcq.jsonl \
      --parallel --devices cuda:3,cuda:4,cuda:5 \
      --max-new-tokens 16 \
      --output runs/q9_codex_mcq_models.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_spatial_items import (  # noqa: E402
    DEFAULT_ITEMS,
    DEFAULT_MODEL,
    render_prompt,
    load_items,
    score_reply,
    summarize,
)


DEFAULT_LOCAL = ROOT / "runs" / "q9_full_history_lora_local_60s"
DEFAULT_CODEX = ROOT / "runs" / "q9_full_history_lora_codex_sol"
DEFAULT_SPATIAL = ROOT / "runs" / "q9_full_history_spatial_official_4s8f"
DEFAULT_SPARSE = ROOT / "runs" / "q9_full_history_spatial_official_16f"


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    memory: Path
    use_memory: bool = True


def _load_memory(args: argparse.Namespace):
    metadata = json.loads((Path(args.memory) / "memory.json").read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    config.update(model_path=args.model_path, device=args.device, dtype=args.dtype)
    if args.kind == "lora":
        from ttt_frame.videoqa import VideoTTTConfig, VideoTTTMemory

        memory = VideoTTTMemory(VideoTTTConfig(**config))
    elif args.kind == "spatial":
        from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory

        memory = SpatialVideoMemory(SpatialVideoConfig(**config))
    else:
        raise ValueError(f"unknown method kind: {args.kind}")
    memory.load_memory(args.memory)
    return memory


def _worker(args: argparse.Namespace) -> int:
    items = load_items(args.items_path)
    memory = _load_memory(args)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for item in items:
        started = time.perf_counter()
        raw = memory.answer(
            render_prompt(item),
            use_memory=args.use_memory,
            max_new_tokens=args.max_new_tokens,
        )
        rows.append({
            "item_id": item["item_id"],
            "axis": item.get("axis"),
            "raw_answer": raw,
            **score_reply(raw, item),
            "gold": item["answer"],
            "elapsed_sec": time.perf_counter() - started,
        })
    payload = {
        "method": args.method,
        "kind": args.kind,
        "memory": args.memory,
        "device": args.device,
        "use_memory": args.use_memory,
        "rows": rows,
        "summary": summarize(rows),
        "elapsed_sec": time.perf_counter() - started_all,
    }
    Path(args.worker_output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _run_worker(method: Method, args: argparse.Namespace, device: str, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        method.name,
        "--kind",
        method.kind,
        "--memory",
        str(method.memory),
        "--items-path",
        str(args.items_path),
        "--model-path",
        args.model_path,
        "--device",
        device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--worker-output",
        str(output),
    ]
    if method.use_memory:
        command.append("--use-memory")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{method.name} on {device} failed: {detail[-4000:]}")
    return json.loads(output.read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-path", default=str(DEFAULT_ITEMS))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--local-memory", default=str(DEFAULT_LOCAL))
    parser.add_argument("--codex-memory", default=str(DEFAULT_CODEX))
    parser.add_argument("--spatial-memory", default=str(DEFAULT_SPATIAL))
    parser.add_argument("--sparse-spatial-memory", default=str(DEFAULT_SPARSE))
    parser.add_argument("--include-sparse-spatial", action="store_true",
                        help="add the older 60s/16f Spatial memory as a fourth method")
    parser.add_argument("--include-spatial-control", action="store_true",
                        help="add a no-fast-memory Spatial control as a fifth method")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--devices", default="", help="comma-separated devices for --parallel")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--output", default="runs/q9_codex_mcq_models.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", default="", help=argparse.SUPPRESS)
    parser.add_argument("--kind", default="", help=argparse.SUPPRESS)
    parser.add_argument("--memory", default="", help=argparse.SUPPRESS)
    parser.add_argument("--use-memory", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _methods(args: argparse.Namespace) -> list[Method]:
    methods = [
        Method("lora_local_teacher", "lora", Path(args.local_memory)),
        Method("lora_codex_teacher", "lora", Path(args.codex_memory)),
        Method("spatial_dense_4s8f", "spatial", Path(args.spatial_memory)),
    ]
    if args.include_sparse_spatial:
        methods.append(Method("spatial_sparse_60s16f", "spatial", Path(args.sparse_spatial_memory)))
    if args.include_spatial_control:
        methods.append(Method("spatial_official_without_memory", "spatial",
                              Path(args.spatial_memory), use_memory=False))
    return methods


def _print_report(payload: dict[str, Any]) -> None:
    print(f"Items: {payload['n_items']}")
    print(f"Gold distribution: {payload['gold_counts']}")
    base = payload["always_letter_baseline"]
    print(f"Always-letter baseline: {base['letter']} = {base['accuracy']:.3f}")
    print("\nMethod                                 Correct   Accuracy")
    print("--------------------------------------------------------")
    for name, result in payload["methods"].items():
        summary = result["summary"]
        print(f"{name:38s} {summary['correct']:>3}/{summary['n']:<3}     {summary['accuracy']:.3f}")
        axes = ", ".join(
            f"{axis}={bucket['correct']}/{bucket['n']}"
            for axis, bucket in sorted(summary["by_axis"].items())
        )
        print(f"  by axis: {axes}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        if not all((args.method, args.kind, args.memory, args.worker_output)):
            raise SystemExit("worker requires method, kind, memory and worker-output")
        return _worker(args)
    items = load_items(args.items_path)
    methods = _methods(args)
    for method in methods:
        if not method.memory.is_dir():
            raise SystemExit(f"memory directory does not exist: {method.memory}")
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if args.parallel:
        if len(devices) < len(methods):
            raise SystemExit(f"--parallel needs {len(methods)} devices; got {len(devices)}")
    else:
        devices = [args.device] * len(methods)
    gold_counts = Counter(item["answer"] for item in items)
    baseline_letter, baseline_count = max(sorted(gold_counts.items()), key=lambda pair: pair[1])
    payload: dict[str, Any] = {
        "schema": "ttt_frame.codex_mcq_eval/1",
        "items_path": str(Path(args.items_path).resolve()),
        "n_items": len(items),
        "gold_counts": dict(sorted(gold_counts.items())),
        "always_letter_baseline": {
            "letter": baseline_letter, "correct": baseline_count, "n": len(items),
            "accuracy": baseline_count / len(items),
        },
        "methods": {},
    }
    with tempfile.TemporaryDirectory(prefix="codex-mcq-") as temp_dir:
        temp = Path(temp_dir)
        jobs = [(method, devices[i], temp / f"{i}.json") for i, method in enumerate(methods)]

        def run(job: tuple[Method, str, Path]) -> tuple[str, dict[str, Any]]:
            method, device, output = job
            return method.name, _run_worker(method, args, device, output)

        if args.parallel:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                results = list(pool.map(run, jobs))
        else:
            results = [run(job) for job in jobs]
    for name, result in results:
        payload["methods"][name] = result
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_report(payload)
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
