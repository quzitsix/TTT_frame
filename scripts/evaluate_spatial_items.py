#!/usr/bin/env python3
"""Batch-evaluate saved Spatial-TTT memories on a JSONL MCQ set.

Each worker process loads one model and one saved episode memory once, then
answers every item.  This is substantially faster and more reproducible than
starting a new Qwen process for every question.  The item gold/evidence fields
are used only by the evaluator; the model receives the rendered question,
options, and the standard single-letter instruction.

Example::

    python scripts/evaluate_spatial_items.py \
      --items-path data/q9_codex_mcq.jsonl \
      --parallel --devices cuda:3,cuda:4,cuda:5 \
      --max-new-tokens 16 \
      --output runs/q9_codex_mcq_eval.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ROOT / "data" / "q9_codex_mcq.jsonl"
DEFAULT_MODEL = "/data/quzitsix/models/Qwen3-VL-2B-Instruct"
DEFAULT_DENSE = ROOT / "runs" / "q9_full_history_spatial_official_4s8f"
DEFAULT_SPARSE = ROOT / "runs" / "q9_full_history_spatial_official_16f"
SINGLE_LETTER_INSTRUCTION = (
    "Answer with the single letter of the best option and nothing else."
)


@dataclass(frozen=True)
class Condition:
    name: str
    memory: Path
    use_memory: bool


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate the evaluator-facing portion of a JSONL question set."""

    path = Path(path).expanduser()
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read items JSONL {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            item_id = item.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"missing item_id at {path}:{line_number}")
            if item_id in seen:
                raise ValueError(f"duplicate item_id {item_id!r} at {path}:{line_number}")
            question = item.get("question")
            options = item.get("options")
            answer = item.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"empty question for {item_id}")
            if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
                raise ValueError(f"{item_id} must have exactly A/B/C/D options")
            if answer not in options:
                raise ValueError(f"invalid answer {answer!r} for {item_id}")
            seen.add(item_id)
            items.append(item)
    if not items:
        raise ValueError(f"no items found in {path}")
    return items


def render_prompt(item: dict[str, Any]) -> str:
    """Render the same MCQ prompt as MEOWBench's native adapter."""

    options = item["options"]
    lines = "\n".join(f"{letter}. {options[letter]}" for letter in sorted(options))
    return f"{item['question'].strip()}\n\n{lines}\n\n{SINGLE_LETTER_INSTRUCTION}"


def score_reply(raw: str, item: dict[str, Any]) -> dict[str, Any]:
    """Parse and score a completion with MEOWBench's deterministic scorer."""

    try:
        from meowbench.scoring.deterministic import extract_mcq_letter, score_mcq

        parsed = extract_mcq_letter(raw, options=item["options"])
        score = float(score_mcq(raw, item["answer"], options=item["options"]))
    except ImportError:
        # Keep model-free local tests usable outside the meowbench environment.
        # Production runs use the scorer above.
        text = str(raw or "").strip()
        match = re.match(r"^\s*\**\s*\(?([ABCD])\)?(?:[.):,]|\s|$)", text, re.I)
        parsed = match.group(1).upper() if match else None
        score = float(parsed == item["answer"])
    return {"parsed": parsed, "correct": bool(score == 1.0), "score": score}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return overall and per-axis exact-MCQ accuracy for one condition."""

    by_axis: dict[str, dict[str, Any]] = {}
    for row in rows:
        axis = str(row.get("axis") or "unspecified")
        bucket = by_axis.setdefault(axis, {"correct": 0, "n": 0})
        bucket["n"] += 1
        bucket["correct"] += int(row["correct"])
    for bucket in by_axis.values():
        bucket["accuracy"] = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
    correct = sum(int(row["correct"]) for row in rows)
    n = len(rows)
    return {"correct": correct, "n": n, "accuracy": correct / n if n else 0.0,
            "by_axis": by_axis}


def _worker(args: argparse.Namespace) -> int:
    """Load one model/memory and answer all items; called in a child process."""

    # Imports are intentionally delayed: the parent can validate/format data
    # without importing transformers or allocating a CUDA model.
    from ttt_frame.spatial_videoqa import SpatialVideoMemory, _config_from_saved

    items = load_items(args.items_path)
    config_args = argparse.Namespace(
        memory=args.memory,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        spatial_checkpoint=None,
    )
    memory = SpatialVideoMemory(_config_from_saved(config_args))
    memory.load_memory(args.memory)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for item in items:
        started = time.perf_counter()
        raw = memory.answer(
            render_prompt(item),
            use_memory=args.use_memory,
            max_new_tokens=args.max_new_tokens,
        )
        parsed = score_reply(raw, item)
        rows.append({
            "item_id": item["item_id"],
            "axis": item.get("axis"),
            "raw_answer": raw,
            **parsed,
            "gold": item["answer"],
            "elapsed_sec": time.perf_counter() - started,
        })
    payload = {
        "condition": args.condition,
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


def _run_worker(condition: Condition, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--condition",
        condition.name,
        "--memory",
        str(condition.memory),
        "--items-path",
        str(args.items_path),
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--worker-output",
        str(output),
    ]
    if condition.use_memory:
        command.append("--use-memory")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{condition.name} on {args.device} failed with {completed.returncode}: {detail[-4000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _build_conditions(args: argparse.Namespace) -> list[Condition]:
    return [
        Condition("spatial_dense_4s8f", Path(args.dense_memory), True),
        Condition("spatial_sparse_60s16f", Path(args.sparse_memory), True),
        Condition("spatial_official_without_memory", Path(args.control_memory), False),
    ]


def _print_report(payload: dict[str, Any]) -> None:
    print(f"Items: {payload['n_items']}")
    print(f"Gold distribution: {payload['gold_counts']}")
    baseline = payload["always_letter_baseline"]
    print(f"Always-letter baseline: {baseline['letter']} = {baseline['accuracy']:.3f}")
    print("\nCondition                              Correct   Accuracy")
    print("--------------------------------------------------------")
    for name, result in payload["conditions"].items():
        summary = result["summary"]
        print(f"{name:38s} {summary['correct']:>3}/{summary['n']:<3}     {summary['accuracy']:.3f}")
        axes = ", ".join(
            f"{axis}={bucket['correct']}/{bucket['n']}"
            for axis, bucket in sorted(summary["by_axis"].items())
        )
        print(f"  by axis: {axes}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-path", default=str(DEFAULT_ITEMS))
    parser.add_argument("--dense-memory", default=str(DEFAULT_DENSE))
    parser.add_argument("--sparse-memory", default=str(DEFAULT_SPARSE))
    parser.add_argument("--control-memory", default=str(DEFAULT_DENSE),
                        help="memory directory used to establish the no-read control")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--devices", default="", help="comma-separated devices for --parallel")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--output", default="runs/q9_codex_mcq_eval.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--condition", default="", help=argparse.SUPPRESS)
    parser.add_argument("--memory", default="", help=argparse.SUPPRESS)
    parser.add_argument("--use-memory", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker:
        if not args.memory or not args.worker_output or not args.condition:
            raise SystemExit("worker requires --memory, --condition and --worker-output")
        return _worker(args)

    items = load_items(args.items_path)
    conditions = _build_conditions(args)
    for condition in conditions:
        if not condition.memory.is_dir():
            raise SystemExit(f"memory directory does not exist: {condition.memory}")
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if args.parallel:
        if len(devices) < len(conditions):
            raise SystemExit(f"--parallel needs {len(conditions)} devices; got {len(devices)}")
    else:
        devices = [args.device] * len(conditions)

    gold_counts = Counter(item["answer"] for item in items)
    baseline_letter, baseline_count = max(sorted(gold_counts.items()), key=lambda pair: pair[1])
    payload: dict[str, Any] = {
        "schema": "ttt_frame.spatial_mcq_eval/1",
        "items_path": str(Path(args.items_path).resolve()),
        "n_items": len(items),
        "gold_counts": dict(sorted(gold_counts.items())),
        "always_letter_baseline": {
            "letter": baseline_letter,
            "correct": baseline_count,
            "n": len(items),
            "accuracy": baseline_count / len(items),
        },
        "conditions": {},
    }
    with tempfile.TemporaryDirectory(prefix="spatial-mcq-") as temp_dir:
        temp = Path(temp_dir)
        jobs = [
            (condition, devices[index], temp / f"{index}.json")
            for index, condition in enumerate(conditions)
        ]

        def run(job: tuple[Condition, str, Path]) -> tuple[str, dict[str, Any]]:
            condition, device, output = job
            worker_args = argparse.Namespace(**vars(args))
            worker_args.device = device
            return condition.name, _run_worker(condition, worker_args, output)

        if args.parallel:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                results = list(pool.map(run, jobs))
        else:
            results = [run(job) for job in jobs]
    for name, result in results:
        payload["conditions"][name] = result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_report(payload)
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
