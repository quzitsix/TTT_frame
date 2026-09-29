#!/usr/bin/env python3
"""Compare dense/sparse Spatial-TTT memories on one user question.

The script deliberately runs each condition in a subprocess, so one Qwen model
is released before the next one is loaded.  The no-memory result is the same
official Spatial-TTT checkpoint with only the episode fast-weight read bypassed;
it is not a raw, untrained Qwen baseline.

Example::

    python scripts/test_spatial_memory.py \
      --device cuda:2 \
      --question 'Where did I put the empty red mesh bag?'
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


DEFAULT_MODEL = "/data/quzitsix/models/Qwen3-VL-2B-Instruct"
DEFAULT_DENSE = "runs/q9_full_history_spatial_official_4s8f"
DEFAULT_SPARSE = "runs/q9_full_history_spatial_official_16f"


@dataclass(frozen=True)
class Condition:
    label: str
    memory: Path
    use_memory: bool = True


def run_condition(condition: Condition, args: argparse.Namespace, question: str) -> int:
    print(f"\n===== {condition.label} =====", flush=True)
    if not condition.memory.is_dir():
        print(f"UNAVAILABLE: {condition.memory}")
        return 0
    command = [
        sys.executable,
        "-m",
        "ttt_frame.spatial_videoqa",
        "ask",
        "--memory",
        str(condition.memory),
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--question",
        question,
    ]
    if not condition.use_memory:
        command.append("--without-memory")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    answer = completed.stdout.strip()
    if answer:
        print(answer)
    if completed.returncode:
        print(f"ERROR: exit status {completed.returncode}")
        if completed.stderr.strip():
            print(completed.stderr.strip()[-4000:])
        return completed.returncode
    notes = [line for line in completed.stderr.splitlines()
             if "truncated" in line.lower() or "max_new_tokens" in line.lower()]
    for note in notes[-3:]:
        print(f"NOTE: {note}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="question (or use --question)")
    parser.add_argument("--question", dest="question_option")
    parser.add_argument("--dense-memory", default=DEFAULT_DENSE)
    parser.add_argument("--sparse-memory", default=DEFAULT_SPARSE)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--concise", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    question = args.question_option or args.question
    if question is None:
        question = input("Question: ").strip()
    else:
        question = question.strip()
    if not question:
        raise SystemExit("question must not be empty")
    if args.concise:
        question = "Answer in one concise sentence. Do not repeat items.\n" + question
    print(f"Question: {question}")
    print(f"Generation budget: {args.max_new_tokens} new tokens")
    conditions = [Condition("Spatial-TTT + 4s/8f memory", Path(args.dense_memory))]
    if not args.skip_sparse:
        conditions.append(Condition("Spatial-TTT + 60s/16f memory", Path(args.sparse_memory)))
    if not args.skip_control:
        conditions.append(Condition(
            "Spatial-TTT official checkpoint (--without-memory)",
            Path(args.dense_memory), use_memory=False,
        ))
    return max(run_condition(condition, args, question) for condition in conditions)


if __name__ == "__main__":
    raise SystemExit(main())
