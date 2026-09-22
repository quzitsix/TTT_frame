#!/usr/bin/env python3
"""Run the same question through the saved video-memory methods.

Each method is executed in a separate subprocess so that one Qwen model is
released before the next one is loaded.  This keeps the comparison usable on a
single GPU and makes the command line configuration identical across methods.

When Spatial-TTT is enabled, the script also runs the same official checkpoint
with ``--without-memory``.  That paired control bypasses only the episode
fast-weight read branch; it still contains the official tuned language model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "/data/quzitsix/models/Qwen3-VL-2B-Instruct"
DEFAULT_LOCAL = "runs/teacher_pilot/lora_local_v2"
DEFAULT_CODEX = "runs/teacher_pilot/lora_codex_v2"
DEFAULT_SPATIAL = "runs/spatial_smoke_cuda2"


@dataclass(frozen=True)
class Method:
    label: str
    module: str
    memory: Path
    use_memory: bool = True


def build_command(method: Method, args: argparse.Namespace, question: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        method.module,
        "ask",
        "--memory",
        str(method.memory),
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--question",
        question,
    ]
    if method.module == "ttt_frame.videoqa":
        command.append("--local-files-only")
    command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if not method.use_memory:
        command.append("--without-memory")
    return command


def run_method(method: Method, args: argparse.Namespace, question: str) -> int:
    print(f"\n===== {method.label} =====", flush=True)
    if not method.memory.is_dir():
        print(f"UNAVAILABLE: memory directory does not exist: {method.memory}")
        return 0

    completed = subprocess.run(
        build_command(method, args, question),
        text=True,
        capture_output=True,
        check=False,
    )
    answer = completed.stdout.strip()
    if completed.returncode == 0:
        print(answer or "(empty answer)")
        diagnostics = completed.stderr.strip()
        if diagnostics:
            # Keep normal Transformers warnings out of the comparison table,
            # but surface the actionable generation-budget warning.
            notes = [line for line in diagnostics.splitlines()
                     if "truncated" in line.lower() or "max_new_tokens" in line.lower()]
            if notes:
                print("NOTE: " + " | ".join(notes[-3:]))
        return 0

    print(f"ERROR: command exited with status {completed.returncode}")
    diagnostics = completed.stderr.strip()
    if diagnostics:
        print(diagnostics[-4000:])
    if answer:
        print("stdout:")
        print(answer[-4000:])
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base, LoRA-teacher, and Spatial-TTT answers to one question."
    )
    parser.add_argument("question", nargs="?", help="question; omit it to be prompted")
    parser.add_argument("--question", dest="question_option", help="question (alternative to positional input)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--concise",
        action="store_true",
        help="prepend an instruction for one short, non-repeating answer",
    )
    parser.add_argument("--local-memory", default=DEFAULT_LOCAL)
    parser.add_argument("--codex-memory", default=DEFAULT_CODEX)
    parser.add_argument("--spatial-memory", default=DEFAULT_SPATIAL)
    parser.add_argument("--skip-base", action="store_true", help="skip the frozen-base control")
    parser.add_argument("--skip-spatial", action="store_true", help="skip Spatial-TTT")
    parser.add_argument(
        "--skip-spatial-control",
        action="store_true",
        help="do not run Spatial-TTT again with --without-memory",
    )
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
        question = (
            "Answer in one concise sentence. Do not repeat items.\n"
            + question
        )

    methods: list[Method] = []
    if not args.skip_base:
        # The videoqa loader needs a checkpoint directory even when the adapter
        # is disabled; this is the frozen-base control for the same base model.
        methods.append(Method("Frozen base (--without-memory)", "ttt_frame.videoqa",
                              Path(args.codex_memory), use_memory=False))
    methods.extend([
        Method("LoRA + local teacher", "ttt_frame.videoqa", Path(args.local_memory)),
        Method("LoRA + Codex teacher", "ttt_frame.videoqa", Path(args.codex_memory)),
    ])
    if not args.skip_spatial:
        methods.append(Method("Spatial-TTT + memory", "ttt_frame.spatial_videoqa",
                              Path(args.spatial_memory)))
        if not args.skip_spatial_control:
            methods.append(Method(
                "Spatial-TTT same checkpoint (--without-memory)",
                "ttt_frame.spatial_videoqa",
                Path(args.spatial_memory),
                use_memory=False,
            ))

    print(f"Question: {question}")
    print(f"Generation budget: {args.max_new_tokens} new tokens")
    if not args.skip_spatial and not args.skip_spatial_control:
        print("Spatial pair: the second run bypasses fast-weight memory only; "
              "it still loads the same Spatial-TTT checkpoint.")
    failures = 0
    for method in methods:
        failures += run_method(method, args, question) != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
