#!/usr/bin/env python3
"""Safely schedule the nine-video matched SVCBench run on selected GPUs.

The launcher deliberately has no default GPU list.  The caller must name the
physical GPUs that are available, and the launcher refuses GPUs with existing
compute processes unless ``--allow-busy-gpus`` is explicit.  It never sends a
signal to, or otherwise manages, an existing GPU process.

Workers receive one physical GPU through ``CUDA_VISIBLE_DEVICES`` and use
``cuda:0`` inside that restricted view.  Each shard owns one source video, so
completed workers can immediately take the next unstarted shard.

Example::

    conda run --no-capture-output -n meowbench python scripts/run_svcbench_natural9_multigpu.py --gpus 0,1,4,5,7

Use ``--plan-only`` to validate paths, videos, existing outputs, and sharding
without loading a model or starting a GPU worker.  A shard is skipped only when
its JSONL and per-shard summary agree on the protocol, hash, row count, arm
coverage, and absence of runtime errors.  An incomplete or incompatible output
causes a fail-closed exit; move it aside before intentionally rerunning it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "evaluate_svcbench_matched.py"
DEFAULT_ANNOTATIONS = Path(
    "/data/quzitsix/svcbench_real_natural9_v1/subset/svcbench_subset.json"
)
DEFAULT_VIDEO_ROOT = Path("/data/quzitsix/svcbench_real_natural9_v1")
DEFAULT_MODEL = Path("/data/quzitsix/models/Qwen3-VL-2B-Instruct")
DEFAULT_CHECKPOINT = Path(
    "/data/quzitsix/models/Spatial-TTT-nano/model.safetensors"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "svcbench_real_natural9_matched_v1"
CONDITIONS = ("blind", "icl", "spatial", "spatial_slow_only", "spatial_base")
SUMMARY_SCHEMA = "ttt_frame.svcbench_matched_summary/1"


@dataclass(frozen=True)
class ShardPaths:
    output: Path
    summary: Path
    log: Path


@dataclass
class Worker:
    process: subprocess.Popen[bytes]
    gpu: int
    shard: int
    log_handle: Any
    paths: ShardPaths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_gpu_list(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError("--gpus must be a comma-separated list, e.g. 0,1,4")
    try:
        gpus = [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("every --gpus entry must be a non-negative integer") from exc
    if any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("every --gpus entry must be a non-negative integer")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus must not contain duplicate indices")
    return gpus


def _shard_paths(output_dir: Path, shard: int) -> ShardPaths:
    return ShardPaths(
        output=output_dir / f"shard{shard}.jsonl",
        summary=output_dir / f"shard{shard}_summary.json",
        log=output_dir / "logs" / f"shard{shard}.log",
    )


def _sampling(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "decoder": "ttt_frame.video.iter_video_chunks",
        "chunk_seconds": args.chunk_seconds,
        "frames_per_chunk": args.frames_per_chunk,
        "prefix_selection": "uniform_over_all_sampled_frames",
        "max_frames": args.max_frames,
        "max_side": args.max_side,
        "cutoff_margin": args.cutoff_margin,
        "write_chunk_frames": args.write_chunk_frames,
        "query_stride": args.query_stride,
        "max_query_points": args.max_query_points,
    }


def _expected_protocol(args: argparse.Namespace, shard: int) -> dict[str, Any]:
    return {
        "runner_sha256": _sha256_file(RUNNER),
        "annotations_sha256": _sha256_file(args.annotations),
        "annotations": str(args.annotations.resolve()),
        "video_root": str(args.video_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "spatial_checkpoint": str(args.spatial_checkpoint.resolve()),
        "conditions": list(CONDITIONS),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "sampling": _sampling(args),
        "num_shards": args.num_shards,
        "shard_index": shard,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _completion_status(
    paths: ShardPaths, expected_protocol: dict[str, Any],
) -> tuple[str, str]:
    """Return ``missing``, ``complete``, or ``blocked`` for one shard."""

    output_exists = paths.output.is_file()
    summary_exists = paths.summary.is_file()
    if not output_exists and not summary_exists:
        return "missing", "no prior output"
    if not output_exists or not summary_exists:
        present = paths.output if output_exists else paths.summary
        missing = paths.summary if output_exists else paths.output
        return "blocked", f"partial artifact: found {present}, missing {missing}"

    try:
        summary = json.loads(paths.summary.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("summary is not a JSON object")
        if summary.get("schema") != SUMMARY_SCHEMA:
            raise ValueError(f"unexpected summary schema {summary.get('schema')!r}")
        protocol = summary.get("run_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("summary has no run_protocol object")
        for key, expected in expected_protocol.items():
            if protocol.get(key) != expected:
                raise ValueError(
                    f"protocol mismatch for {key}: expected {expected!r}, "
                    f"found {protocol.get(key)!r}"
                )
        expected_rows = summary.get("expected_rows")
        if not isinstance(expected_rows, int) or expected_rows <= 0:
            raise ValueError("summary expected_rows is not a positive integer")
        if summary.get("n_rows") != expected_rows:
            raise ValueError(
                f"summary is incomplete: n_rows={summary.get('n_rows')!r}, "
                f"expected_rows={expected_rows}"
            )
        actual_hash = _sha256_file(paths.output)
        if summary.get("output_sha256") != actual_hash:
            raise ValueError("JSONL SHA-256 differs from the per-shard summary")
        rows = _read_jsonl(paths.output)
        if len(rows) != expected_rows:
            raise ValueError(f"JSONL has {len(rows)} rows, expected {expected_rows}")

        by_episode: dict[str, list[dict[str, Any]]] = {}
        for index, row in enumerate(rows, 1):
            row_protocol = row.get("run_protocol")
            if not isinstance(row_protocol, dict):
                raise ValueError(f"row {index} has no run_protocol object")
            if row_protocol.get("comparison_fingerprint") != protocol.get(
                "comparison_fingerprint"
            ):
                raise ValueError(f"row {index} has a different comparison fingerprint")
            if row.get("status") not in {"ok", "parse_error"}:
                raise ValueError(
                    f"row {index} has an incomplete or runtime-error status "
                    f"{row.get('status')!r}"
                )
            if row_protocol.get("num_shards") != expected_protocol["num_shards"]:
                raise ValueError(f"row {index} has a different num_shards value")
            if row_protocol.get("shard_index") != expected_protocol["shard_index"]:
                raise ValueError(f"row {index} belongs to another shard")
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"row {index} has no episode_id")
            by_episode.setdefault(episode_id, []).append(row)
        for episode_id, episode_rows in by_episode.items():
            observed = [str(row.get("condition")) for row in episode_rows]
            if len(observed) != len(CONDITIONS) or set(observed) != set(CONDITIONS):
                raise ValueError(
                    f"episode {episode_id!r} does not contain each of the five arms once"
                )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "blocked", str(exc)
    return "complete", f"validated {expected_rows} rows and five-arm coverage"


def _common_runner_args(args: argparse.Namespace, shard: int, paths: ShardPaths) -> list[str]:
    return [
        str(RUNNER),
        "--annotations", str(args.annotations),
        "--video-root", str(args.video_root),
        "--model-path", str(args.model_path),
        "--spatial-checkpoint", str(args.spatial_checkpoint),
        "--device", "cuda:0",
        "--dtype", args.dtype,
        "--max-side", str(args.max_side),
        "--chunk-seconds", str(args.chunk_seconds),
        "--frames-per-chunk", str(args.frames_per_chunk),
        "--max-frames", str(args.max_frames),
        "--write-chunk-frames", str(args.write_chunk_frames),
        "--cutoff-margin", str(args.cutoff_margin),
        "--max-new-tokens", str(args.max_new_tokens),
        "--query-stride", str(args.query_stride),
        "--max-query-points", str(args.max_query_points),
        "--num-shards", str(args.num_shards),
        "--shard-index", str(shard),
        "--output", str(paths.output),
        "--summary-output", str(paths.summary),
    ]


def _preflight_shards(args: argparse.Namespace, shards: Iterable[int]) -> None:
    for shard in shards:
        paths = _shard_paths(args.output_dir, shard)
        command = [sys.executable, *_common_runner_args(args, shard, paths), "--dry-run"]
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"dry-run failed for shard {shard}: {detail}")
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"dry-run returned invalid JSON for shard {shard}") from exc
        if report.get("selected_videos") != 1:
            raise RuntimeError(
                f"shard {shard} selects {report.get('selected_videos')} videos; "
                "Natural9 expects exactly one video per shard"
            )
        print(
            f"PREFLIGHT shard={shard} video={report['videos'][0]['video_key']} "
            f"queries={report['selected_query_points']} rows={report['expected_rows']}",
            flush=True,
        )


def _query_gpu_processes() -> dict[int, list[str]]:
    """Return compute processes keyed by physical GPU index."""

    gpu_query = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True,
    )
    if gpu_query.returncode:
        raise RuntimeError(f"nvidia-smi GPU query failed: {gpu_query.stderr.strip()}")
    uuid_to_index: dict[str, int] = {}
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2:
            uuid_to_index[fields[1]] = int(fields[0])

    process_query = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True,
    )
    if process_query.returncode:
        raise RuntimeError(f"nvidia-smi process query failed: {process_query.stderr.strip()}")
    processes: dict[int, list[str]] = {}
    for line in process_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or fields[0] not in uuid_to_index:
            continue
        processes.setdefault(uuid_to_index[fields[0]], []).append(
            f"pid={fields[1]} process={fields[2]}"
        )
    return processes


def _validate_selected_gpus(gpus: list[int], *, allow_busy: bool) -> None:
    processes = _query_gpu_processes()
    known = set()
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    if query.returncode:
        raise RuntimeError(f"nvidia-smi GPU index query failed: {query.stderr.strip()}")
    for line in query.stdout.splitlines():
        if line.strip():
            known.add(int(line.strip()))
    unknown = [gpu for gpu in gpus if gpu not in known]
    if unknown:
        raise RuntimeError(f"unknown physical GPU indices: {unknown}; available: {sorted(known)}")
    busy = {gpu: processes[gpu] for gpu in gpus if processes.get(gpu)}
    if busy and not allow_busy:
        details = "; ".join(f"GPU {gpu}: {', '.join(entries)}" for gpu, entries in busy.items())
        raise RuntimeError(
            f"selected GPUs already have compute processes ({details}); choose idle GPUs or "
            "pass --allow-busy-gpus only when sharing is intentional"
        )
    if busy:
        print(f"WARNING selected busy GPUs by explicit override: {busy}", file=sys.stderr)


def _launch(args: argparse.Namespace, gpu: int, shard: int) -> Worker:
    paths = _shard_paths(args.output_dir, shard)
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = paths.log.open("wb")
    command = [sys.executable, *_common_runner_args(args, shard, paths)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise
    print(
        f"START shard={shard} physical_gpu={gpu} pid={process.pid} log={paths.log}",
        flush=True,
    )
    return Worker(process, gpu, shard, log_handle, paths)


def _run_workers(args: argparse.Namespace, pending: list[int]) -> int:
    queue = list(pending)
    available = list(args.gpus)
    running: list[Worker] = []
    failures: list[tuple[int, int, str]] = []

    while queue or running:
        while queue and available:
            gpu = available.pop(0)
            running.append(_launch(args, gpu, queue.pop(0)))

        if not running:
            break
        time.sleep(1.0)
        for worker in list(running):
            returncode = worker.process.poll()
            if returncode is None:
                continue
            worker.log_handle.close()
            running.remove(worker)
            available.append(worker.gpu)
            status, reason = _completion_status(
                worker.paths, _expected_protocol(args, worker.shard),
            )
            if returncode == 0 and status == "complete":
                print(
                    f"DONE shard={worker.shard} physical_gpu={worker.gpu}: {reason}",
                    flush=True,
                )
            else:
                failures.append((worker.shard, returncode, reason))
                print(
                    f"FAILED shard={worker.shard} physical_gpu={worker.gpu} "
                    f"exit={returncode}: {reason}; see {worker.paths.log}",
                    file=sys.stderr, flush=True,
                )

    if failures:
        print(f"{len(failures)} shard(s) failed validation: {failures}", file=sys.stderr)
        return 1
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpus", required=True, type=_parse_gpu_list,
        help="comma-separated physical GPU indices selected by the user; no implicit default",
    )
    parser.add_argument(
        "--allow-busy-gpus", action="store_true",
        help="allow selected GPUs with existing compute processes; never stops those processes",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--spatial-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shards", type=int, default=9)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-side", type=int, default=224)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--write-chunk-frames", type=int, default=8)
    parser.add_argument("--cutoff-margin", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--query-stride", type=int, default=1)
    parser.add_argument("--max-query-points", type=int, default=0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_shards != 9:
        raise ValueError("Natural9 requires --num-shards 9 so each shard owns one source video")
    if not RUNNER.is_file():
        raise ValueError(f"matched evaluator does not exist: {RUNNER}")
    if not args.annotations.is_file():
        raise ValueError(f"annotations do not exist: {args.annotations}")
    if not args.video_root.is_dir():
        raise ValueError(f"video root does not exist: {args.video_root}")
    if not args.model_path.exists():
        raise ValueError(f"model does not exist: {args.model_path}")
    if not args.spatial_checkpoint.is_file():
        raise ValueError(f"Spatial-TTT checkpoint does not exist: {args.spatial_checkpoint}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_args(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pending: list[int] = []
        blocked: list[tuple[int, str]] = []
        for shard in range(args.num_shards):
            paths = _shard_paths(args.output_dir, shard)
            status, reason = _completion_status(
                paths, _expected_protocol(args, shard),
            )
            print(f"STATE shard={shard} status={status}: {reason}", flush=True)
            if status == "missing":
                pending.append(shard)
            elif status == "blocked":
                blocked.append((shard, reason))
        if blocked:
            detail = "\n".join(f"  shard {shard}: {reason}" for shard, reason in blocked)
            raise RuntimeError(
                "existing shard artifacts are incomplete or use another protocol; "
                "they were left untouched:\n" + detail
            )

        _preflight_shards(args, pending)
        if args.plan_only:
            print(
                f"PLAN complete={args.num_shards - len(pending)} pending={pending} "
                f"selected_gpus={args.gpus} (no workers started)",
                flush=True,
            )
            return 0
        if not pending:
            print("All nine shards are already complete; no workers started.", flush=True)
            return 0
        _validate_selected_gpus(args.gpus, allow_busy=args.allow_busy_gpus)
        return _run_workers(args, pending)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
