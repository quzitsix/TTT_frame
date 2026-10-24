#!/usr/bin/env python3
"""Run independent Idea3 shards on one available GPU and merge only complete runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_idea3_spatial import atomic_json, completed, validate_protocol
from scripts.evaluate_forecast_mcq import file_hash


def merge(protocol: Path, directory: Path, workers: int) -> Path:
    record = validate_protocol(protocol)
    ids = record["payload"]["baseline_protocol"]["payload"]["case_ids"]
    digest = record["protocol_fingerprint"]
    collected, sources = {}, {}
    for index in range(workers):
        path = directory / f"scores_{index}.jsonl"
        selected = set(ids[index::workers])
        if completed(path, selected, digest) != selected:
            raise ValueError(f"shard {index} is incomplete")
        completion = json.loads(path.with_suffix(".completion.json").read_text())
        if (completion.get("complete") is not True or completion.get("test_answers_loaded") is not False
                or completion["protocol_fingerprint"] != digest
                or completion["scores_sha256"] != file_hash(path)):
            raise ValueError(f"shard {index} completion hash mismatch")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["id"] in collected:
                raise ValueError("duplicate merged case")
            collected[row["id"]] = row
        sources[path.name] = file_hash(path)
    destination = directory / "scores.jsonl"
    text = "".join(json.dumps(collected[sid], ensure_ascii=False, allow_nan=False) + "\n" for sid in ids)
    if destination.exists() and destination.read_text() != text:
        raise ValueError("existing combined scores differ")
    temporary = destination.with_suffix(".jsonl.tmp")
    temporary.write_text(text)
    temporary.replace(destination)
    atomic_json(directory / "scores.completion.json", {
        "complete": True, "rows": len(ids), "test_answers_loaded": False,
        "protocol_fingerprint": digest, "scores_sha256": file_hash(destination), "shards": sources,
    })
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    record = validate_protocol(args.protocol)
    base = record["payload"]["baseline_protocol"]["payload"]
    if args.output_dir.resolve().is_relative_to(Path(base["root"]).resolve()):
        parser.error("output directory must be outside benchmark")
    if args.workers > len(base["case_ids"]):
        parser.error("workers exceed cases")
    if args.merge_only:
        print(merge(args.protocol, args.output_dir, args.workers))
        return 0
    import torch
    free, total = torch.cuda.mem_get_info(args.device)
    if free < args.workers * 8 * 2**30:
        raise RuntimeError(f"need 8 GiB free per worker; {free / 2**30:.1f} GiB currently free")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launch_path = args.output_dir / "launch.json"
    if launch_path.exists() and not args.resume:
        raise FileExistsError("existing launch requires --resume")
    started = time.perf_counter()
    processes, logs, commands = [], [], []
    try:
        for index in range(args.workers):
            command = [sys.executable, str(REPO / "scripts/run_idea3_spatial.py"),
                       "--protocol", str(args.protocol.resolve()),
                       "--output", str((args.output_dir / f"scores_{index}.jsonl").resolve()),
                       "--device", args.device, "--num-shards", str(args.workers),
                       "--shard-index", str(index), "--threads", str(args.threads)]
            if args.resume:
                command.append("--resume")
            log = (args.output_dir / f"worker_{index}.log").open("a" if args.resume else "x")
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
            commands.append(command)
        atomic_json(launch_path, {"commands": commands, "device": args.device, "workers": args.workers,
                                 "gpu_free_bytes_at_launch": free, "gpu_total_bytes": total,
                                 "protocol_fingerprint": record["protocol_fingerprint"],
                                 "launcher_sha256": file_hash(Path(__file__)), "pids": [p.pid for p in processes]})
        while any(process.poll() is None for process in processes):
            failures = [i for i, process in enumerate(processes) if process.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f"failed workers {failures}; inspect worker logs")
            counts = []
            for index in range(args.workers):
                path = args.output_dir / f"scores_{index}.jsonl"
                counts.append(len(path.read_text().splitlines()) if path.exists() else 0)
            print(json.dumps({"completed": sum(counts), "by_worker": counts,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
            time.sleep(10)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError("worker failed; inspect worker logs")
        combined = merge(args.protocol, args.output_dir, args.workers)
        atomic_json(args.output_dir / "launch.completion.json", {
            "complete": True, "elapsed_seconds": time.perf_counter() - started,
            "workers": args.workers, "device": args.device,
            "scores_sha256": file_hash(combined), "launcher_sha256": file_hash(Path(__file__)),
        })
        print(f"Complete: {combined}", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        for log in logs:
            log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
