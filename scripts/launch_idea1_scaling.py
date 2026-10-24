#!/usr/bin/env python3
"""Run the fixed 12-trial scaling plan across spare memory on eight GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_idea1_scaling import atomic_json


def schedules() -> dict[int, list[dict]]:
    # Six small jobs use <2 GiB per worker; large single-step trials use the
    # two cards with more spare memory. The expensive 4-step trial is divided
    # between the remaining six cards after their first jobs finish.
    assignments = {
        0: ["h128_random_u1", "h128_pre200_u1"],
        1: ["h128_random_u4", "h128_pre800_u1"],
        4: ["h512_random_u1"], 5: ["h512_random_u4"],
        6: ["h512_pre200_u1"], 7: ["h512_pre800_u1"],
    }
    jobs = {gpu: [{"trials": trials, "batch_size": 4, "num_shards": 1, "shard_index": 0}]
            for gpu, trials in assignments.items()}
    for shard_index, gpu in enumerate((0, 1, 4, 5, 6, 7)):
        jobs[gpu].append({"trials": ["h2048_random_u4"], "batch_size": 4,
                          "num_shards": 6, "shard_index": shard_index})
    for shard_index, gpu in enumerate((2, 3)):
        jobs[gpu] = [{"trials": ["h2048_random_u1", "h2048_pre200_u1", "h2048_pre800_u1"],
                      "batch_size": 8, "num_shards": 2, "shard_index": shard_index}]
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/forecast_bench_v0/idea01_scaling_v1"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    plan = json.loads((output / "plan.json").read_text())
    missing = [trial["configuration"]["initialization_checkpoint"] for trial in plan["trials"]
               if trial["configuration"]["initialization_checkpoint"] is not None
               and not Path(trial["configuration"]["initialization_checkpoint"]).exists()]
    if missing:
        raise ValueError(f"pretraining not complete: {missing}")
    memory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    free_mib = {int(line.split(",")[0]): int(line.split(",")[1]) for line in memory.splitlines()}
    jobs = schedules()
    if any(free_mib.get(gpu, 0) < 2300 for gpu in jobs):
        raise ValueError(f"need at least 2300 MiB spare per selected GPU; current {free_mib}")
    logs = output / "scaling_logs"
    logs.mkdir(exist_ok=True)
    status = {"schema": "idea01_scaling_gpu_execution_v1", "started_unix": time.time(),
              "free_mib_at_start": free_mib, "jobs": {}, "complete": False}
    lock = threading.Lock()

    def worker(gpu: int, assignments: list[dict]) -> None:
        for ordinal, job in enumerate(assignments):
            cmd = [sys.executable, "-u", str(ROOT / "scripts/run_idea1_scaling.py"),
                   "--output-dir", str(output), "--device", f"cuda:{gpu}",
                   "--batch-size", str(job["batch_size"]), "--threads", str(args.threads),
                   "--num-shards", str(job["num_shards"]), "--shard-index", str(job["shard_index"]),
                   "--resume", "--trials", *job["trials"]]
            log = logs / f"cuda{gpu}_job{ordinal}.log"
            key = f"cuda{gpu}_job{ordinal}"
            env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
            with log.open("w") as stream:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                with lock:
                    status["jobs"][key] = {**job, "pid": process.pid, "command": cmd,
                                            "log": str(log), "started_unix": time.time()}
                    atomic_json(output / "execution.json", status)
                code = process.wait()
            with lock:
                status["jobs"][key].update(exit_code=code, finished_unix=time.time())
                atomic_json(output / "execution.json", status)
            if code:
                raise RuntimeError(f"{key} failed with {code}; inspect {log}")

    errors = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(worker, gpu, assignments): gpu for gpu, assignments in jobs.items()}
        for future in as_completed(futures):
            try:
                future.result()
                print(f"GPU {futures[future]} assignments complete", flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(str(exc), flush=True)
    status.update(complete=not errors, errors=errors, finished_unix=time.time())
    atomic_json(output / "execution.json", status)
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
