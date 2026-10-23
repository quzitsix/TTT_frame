#!/usr/bin/env python3
"""Launch the fixed forecasting MCQ protocol across explicitly selected GPUs.

Each model family has its own independent sharding. Devices may repeat, including
across families; the caller is responsible for GPU memory and availability. This
launcher never discovers, stops or manages unrelated GPU processes. Only its own
children are terminated on interruption or failure. Resume requires the original
launch plan and delegates completed-episode validation to the inference runner.

Dry-run validates the prepared protocol JSON and local paths without importing
the inference runner, loading models, or creating output directories. Actual
workers independently verify source and checkpoint hashes before inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "evaluate_forecast_mcq.py"
SCORER = ROOT / "scripts" / "summarize_forecast_mcq.py"
ARMS = {
    "qwen": ["qwen_blind", "qwen_recent_icl", "qwen_full_icl"],
    "spatial": ["blind", "recent_icl", "full_icl", "spatial", "spatial_slow_only", "spatial_base"],
}


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _time() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def build_plan(args: argparse.Namespace) -> dict:
    """Validate paths and deterministically plan independent family shards."""
    protocol_path = args.protocol.resolve()
    prepared = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload = prepared.get("payload", {})
    if (payload.get("schema") != "forecast_mcq_protocol_v1"
            or _fingerprint(payload) != prepared.get("protocol_fingerprint")):
        raise ValueError("prepared protocol schema or fingerprint mismatch")
    source_root = Path(payload["root"]).resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(source_root):
        raise ValueError("output directory must be outside the source benchmark")
    required_files = [source_root / "inputs_mcq.jsonl", Path(payload["spatial_checkpoint"]),
                      RUNNER, SCORER]
    if args.split_manifest:
        required_files.append(args.split_manifest)
        json.loads(args.split_manifest.read_text(encoding="utf-8"))
    for path in required_files:
        if not path.is_file():
            raise ValueError(f"required file does not exist: {path}")
    if not source_root.is_dir() or not Path(payload["model_path"]).is_dir():
        raise ValueError("benchmark root or model directory does not exist")
    case_ids = payload.get("case_ids", [])
    if (not case_ids or any(not isinstance(sample_id, str) for sample_id in case_ids)
            or len(set(case_ids)) != len(case_ids)):
        raise ValueError("prepared protocol requires distinct, nonempty case IDs")
    if set(payload.get("arms", [])) != set(ARMS["qwen"] + ARMS["spatial"]):
        raise ValueError("prepared protocol must contain the nine expected arms")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    family_devices = {"qwen": args.qwen_devices, "spatial": args.spatial_devices}
    for family, devices in family_devices.items():
        if (not devices or len(devices) > len(case_ids)
                or any(not re.fullmatch(r"cuda:\d+", device) for device in devices)):
            raise ValueError(f"invalid {family} devices or more shards than cases")
    split_path = str(args.split_manifest.resolve()) if args.split_manifest else None
    identity = {
        "schema": "forecast_mcq_launch_identity_v1",
        "protocol_path": str(protocol_path), "protocol_fingerprint": prepared["protocol_fingerprint"],
        "output_dir": str(output_dir), "family_devices": family_devices,
        "threads": args.threads, "split_manifest": split_path,
        "split_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
            if args.split_manifest else None,
    }
    if output_dir.exists():
        if not args.resume:
            raise ValueError("output directory exists; use --resume with its original launch plan")
        if not output_dir.is_dir():
            raise ValueError("output path is not a directory")
        existing_plan = output_dir / "plan.json"
        if not existing_plan.is_file():
            raise ValueError("resume requires the original output plan.json")
        old = json.loads(existing_plan.read_text(encoding="utf-8"))
        if old.get("identity") != identity:
            raise ValueError("resume configuration differs from the saved launch plan")
        if (output_dir / "report").exists():
            raise ValueError("report already exists; it will not be overwritten or summarized again")
    workers = []
    for family, devices in family_devices.items():
        for index, device in enumerate(devices):
            name = f"{family}_shard{index:02d}"
            output = output_dir / f"{name}.jsonl"
            command = [sys.executable, "-u", str(RUNNER), "--protocol", str(protocol_path),
                       "--family", family, "--device", device,
                       "--num-shards", str(len(devices)), "--shard-index", str(index),
                       "--threads", str(args.threads), "--output", str(output)]
            if args.resume:
                command.append("--resume")
            selected = case_ids[index::len(devices)]
            workers.append({
                "name": name, "family": family, "device": device,
                "num_shards": len(devices), "shard_index": index,
                "case_ids": selected, "arms": ARMS[family],
                "expected_rows": len(selected) * len(ARMS[family]),
                "output": str(output), "log": str(output_dir / "logs" / f"{name}.log"),
                "command": command,
            })
    score_command = [sys.executable, "-u", str(SCORER), "--root", str(source_root),
                     "--predictions", *[worker["output"] for worker in workers],
                     "--expected-arms", *payload["arms"],
                     "--output-dir", str(output_dir / "report")]
    if args.split_manifest:
        score_command.extend(["--split-manifest", split_path])
    return {
        "schema": "forecast_mcq_launch_plan_v1", "identity": identity,
        "protocol_fingerprint": prepared["protocol_fingerprint"],
        "case_count": len(case_ids), "expected_rows": len(case_ids) * 9,
        "device_policy": "Explicit device list; repeated devices allowed; no unrelated process management",
        "workers": workers, "score_command": score_command,
        "scorer_log": str(output_dir / "logs" / "summarize.log"),
    }


def _progress(path: Path) -> dict:
    rows, malformed = 0, 0
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.endswith("\n"):
                    continue  # worker may currently be writing the final line
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    rows += 1
                except json.JSONDecodeError:
                    malformed += 1
    return {"written_rows": rows, "malformed_complete_lines": malformed,
            "output_bytes": path.stat().st_size if path.is_file() else 0}


def validate_completed_shard(worker: dict, fingerprint: str) -> None:
    """A zero child exit is insufficient without exact successful row coverage."""
    expected = {(sample_id, arm) for sample_id in worker["case_ids"] for arm in worker["arms"]}
    seen = set()
    path = Path(worker["output"])
    if not path.is_file():
        raise ValueError(f"missing output for {worker['name']}: {path}")
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("id"), row.get("arm"))
            if (key not in expected or key in seen
                    or row.get("schema") != "forecast_mcq_prediction_v1"
                    or row.get("protocol_fingerprint") != fingerprint
                    or row.get("status") != "ok"
                    or row.get("prediction") not in tuple("ABCDEF")):
                raise ValueError(f"invalid, duplicate or failed row in {worker['name']}: {key}")
            seen.add(key)
    if seen != expected:
        raise ValueError(f"incomplete {worker['name']}: {len(seen)}/{len(expected)} rows")


def _stop_owned(children: list[tuple[Any, Any]]) -> None:
    """Signal only Popen handles created by this launcher."""
    for process, _ in children:
        if process.poll() is None:
            process.terminate()
    for process, _ in children:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def run_plan(plan: dict, *, resume: bool = False, popen=None, sleep=None) -> int:
    popen = popen or subprocess.Popen
    sleep = sleep or time.sleep
    output_dir = Path(plan["identity"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=resume)
    (output_dir / "logs").mkdir(exist_ok=resume)
    if not resume or not (output_dir / "plan.json").exists():
        _write_json(output_dir / "plan.json", plan)
    status = {
        "schema": "forecast_mcq_launch_status_v1", "state": "launching", "started_at": _time(),
        "updated_at": _time(), "protocol_fingerprint": plan["protocol_fingerprint"],
        "expected_rows": plan["expected_rows"], "workers": [], "scorer": None,
    }
    children: list[tuple[Any, Any]] = []
    def update():
        status["updated_at"] = _time()
        for record in status["workers"]:
            record.update(_progress(Path(record["output"])))
        status["written_rows"] = sum(record["written_rows"] for record in status["workers"])
        _write_json(output_dir / "status.json", status)

    try:
        update()
        for worker in plan["workers"]:
            log = Path(worker["log"]).open("a" if resume else "x", encoding="utf-8")
            try:
                process = popen(worker["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            children.append((process, log))
            status["workers"].append({**worker, "pid": process.pid, "exit_code": None,
                                       "state": "running"})
            update()
        status["state"] = "running"
        while True:
            active = False
            failed = []
            for (process, _), worker in zip(children, status["workers"]):
                code = process.poll()
                worker["exit_code"] = code
                worker["state"] = "running" if code is None else "completed" if code == 0 else "failed"
                active |= code is None
                if code is not None and code != 0:
                    failed.append(f"{worker['name']} exited {code}; log: {worker['log']}")
            update()
            if failed:
                raise RuntimeError("; ".join(failed))
            if not active:
                break
            sleep(2)
        for worker in plan["workers"]:
            try:
                validate_completed_shard(worker, plan["protocol_fingerprint"])
            except (ValueError, OSError) as exc:
                raise RuntimeError(f"{exc}; log: {worker['log']}") from exc
        status["state"] = "scoring"
        log = Path(plan["scorer_log"]).open("a" if resume else "x", encoding="utf-8")
        try:
            process = popen(plan["score_command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            raise
        children.append((process, log))
        status["scorer"] = {"command": plan["score_command"], "pid": process.pid,
                            "log": plan["scorer_log"], "exit_code": None}
        while process.poll() is None:
            update()
            sleep(2)
        status["scorer"]["exit_code"] = process.returncode
        if process.returncode != 0:
            raise RuntimeError(f"scorer exited {process.returncode}; log: {plan['scorer_log']}")
        summary = json.loads((output_dir / "report" / "summary.json").read_text())
        if summary.get("complete") is not True:
            raise RuntimeError("scorer report is incomplete; refusing to mark the launch successful")
        status["state"] = "completed"
        status["finished_at"] = _time()
        update()
        print(f"All {plan['expected_rows']} predictions scored: {output_dir / 'report' / 'results.md'}",
              flush=True)
        return 0
    except BaseException as exc:
        _stop_owned(children)
        status["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        for (process, _), worker in zip(children, status["workers"]):
            worker["exit_code"] = process.poll()
            if worker["state"] == "running":
                worker["state"] = "terminated_by_launcher"
        if status["scorer"] is not None:
            status["scorer"]["exit_code"] = children[-1][0].poll()
        status["finished_at"] = _time()
        update()
        raise
    finally:
        for _, log in children:
            log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-devices", nargs="+", required=True,
                        help="one worker per entry; repeated cuda:N entries are allowed")
    parser.add_argument("--spatial-devices", nargs="+", required=True,
                        help="one worker per entry; repeated cuda:N entries are allowed")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        return run_plan(plan, resume=args.resume)
    except KeyboardInterrupt:
        print("Interrupted; only this launcher's child processes were stopped.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
