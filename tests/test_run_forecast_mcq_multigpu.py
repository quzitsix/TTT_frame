"""Launcher sharding, dry-run and owned-child failure handling without GPUs."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_forecast_mcq_multigpu import ARMS, build_plan, main, run_plan


def arguments(tmp_path):
    root = tmp_path / "bench"
    root.mkdir()
    (root / "inputs_mcq.jsonl").touch()
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.touch()
    payload = {
        "schema": "forecast_mcq_protocol_v1", "root": str(root),
        "case_ids": [f"egolife_{i:03d}" for i in range(1, 6)],
        "arms": ARMS["qwen"] + ARMS["spatial"],
        "model_path": str(model), "spatial_checkpoint": str(checkpoint),
        "source_hashes": {"intentionally_not_checked_by_launcher": "stale"},
    }
    protocol = tmp_path / "protocol.json"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    protocol.write_text(json.dumps({"payload": payload, "protocol_fingerprint": digest}))
    return Namespace(protocol=protocol, output_dir=tmp_path / "run", qwen_devices=["cuda:2", "cuda:2"],
                     spatial_devices=["cuda:4", "cuda:5", "cuda:4"], split_manifest=None,
                     threads=4, resume=False, dry_run=False)


def test_independent_shards_and_repeated_devices(tmp_path):
    args = arguments(tmp_path)
    plan = build_plan(args)
    assert plan["expected_rows"] == 45
    assert [len(worker["case_ids"]) for worker in plan["workers"]] == [3, 2, 2, 2, 1]
    assert [worker["num_shards"] for worker in plan["workers"]] == [2, 2, 3, 3, 3]
    for family in ("qwen", "spatial"):
        all_ids = [sample_id for worker in plan["workers"] if worker["family"] == family
                   for sample_id in worker["case_ids"]]
        assert len(all_ids) == len(set(all_ids)) == 5
    assert not args.output_dir.exists()


def test_dry_run_does_not_create_outputs_or_check_runner_source_hash(tmp_path):
    args = arguments(tmp_path)
    code = main(["--protocol", str(args.protocol), "--output-dir", str(args.output_dir),
                 "--qwen-devices", "cuda:2", "--spatial-devices", "cuda:4", "--dry-run"])
    assert code == 0
    assert not args.output_dir.exists()


def test_source_output_and_empty_shards_rejected(tmp_path):
    args = arguments(tmp_path)
    args.output_dir = tmp_path / "bench" / "results"
    with pytest.raises(ValueError, match="outside"):
        build_plan(args)
    args.output_dir = tmp_path / "run"
    args.qwen_devices = ["cuda:2"] * 6
    with pytest.raises(ValueError, match="more shards"):
        build_plan(args)


class FakeProcess:
    def __init__(self, pid, code):
        self.pid, self.returncode, self.terminated = pid, code, False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout):
        return self.returncode


def fake_factory(plan, *, failure=None, partial=None, hold=False):
    processes, commands = [], []
    def spawn(command, **kwargs):
        commands.append(command)
        if "--family" in command:
            output = command[command.index("--output") + 1]
            worker = next(worker for worker in plan["workers"] if worker["output"] == output)
            rows = [
                {"schema": "forecast_mcq_prediction_v1", "id": sample_id, "arm": arm,
                 "prediction": "A", "status": "ok", "protocol_fingerprint": plan["protocol_fingerprint"]}
                for sample_id in worker["case_ids"] for arm in worker["arms"]
            ]
            if worker["name"] == partial:
                rows = rows[:-1]
            Path(output).write_text("".join(json.dumps(row) + "\n" for row in rows))
            code = 7 if worker["name"] == failure else None if hold else 0
        else:
            report = Path(command[command.index("--output-dir") + 1])
            report.mkdir()
            (report / "summary.json").write_text(json.dumps({"complete": True}))
            code = 0
        process = FakeProcess(100 + len(processes), code)
        processes.append(process)
        return process
    return spawn, processes, commands


def test_success_scores_only_after_all_shards_complete(tmp_path):
    args = arguments(tmp_path)
    plan = build_plan(args)
    spawn, processes, commands = fake_factory(plan)
    assert run_plan(plan, popen=spawn, sleep=lambda _: None) == 0
    assert len(commands) == len(plan["workers"]) + 1
    assert "--predictions" in commands[-1]
    status = json.loads((args.output_dir / "status.json").read_text())
    assert status["state"] == "completed" and status["written_rows"] == 45
    assert all(row["exit_code"] == 0 and row["pid"] for row in status["workers"])
    assert status["scorer"]["exit_code"] == 0
    assert not any(process.terminated for process in processes)


def test_failed_shard_stops_only_owned_active_children_without_scoring(tmp_path):
    args = arguments(tmp_path)
    plan = build_plan(args)
    spawn, processes, commands = fake_factory(plan, failure="qwen_shard00", hold=True)
    with pytest.raises(RuntimeError, match="qwen_shard00 exited 7; log:"):
        run_plan(plan, popen=spawn, sleep=lambda _: None)
    assert len(commands) == len(plan["workers"])
    assert all(process.terminated for process in processes[1:])
    status = json.loads((args.output_dir / "status.json").read_text())
    assert status["state"] == "failed" and status["scorer"] is None
    assert not (args.output_dir / "report").exists()


def test_zero_exit_with_partial_output_is_failure(tmp_path):
    args = arguments(tmp_path)
    plan = build_plan(args)
    spawn, _, commands = fake_factory(plan, partial="spatial_shard00")
    with pytest.raises(RuntimeError, match="incomplete spatial_shard00"):
        run_plan(plan, popen=spawn, sleep=lambda _: None)
    assert len(commands) == len(plan["workers"])
    assert not (args.output_dir / "report").exists()


def test_interrupt_terminates_owned_children_and_records_status(tmp_path):
    args = arguments(tmp_path)
    plan = build_plan(args)
    spawn, processes, _ = fake_factory(plan, hold=True)
    def interrupted(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_plan(plan, popen=spawn, sleep=interrupted)
    assert all(process.terminated for process in processes)
    status = json.loads((args.output_dir / "status.json").read_text())
    assert status["state"] == "interrupted"


def test_resume_requires_same_plan_and_refuses_existing_report(tmp_path):
    args = arguments(tmp_path)
    plan = build_plan(args)
    args.output_dir.mkdir()
    (args.output_dir / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="exists"):
        build_plan(args)
    args.resume = True
    resumed = build_plan(args)
    assert all("--resume" in worker["command"] for worker in resumed["workers"])
    args.threads = 3
    with pytest.raises(ValueError, match="differs"):
        build_plan(args)
    args.threads = 4
    (args.output_dir / "report").mkdir()
    with pytest.raises(ValueError, match="report already exists"):
        build_plan(args)
