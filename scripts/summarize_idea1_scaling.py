#!/usr/bin/env python3
"""Archive already-scored scaling runs without opening benchmark answers.

The archive contains compact predictions and diagnostics, never checkpoints,
video contents, candidate scores, or per-case answer keys.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


ROLES = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def without_cases(value: object) -> object:
    """Drop case-level labels even if a future aggregate accidentally includes them."""
    if isinstance(value, dict):
        return {key: without_cases(item) for key, item in value.items()
                if key not in ("cases", "correct_answer", "candidate_scores", "raw_answer")}
    if isinstance(value, list):
        return [without_cases(item) for item in value]
    return value


def numeric_summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("nonempty finite diagnostic values required")
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def summarize_stats(records: list[dict]) -> dict:
    counts = [record["mature_count"] for record in records]
    if any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError("positive mature counts required for every episode")
    mse = [float(record["mature_mse"]) for record in records]
    result = {
        "cases": len(records), "mature_pairs": sum(counts),
        "mature_mse": numeric_summary(mse),
        "mature_mse_pair_weighted": sum(value * count for value, count in zip(mse, counts)) / sum(counts),
    }
    for key in ("updates", "gradient_steps", "state_bytes", "pending_bytes", "fast_weight_delta_l2"):
        if all(key in row for row in records):
            result[key] = numeric_summary([float(row[key]) for row in records])
    horizons = set(records[0].get("mature_mse_by_horizon", {}))
    if any(set(row.get("mature_mse_by_horizon", {})) != horizons for row in records):
        raise ValueError("inconsistent horizon diagnostics")
    result["mature_mse_by_horizon"] = {
        horizon: numeric_summary([float(row["mature_mse_by_horizon"][horizon]) for row in records])
        for horizon in sorted(horizons, key=int)
    }
    return result


def compact_predictions(path: Path, item: dict, protocol: dict, head: str) -> list[dict]:
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"frozen prediction checksum mismatch: {head}")
    role_by_id = {sample_id: role for role in ROLES for sample_id in protocol["splits"][role]}
    views = protocol["views"] + ["text_prior"]
    predictions = {sample_id: {} for sample_id in role_by_id}
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        sample_id, arm = row["id"], row["arm"]
        if (sample_id not in predictions or arm not in views or arm in predictions[sample_id]
                or row.get("prediction") not in "ABCDEF" or len(row["prediction"]) != 1
                or row.get("research_split") != role_by_id[sample_id]
                or row.get("readout") != head or row.get("status") != "ok"):
            raise ValueError(f"invalid or duplicate prediction: {head}/{sample_id}/{arm}")
        predictions[sample_id][arm] = row["prediction"]
        count += 1
    if (count != item["rows"] or count != len(predictions) * len(views)
            or any(set(values) != set(views) for values in predictions.values())):
        raise ValueError(f"incomplete frozen predictions: {head}")
    return [{"id": sample_id, "research_split": role_by_id[sample_id],
             "predictions": values} for sample_id, values in predictions.items()]


def summarize_trial(trial: dict, protocol: dict, split: dict, expected_files: dict) -> tuple[dict, list[Path]]:
    directory = Path(trial["forecast_dir"])
    installed = read_json(directory / "protocol.json")
    if (canonical_sha256(installed["protocol"]) != installed["protocol_sha256"]
            or installed != protocol or installed["protocol"]["configuration"] != trial["configuration"]):
        raise ValueError(f"trial protocol mismatch: {trial['name']}")
    files = sorted(directory.glob("summary*.json"))
    if not files:
        raise ValueError(f"missing completion summaries: {trial['name']}")
    summaries = [read_json(path) for path in files]
    finished = {}
    for summary in summaries:
        if summary.get("complete") is not True or summary["protocol_sha256"] != installed["protocol_sha256"]:
            raise ValueError("incomplete trial summary")
        for row in summary["cases"]:
            sample_id = row["sample_id"]
            if sample_id in finished:
                raise ValueError("overlapping trial completion summaries")
            finished[sample_id] = row["artifact_sha256"]
    all_ids = [sample_id for role in ROLES for sample_id in split[role]]
    if finished != expected_files or set(finished) != set(all_ids):
        raise ValueError(f"trial completion/fit-time artifact mismatch: {trial['name']}")
    stats = {arm: {role: [] for role in ROLES} for arm in trial["arms"]}
    runtime = 0.0
    metadata = None
    role_by_id = {sample_id: role for role in ROLES for sample_id in split[role]}
    for sample_id in all_ids:
        path = directory / f"{sample_id}_forecasts.pt"
        if sha256_file(path) != finished[sample_id]:
            raise ValueError(f"forecast artifact checksum mismatch: {trial['name']}/{sample_id}")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (artifact["sample_id"] != sample_id or artifact["trial"] != trial["name"]
                or artifact["protocol_sha256"] != installed["protocol_sha256"]
                or set(artifact["stats"]) != set(trial["arms"])):
            raise ValueError("forecast artifact identity mismatch")
        checked = {key: value for key, value in artifact.items()
                   if key not in ("last_forecasts", "metadata_sha256")}
        if artifact["metadata_sha256"] != canonical_sha256(checked):
            raise ValueError("forecast metadata checksum mismatch")
        if metadata is not None and metadata != artifact["result_metadata"]:
            raise ValueError("episode diagnostic policies differ within trial")
        metadata = artifact["result_metadata"]
        if artifact["batch_cases"] <= 0 or not math.isfinite(artifact["batch_seconds"]):
            raise ValueError("invalid runtime diagnostic")
        runtime += artifact["batch_seconds"] / artifact["batch_cases"]
        for arm in trial["arms"]:
            stats[arm][role_by_id[sample_id]].append(artifact["stats"][arm])
    return {
        "configuration": trial["configuration"],
        "episodes": len(all_ids),
        "parameter_values_per_target": 2 * len(metadata["horizons"]) * metadata["feature_shape"][1] * trial["configuration"]["hidden_dim"],
        "fp32_weight_bytes_per_target": 8 * len(metadata["horizons"]) * metadata["feature_shape"][1] * trial["configuration"]["hidden_dim"],
        "result_metadata": metadata,
        "forecast_batch_compute_seconds_summed": runtime,
        "forecast_seconds_per_case_amortized": runtime / len(all_ids),
        "shard_elapsed_seconds_summed": sum(row["elapsed_seconds"] for row in summaries),
        "shard_elapsed_seconds_max": max(row["elapsed_seconds"] for row in summaries),
        "runtime_scope": "all eight arms together; summed batch compute excludes cache loading, and is not parallel wall time",
        "by_arm": {arm: {role: summarize_stats(rows) for role, rows in by_role.items()}
                   for arm, by_role in stats.items()},
    }, files


def prediction_disagreements(rows: list[dict], trial: dict) -> dict:
    name, arms = trial["name"], trial["arms"]
    pairs = [(arm, "recent_only") for arm in arms if arm != "recent_only"]
    pairs += [(arm, "residual_no_update" if "residual" in arm else "plain_no_update")
              for arm in ("residual", "reset_residual", "recent_tail_residual", "plain_delayed")
              if arm in arms]
    result = {}
    for arm, control in pairs:
        if control not in arms:
            continue
        result[f"{arm}_vs_{control}"] = {}
        for role in ROLES:
            group = [row for row in rows if row["research_split"] == role]
            result[f"{arm}_vs_{control}"][role] = {
                "n": len(group), "different_answers": sum(
                    row["predictions"][f"{name}__{arm}"] != row["predictions"][f"{name}__{control}"]
                    for row in group),
            }
    return result


def format_accuracy(record: dict) -> str:
    return f"{record['correct']}/{record['n']} ({record['accuracy_all']:.2%})"


def results_markdown(summary: dict) -> str:
    primary = summary["primary_head"]
    lines = ["# Idea 01 容量与训练量比较归档", "", summary["test_status"], "",
             "所有配置相同数据划分、相同视频缓存。此处 test57 已在此前实验查看，属于探索性复用。",
             "离线预测预训练不等于对在线适应过程做 meta/outer training；增加配对/step 不增加独立标签。", "",
             f"主读出：`{primary}`。", "",
             "| Trial | 参数值 | Residual val | Residual test | Frozen residual test | Plain test |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, trial in summary["by_trial"].items():
        arms = summary["by_head"][primary]["by_trial"][name]["by_arm"]
        cells = [format_accuracy(arms["residual"]["validation"]),
                 format_accuracy(arms["residual"]["test"]),
                 format_accuracy(arms["residual_no_update"]["test"]),
                 format_accuracy(arms["plain_delayed"]["test"])]
        lines.append(f"| {name} | {trial['parameter_values_per_target']:,} | " + " | ".join(cells) + " |")
    lines += ["", "## 读出训练量诊断", "",
              "| Readout | Trial | Residual test | Frozen residual test | Plain test |",
              "|---|---|---:|---:|---:|"]
    for head, record in summary["by_head"].items():
        for name, trial in record["by_trial"].items():
            cells = [format_accuracy(trial["by_arm"][arm]["test"])
                     for arm in ("residual", "residual_no_update", "plain_delayed")]
            lines.append(f"| {head} | {name} | " + " | ".join(cells) + " |")
    lines += ["", "## 过去目标的因果视觉预测 MSE", "",
              "每题最后 30 秒内发出的预测，目标实际到达后才计分；不含截止之后的未来真值。",
              "这些 MSE 是视觉潜变量的标准化误差，不是最终预测动作的选择题准确率。", "",
              "| Trial | Residual test MSE | Frozen residual test MSE | Plain test MSE | Compute s/case |",
              "|---|---:|---:|---:|---:|"]
    for name, trial in summary["by_trial"].items():
        cells = [f"{trial['by_arm'][arm]['test']['mature_mse_pair_weighted']:.6f}"
                 for arm in ("residual", "residual_no_update", "plain_delayed")]
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {trial['forecast_seconds_per_case_amortized']:.3f} |")
    lines += ["", "Compute 包含该配置八个 arms 的批处理总量按题摊销，不是单一 arm 延迟；",
              "并行 wall time、缓存读入与离线预训练时间另见 compact_summary.json。", "",
              "## 归档范围", "",
              "`compact_summary.json` 包含各读出、配置、arm、split 的准确率、配对得失、答案翻转与视觉诊断。",
              "`*_predictions.jsonl` 每个读出每题一行，仅保存各 arm 的选择字母；不含答案键和候选分数。",
              "`manifest.json` 校验归档内容，并记录来源文件校验和。未归档模型权重或视频。", ""]
    return "\n".join(lines)


def derive_pretraining_audit(run_dir: Path, plan: dict) -> dict:
    """Build a provenance-only audit from immutable pretraining summaries.

    The normal run includes a richer independently generated audit.  A fresh
    run made with the documented commands may only have the producer's
    ``summary.json`` files, so this fallback deliberately copies diagnostics
    recorded by the producer and marks their provenance.  It never loads a
    checkpoint tensor and never reads benchmark answers.
    """
    by_width = {}
    pretraining_root = run_dir / "pretraining"
    for directory in sorted(pretraining_root.glob("h*")):
        if not directory.is_dir():
            continue
        summary_path, protocol_path = directory / "summary.json", directory / "protocol.json"
        if not summary_path.exists() or not protocol_path.exists():
            raise ValueError(f"incomplete pretraining summary/protocol: {directory.name}")
        summary, installed = read_json(summary_path), read_json(protocol_path)
        protocol = installed.get("protocol")
        if (summary.get("schema") != "idea01_offline_initialization_summary_v1"
                or summary.get("complete") is not True
                or installed.get("protocol_sha256") != canonical_sha256(protocol)
                or summary.get("protocol_sha256") != installed["protocol_sha256"]):
            raise ValueError(f"invalid pretraining summary/protocol: {directory.name}")
        snapshots = summary.get("snapshots")
        if not isinstance(snapshots, dict) or not snapshots:
            raise ValueError(f"pretraining snapshots missing: {directory.name}")
        first = next(iter(snapshots.values()))
        hidden = first.get("hidden_dim")
        if type(hidden) is not int or hidden < 1:
            raise ValueError(f"invalid hidden dimension: {directory.name}")
        expected_width = directory.name[1:]
        if str(hidden) != expected_width:
            raise ValueError(f"pretraining directory/width mismatch: {directory.name}")
        train_ids = first.get("train_ids")
        if not isinstance(train_ids, list) or len(set(train_ids)) != len(train_ids):
            raise ValueError(f"invalid pretraining train IDs: {directory.name}")
        if (protocol.get("train_ids") != train_ids
                or protocol.get("config", {}).get("hidden_dim") != hidden
                or any(protocol.get(flag) is not False for flag in (
                    "test_ids_or_labels_used", "mcq_answers_used", "future_media_used", "meta_outer_training"))):
            raise ValueError(f"pretraining data-use policy mismatch: {directory.name}")
        matching = [trial for trial in plan["trials"] if trial["configuration"]["hidden_dim"] == hidden]
        if not matching:
            raise ValueError(f"pretraining width missing from experiment plan: {hidden}")
        trial_installed = read_json(Path(matching[0]["forecast_dir"]) / "protocol.json")
        if canonical_sha256(trial_installed["protocol"]) != trial_installed["protocol_sha256"]:
            raise ValueError("invalid forecast protocol for pretraining dimensions")
        # Result metadata carries the latent feature width.  The producer
        # summaries carry only horizons and MLP width, so obtain D from one
        # already-scored per-case metadata record through the trial diagnostics
        # later in archive(), without inferring it from the checkpoint tensor.
        by_width[expected_width] = {
            "independent_training_cases": len(train_ids),
            "elapsed_seconds": summary["elapsed_seconds"],
            "training_seconds": summary["training_seconds"],
            "peak_cuda_allocated_bytes": summary.get("peak_cuda_allocated_bytes"),
            "snapshots": snapshots,
            "protocol_sha256": installed["protocol_sha256"],
            "source_summary_sha256": sha256_file(summary_path),
            "source_protocol_file_sha256": sha256_file(protocol_path),
            "derived_from_saved_summaries": True,
        }
        for step, snapshot in snapshots.items():
            if (snapshot.get("train_ids") != train_ids or snapshot.get("hidden_dim") != hidden
                    or snapshot.get("protocol_sha256") != installed["protocol_sha256"]
                    or str(snapshot.get("steps")) != step
                    or snapshot.get("test_ids_or_labels_used") is not False):
                raise ValueError(f"inconsistent pretraining snapshot: {directory.name}/{step}")
        gains = {snapshot.get("width_output_gain") for snapshot in snapshots.values()}
        if len(gains) == 1:
            by_width[expected_width]["width_output_gain"] = gains.pop()
    if not by_width:
        raise ValueError("no pretraining summaries found")
    return {
        "schema": "idea01_scaling_pretraining_derived_audit_v1",
        "derived_from_saved_summaries": True,
        "independent_audit_available": False,
        "private_answers_or_future_read": False,
        "is_meta_outer_training": False,
        "by_width": by_width,
    }


def archive(run_dir: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("archive output directory already exists")
    plan_path = run_dir / "plan.json"
    plan = read_json(plan_path)
    if plan.get("schema") != "idea01_scaling_experiment_plan_v1":
        raise ValueError("unsupported experiment plan")
    mcq = run_dir / "mcq"
    installed = read_json(mcq / "protocol.json")
    protocol = installed["protocol"]
    completion = read_json(mcq / "completion.json")
    aggregate_path = mcq / "report/aggregate_metrics.json"
    aggregate = read_json(aggregate_path)
    fingerprint = canonical_sha256(protocol)
    if (installed["protocol_fingerprint"] != fingerprint
            or completion.get("complete") is not True
            or completion.get("test_answers_loaded") is not False
            or completion["protocol_fingerprint"] != fingerprint
            or aggregate["protocol_fingerprint"] != fingerprint
            or protocol["plan_sha256"] != sha256_file(plan_path)
            or protocol["plan"] != plan
            or set(protocol["heads"]) != set(completion["files"])
            or set(protocol["heads"]) != set(aggregate["by_head"])):
        raise ValueError("scored protocol, plan and completion mismatch")
    split = protocol["splits"]
    all_ids = [sample_id for role in ROLES for sample_id in split[role]]
    if len(all_ids) != len(set(all_ids)) or any(not split[role] for role in ROLES):
        raise ValueError("nonempty disjoint research splits required")
    copies = {"plan.json": plan_path, "readout/protocol.json": mcq / "protocol.json",
              "readout/completion.json": mcq / "completion.json",
              "readout/validation_selection.json": mcq / "validation_selection.json"}
    if (run_dir / "execution.json").exists():
        execution = read_json(run_dir / "execution.json")
        if execution.get("complete") is not True or execution.get("errors"):
            raise ValueError("incomplete or failed scaling execution")
        copies["execution.json"] = run_dir / "execution.json"
    else:
        execution = None
    integrity_path = run_dir / "integrity_audit.json"
    if integrity_path.exists():
        copies["integrity_audit.json"] = integrity_path
    audit_path = run_dir / "pretraining_audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        copies["pretraining_audit.json"] = audit_path
    else:
        audit = derive_pretraining_audit(run_dir, plan)
        # Written below as generated JSON; keeping it in the archive makes the
        # fallback explicit and reproducible for a run lacking the audit file.
    diagnostics = {}
    for trial in plan["trials"]:
        name = trial["name"]
        diagnostics[name], summaries = summarize_trial(
            trial, protocol["trial_protocols"][name], split, protocol["trial_forecast_sha256"][name])
        copies[f"trials/{name}/protocol.json"] = Path(trial["forecast_dir"]) / "protocol.json"
        for path in summaries:
            copies[f"trials/{name}/{path.name}"] = path
    for directory in sorted((run_dir / "pretraining").glob("h*")):
        for filename in ("protocol.json", "summary.json"):
            copies[f"pretraining/{directory.name}/{filename}"] = directory / filename
    if audit.get("derived_from_saved_summaries"):
        for width, record in audit["by_width"].items():
            matches = [row for row in diagnostics.values() if row["configuration"]["hidden_dim"] == int(width)]
            if not matches:
                raise ValueError(f"missing forecast diagnostics for pretrained width {width}")
            record["parameter_values_per_target"] = matches[0]["parameter_values_per_target"]
            record["persistent_state_bytes_per_target"] = matches[0]["fp32_weight_bytes_per_target"]
    by_head, compact = {}, {}
    source_predictions = {}
    for head in protocol["heads"]:
        item = completion["files"][head]
        if Path(item["file"]).name != item["file"]:
            raise ValueError("prediction path must be a basename")
        path = mcq / item["file"]
        rows = compact_predictions(path, item, protocol, head)
        if len(rows) != len(all_ids):
            raise ValueError("compact prediction case count mismatch")
        compact[head] = rows
        source_predictions[head] = {"file": str(path), "sha256": item["sha256"], "rows": item["rows"]}
        report = aggregate["by_head"][head]
        by_trial = {}
        for trial in plan["trials"]:
            name = trial["name"]
            by_trial[name] = {
                "by_arm": {arm: report["by_arm"][f"{name}__{arm}"]["by_research_split"]
                           for arm in trial["arms"]},
                "paired_within_trial": report["paired_within_trial"][name],
                "answer_disagreements": prediction_disagreements(rows, trial),
            }
        by_head[head] = {
            "fit_train_cases": len(protocol["head_train_ids"].get(head, split["train"])),
            "selected_lambda": protocol["selected_lambdas"][head],
            "by_trial": by_trial,
            "text_prior": report["by_arm"]["text_prior"]["by_research_split"],
        }
    summary = {
        "schema": "idea01_scaling_compact_archive_v1", "primary_head": protocol["primary_head"],
        "protocol_fingerprint": fingerprint, "test_status": aggregate["test_status"],
        "split_cases": {role: len(split[role]) for role in ROLES},
        "unique_labeled_training_cases_increase": plan["unique_labeled_training_cases_increase"],
        "pretraining": audit, "by_trial": diagnostics, "by_head": by_head,
        "prediction_stage_elapsed_seconds": completion["elapsed_seconds"],
        "forecast_parallel_wall_seconds": (execution["finished_unix"] - execution["started_unix"])
            if execution else None,
        "answer_keys_read_by_archiver": False,
        "uncertainty": "exploratory reuse; only two test recording days, overlapping prefixes, one training seed",
    }
    # Validate all inputs before creating a reviewable immutable archive directory.
    output.mkdir(parents=True)
    for relative, path in copies.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(output / "compact_summary.json", summary)
    if not audit_path.exists():
        write_json(output / "pretraining_audit.json", audit)
    write_json(output / "aggregate_metrics.json", without_cases(aggregate))
    for head, rows in compact.items():
        (output / f"{head}_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8")
    (output / "results.md").write_text(results_markdown(summary), encoding="utf-8")
    files = {str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", {
        "schema": "idea01_scaling_archive_manifest_v1", "source_run": str(run_dir.resolve()),
        "source_aggregate_sha256": sha256_file(aggregate_path),
        "source_predictions": source_predictions,
        "files": files, "manifest_includes_itself": False,
    })
    print(f"Archived {len(plan['trials'])} trials, {len(by_head)} heads and {len(all_ids)} cases in {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/forecast_bench_v0/idea01_scaling_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bench_v0_idea01_scaling_v1"))
    args = parser.parse_args(argv)
    archive(args.run_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
