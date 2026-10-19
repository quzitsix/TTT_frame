#!/usr/bin/env python3
"""Merge experiment-C visual JSONL arms and write paired MCQ metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_experiment_c import condition_stream  # noqa: E402
from scripts.evaluate_experiment_c_visual import (  # noqa: E402
    _answer_pools,
    _choices,
    _parse_mcq_letters,
    _query_prompt,
)


CONDITION_ORDER = ("relocated", "same_scene_control", "filler_only_control")
CONDITIONS = set(CONDITION_ORDER)
QUERY_TYPES = {"location", "mover", "owner", "state", "next_action"}
ARM_MEMORY_STATES = {
    "icl": "visual_context",
    "blind": "no_visual_input",
    "spatial": "episode_fast_state",
    "spatial_slow_only": "learned_initial_state",
    "spatial_base": "spatial_branch_bypassed",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_tree(path: Path) -> dict[str, Any]:
    """Create a byte-level manifest for a model directory or checkpoint file."""

    if path.is_file():
        files = [path]
        root = path.parent
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        root = path
    else:
        raise FileNotFoundError(f"model artifact does not exist: {path}")
    tree = hashlib.sha256()
    entries = []
    for item in files:
        relative = item.name if path.is_file() else str(item.relative_to(root))
        file_hash = _sha256_file(item)
        size = item.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": file_hash})
        tree.update(relative.encode())
        tree.update(b"\0")
        tree.update(bytes.fromhex(file_hash))
    return {
        "path": str(path.resolve()),
        "files": entries,
        "total_bytes": sum(item["bytes"] for item in entries),
        "tree_sha256": tree.hexdigest(),
    }


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    return {
        "n": len(rows),
        "ok": len(valid),
        "errors": len(rows) - len(valid),
        "parse_rate": round(
            sum(row.get("status") == "ok" and bool(row.get("parse_valid")) for row in rows)
            / len(rows), 4
        ) if rows else None,
        "choice_accuracy": round(
            sum(row.get("status") == "ok" and bool(row.get("score", {}).get("choice_correct"))
                for row in rows) / len(rows), 4
        ) if rows else None,
    }


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["episode_id"]), str(row["condition"]), str(row["query_id"])


def _display_gold(row: dict[str, Any]) -> str:
    value = str(row["gold"]).strip()
    if row["query_type"] in {"location", "next_action"}:
        value = re.sub(r"\b[AB]/", "", value, flags=re.IGNORECASE)
    return value


def _validate_row(row: dict[str, Any]) -> None:
    required = {
        "episode_id", "condition", "arm", "query_id", "query_type", "gold",
        "question", "options", "gold_choice", "predicted_choice", "raw_response",
        "parse_valid", "status",
        "score", "observed_frames", "forecast_card_hidden", "answer_mode",
        "mcq_layout", "card_protocol", "episode_memory_state", "unique_observed_frames",
        "condition_visual_encode_passes", "condition_visual_frame_presentations",
        "frames_provided_to_answer",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"result row is missing required fields: {sorted(missing)}")
    if row["card_protocol"] != "experiment_c_visual_v2":
        raise ValueError(f"unsupported card protocol: {row['card_protocol']}")
    if row["forecast_card_hidden"] is not True:
        raise ValueError("future-visible rows cannot enter the formal C-PNG summary")
    if row["answer_mode"] != "mcq" or row["mcq_layout"] not in {"separate", "batched"}:
        raise ValueError("formal C-PNG summary requires an explicit MCQ layout")
    if row["condition"] not in CONDITIONS or row["query_type"] not in QUERY_TYPES:
        raise ValueError("unknown condition or query type")
    arm = str(row["arm"])
    if arm not in ARM_MEMORY_STATES:
        raise ValueError(f"unsupported formal C-PNG arm: {arm}")
    if row["episode_memory_state"] != ARM_MEMORY_STATES[arm]:
        raise ValueError(f"arm {arm} has the wrong episode_memory_state")
    options = row["options"]
    if not isinstance(options, list) or len(options) != 4 or len(set(map(str, options))) != 4:
        raise ValueError("every MCQ row must contain four unique options")
    if row["gold_choice"] not in "ABCD":
        raise ValueError("gold_choice must be A, B, C, or D")
    if str(options["ABCD".index(row["gold_choice"])]).strip() != _display_gold(row):
        raise ValueError("gold_choice does not point to the row's canonical gold answer")
    predicted = row["predicted_choice"]
    if predicted is not None and predicted not in "ABCD":
        raise ValueError("predicted_choice must be null or A, B, C, or D")
    if not isinstance(row["parse_valid"], bool):
        raise ValueError("parse_valid must be boolean")
    if row["parse_valid"] != (predicted in "ABCD" if predicted is not None else False):
        raise ValueError("parse_valid is inconsistent with predicted_choice")
    if row["status"] not in {"ok", "error"}:
        raise ValueError(f"unknown row status: {row['status']}")
    if row["status"] == "error" and predicted is not None:
        raise ValueError("error rows cannot contain a predicted_choice")
    expected_correct = predicted == row["gold_choice"]
    if bool(row["score"].get("choice_correct")) != expected_correct:
        raise ValueError("choice_correct is inconsistent with predicted_choice and gold_choice")
    if not isinstance(row["observed_frames"], int) or row["observed_frames"] <= 0:
        raise ValueError("observed_frames must be a positive integer")
    if row["unique_observed_frames"] != row["observed_frames"]:
        raise ValueError("unique_observed_frames must match observed_frames")
    expected_answer_frames = row["observed_frames"] if arm == "icl" else 0
    if row["frames_provided_to_answer"] != expected_answer_frames:
        raise ValueError(f"arm {arm} has an inconsistent answer-frame budget")
    expected_passes = 0 if arm == "blind" else (
        len(QUERY_TYPES) if arm == "icl" and row["mcq_layout"] == "separate" else 1
    )
    if row["condition_visual_encode_passes"] != expected_passes:
        raise ValueError(f"arm {arm} has an inconsistent visual encode-pass budget")
    if row["condition_visual_frame_presentations"] != expected_passes * row["observed_frames"]:
        raise ValueError(f"arm {arm} has an inconsistent visual frame-presentation budget")


def _validate_against_manifest(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """Rebuild formal questions and choices, then reparse every raw response."""

    if manifest.get("schema") != "experiment_c_relocation_v2":
        raise ValueError("formal visual rows require fixture schema v2")
    episodes = {str(item["episode_id"]): item for item in manifest.get("episodes", [])}
    pools = _answer_pools(manifest, list(CONDITION_ORDER))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id not in episodes:
            raise ValueError(f"result references unknown manifest episode: {episode_id}")
        episode = episodes[episode_id]
        queries = {str(item["id"]): item for item in episode["queries"]}
        query_id = str(row["query_id"])
        if query_id not in queries:
            raise ValueError(f"result references unknown manifest query: {episode_id}/{query_id}")
        query = queries[query_id]
        if str(row["query_type"]) != str(query["type"]):
            raise ValueError(
                f"manifest query type mismatch for {episode_id}/{query_id}: "
                f"row={row['query_type']!r}, manifest={query['type']!r}"
            )
        condition = str(row["condition"])
        events, answers = condition_stream(episode, condition)
        events = [event for event in events if event.get("kind") != "forecast"]
        protocol = row.get("run_protocol")
        max_frames = int(protocol.get("max_frames", 0)) if isinstance(protocol, dict) else 0
        expected_frames = min(len(events), max_frames) if max_frames else len(events)
        if row["observed_frames"] != expected_frames:
            raise ValueError(f"manifest frame budget mismatch for {episode_id}/{condition}")
        expected_gold = str(answers[query_id])
        expected_question = _query_prompt(
            query, forecast_hidden=True, condition=condition, answer_mode="mcq"
        )
        expected_options, expected_choice = _choices(
            query, expected_gold, episode_id=episode_id, condition=condition, pools=pools
        )
        if str(row["gold"]) != expected_gold:
            raise ValueError(f"manifest gold mismatch for {episode_id}/{condition}/{query_id}")
        if str(row["question"]) != expected_question:
            raise ValueError(f"manifest question mismatch for {episode_id}/{condition}/{query_id}")
        if list(row["options"]) != expected_options or row["gold_choice"] != expected_choice:
            raise ValueError(f"manifest MCQ mismatch for {episode_id}/{condition}/{query_id}")
        grouped[(str(row["arm"]), episode_id, condition)].append(row)

    for (arm, episode_id, condition), group in grouped.items():
        layouts = {str(row["mcq_layout"]) for row in group}
        if len(layouts) != 1:
            raise ValueError(f"mixed MCQ layouts in {arm}/{episode_id}/{condition}")
        layout = next(iter(layouts))
        if layout == "separate":
            for row in group:
                parsed = _parse_mcq_letters(str(row["raw_response"]), 1)
                expected = parsed[0] if len(parsed) == 1 else None
                if row["predicted_choice"] != expected or row["parse_valid"] != bool(parsed):
                    raise ValueError(
                        f"raw response parse mismatch for {arm}/{episode_id}/{condition}/"
                        f"{row['query_id']}"
                    )
            continue
        raw_responses = {str(row["raw_response"]) for row in group}
        if len(raw_responses) != 1:
            raise ValueError(f"batched rows have different raw responses in {arm}/{episode_id}/{condition}")
        query_order = [str(item["id"]) for item in episodes[episode_id]["queries"]]
        by_query = {str(row["query_id"]): row for row in group}
        if set(by_query) != set(query_order):
            raise ValueError(
                f"batched rows do not cover the manifest queries in {arm}/{episode_id}/{condition}"
            )
        parsed = _parse_mcq_letters(next(iter(raw_responses)), len(query_order))
        for index, query_id in enumerate(query_order):
            expected = parsed[index] if len(parsed) == len(query_order) else None
            row = by_query[query_id]
            if row["predicted_choice"] != expected or row["parse_valid"] != bool(parsed):
                raise ValueError(
                    f"raw batched response parse mismatch for {arm}/{episode_id}/{condition}/{query_id}"
                )


def _paired_protocol(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["query_type"], row["question"], row["gold"], tuple(row["options"]),
        row["gold_choice"], row["observed_frames"], row["forecast_card_hidden"],
        row["answer_mode"], row["mcq_layout"], row["card_protocol"],
    )


def _paired_delta(
    rows: list[dict[str, Any]], left: str, right: str,
    independence_groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    by_arm = defaultdict(dict)
    for row in rows:
        if row.get("arm") in {left, right}:
            layout = str(row["mcq_layout"])
            correct = row.get("status") == "ok" and bool(row["score"]["choice_correct"])
            by_arm[str(row["arm"])][_key(row)] = (correct, layout)
    keys = sorted(key for key in set(by_arm[left]) & set(by_arm[right])
                  if by_arm[left][key][1] == by_arm[right][key][1])
    outcomes = [(by_arm[left][key][0], by_arm[right][key][0]) for key in keys]
    left_only = sum(left_ok and not right_ok for left_ok, right_ok in outcomes)
    right_only = sum(right_ok and not left_ok for left_ok, right_ok in outcomes)
    discordant = left_only + right_only
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
        mcnemar_p = min(1.0, 2.0 * tail / (2 ** discordant))
    else:
        mcnemar_p = 1.0 if keys else None
    delta = ((left_only - right_only) / len(keys) if keys else None)
    cluster_deltas: dict[str, int] = defaultdict(int)
    for key, (left_ok, right_ok) in zip(keys, outcomes):
        cluster = independence_groups.get(key[0], key[0]) if independence_groups else key[0]
        cluster_deltas[cluster] += int(left_ok) - int(right_ok)
    episode_values = [cluster_deltas[key] for key in sorted(cluster_deltas)]
    episode_p = None
    if episode_values and len(episode_values) <= 20:
        observed = abs(sum(episode_values))
        total = 1 << len(episode_values)
        extreme = 0
        for assignment in range(total):
            permuted = sum(
                value if assignment & (1 << index) else -value
                for index, value in enumerate(episode_values)
            )
            extreme += abs(permuted) >= observed
        episode_p = extreme / total
    return {"left": left, "right": right, "matched": len(keys),
            "left_only_correct": left_only, "right_only_correct": right_only,
            "both_correct": sum(left_ok and right_ok for left_ok, right_ok in outcomes),
            "both_wrong": sum(not left_ok and not right_ok for left_ok, right_ok in outcomes),
            "exact_mcnemar_p": round(mcnemar_p, 6) if mcnemar_p is not None else None,
            "independence_clusters": len(episode_values),
            "independence_cluster_unit": "content_group" if independence_groups else "episode",
            "cluster_signflip_p": round(episode_p, 8) if episode_p is not None else None,
            "accuracy_delta": round(delta, 4) if delta is not None else None}


def _content_independence_groups(manifest: dict[str, Any]) -> dict[str, str]:
    """Merge episodes whose observed streams match under any formal condition."""

    episodes = list(manifest.get("episodes", []))
    identifiers = [str(episode["episode_id"]) for episode in episodes]
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for condition in CONDITIONS:
        seen: dict[str, str] = {}
        for episode in episodes:
            events, _ = condition_stream(episode, condition)
            events = [event for event in events if event.get("kind") != "forecast"]
            digest = hashlib.sha256(
                json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            identifier = str(episode["episode_id"])
            if digest in seen:
                union(identifier, seen[digest])
            else:
                seen[digest] = identifier
    return {identifier: find(identifier) for identifier in identifiers}


def summarize(rows: list[dict[str, Any]], inputs: list[Path], *,
              allow_legacy_metadata: bool = False, expected_episodes: int = 12,
              expected_conditions: set[str] | None = None,
              expected_observed_frames: int = 16,
              independence_groups: dict[str, str] | None = None,
              expected_arms: set[str] | None = None,
              manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if not rows:
        raise ValueError("no result rows supplied")
    seen = set()
    for row in rows:
        _validate_row(row)
        identity = (str(row.get("arm")), *_key(row))
        if identity in seen:
            raise ValueError(f"duplicate result row: {identity}")
        seen.add(identity)
    if manifest is not None:
        _validate_against_manifest(rows, manifest)
    rows_by_arm: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        rows_by_arm[str(row["arm"])][_key(row)] = row
    required_arms = set(ARM_MEMORY_STATES) if expected_arms is None else set(expected_arms)
    if set(rows_by_arm) != required_arms:
        raise ValueError(
            f"expected arms {sorted(required_arms)}, found {sorted(rows_by_arm)}"
        )
    reference_arm = sorted(rows_by_arm)[0]
    reference_keys = set(rows_by_arm[reference_arm])
    for arm, mapping in sorted(rows_by_arm.items()):
        keys = set(mapping)
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)[:5]
            extra = sorted(keys - reference_keys)[:5]
            raise ValueError(f"arm {arm} has an incomplete/mismatched grid; missing={missing}, extra={extra}")
        for key in reference_keys:
            if _paired_protocol(mapping[key]) != _paired_protocol(rows_by_arm[reference_arm][key]):
                raise ValueError(f"arm {arm} has mismatched question/gold/options/budget for {key}")

    episodes = sorted({key[0] for key in reference_keys})
    conditions = sorted({key[1] for key in reference_keys})
    required_conditions = CONDITIONS if expected_conditions is None else set(expected_conditions)
    if expected_episodes and len(episodes) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes, found {len(episodes)}: {episodes}"
        )
    if set(conditions) != required_conditions:
        raise ValueError(
            f"expected conditions {sorted(required_conditions)}, found {conditions}"
        )
    observed_budgets = {int(row["observed_frames"]) for row in rows}
    if expected_observed_frames and observed_budgets != {expected_observed_frames}:
        raise ValueError(
            f"expected {expected_observed_frames} observed frames, found {sorted(observed_budgets)}"
        )
    if independence_groups is not None:
        missing_groups = sorted(set(episodes) - set(independence_groups))
        if missing_groups:
            raise ValueError(f"missing independence groups for episodes: {missing_groups}")
    for episode in episodes:
        query_ids = {key[2] for key in reference_keys if key[0] == episode}
        for condition in conditions:
            present = {key[2] for key in reference_keys
                       if key[0] == episode and key[1] == condition}
            if present != query_ids:
                raise ValueError(f"incomplete query grid for {episode}/{condition}")
            query_types = {
                rows_by_arm[reference_arm][(episode, condition, query_id)]["query_type"]
                for query_id in present
            }
            if query_types != QUERY_TYPES:
                raise ValueError(
                    f"{episode}/{condition} does not contain every required query type: "
                    f"{sorted(query_types)}"
                )

    pair_specs = (
        ("spatial", "spatial_slow_only"),
        ("spatial", "spatial_base"),
        ("icl", "blind"),
    )
    metadata_complete = all(isinstance(row.get("run_protocol"), dict) for row in rows)
    if not metadata_complete and not allow_legacy_metadata:
        raise ValueError(
            "rows lack run_protocol fingerprints; rerun with the current runner or pass "
            "--allow-legacy-metadata for explicitly marked legacy-metadata results"
        )
    protocol_by_arm: dict[str, dict[str, Any]] = {}
    for arm, mapping in sorted(rows_by_arm.items()):
        protocols = [row.get("run_protocol") for row in mapping.values()]
        available = [protocol for protocol in protocols if isinstance(protocol, dict)]
        if available:
            serialized = {json.dumps(protocol, sort_keys=True) for protocol in available}
            if len(serialized) != 1 or len(available) != len(protocols):
                raise ValueError(f"arm {arm} mixes run protocols or metadata presence")
            protocol = available[0]
            if protocol.get("manifest_schema") != "experiment_c_relocation_v2":
                raise ValueError(f"arm {arm} is not from fixture schema v2")
            if set(protocol.get("conditions", [])) != set(conditions):
                raise ValueError(f"arm {arm} run_protocol conditions do not match its rows")
            fingerprint_payload = {
                key: value for key, value in protocol.items()
                if key not in {
                    "comparison_fingerprint", "runner_sha256", "spatial_checkpoint",
                    "metadata_origin",
                }
            }
            computed_fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True).encode()
            ).hexdigest()
            if protocol.get("comparison_fingerprint") != computed_fingerprint:
                raise ValueError(f"arm {arm} has an invalid comparison_fingerprint")
            protocol_by_arm[arm] = protocol
    for left, right in pair_specs:
        if left not in rows_by_arm or right not in rows_by_arm:
            continue
        left_protocol = protocol_by_arm.get(left)
        right_protocol = protocol_by_arm.get(right)
        if isinstance(left_protocol, dict) and isinstance(right_protocol, dict):
            if left_protocol.get("comparison_fingerprint") != right_protocol.get("comparison_fingerprint"):
                raise ValueError(f"paired arms {left}/{right} have different run protocols")
            if left_protocol.get("runner_sha256") != right_protocol.get("runner_sha256"):
                raise ValueError(f"paired arms {left}/{right} came from different runner versions")
    groups = {
        "by_arm": defaultdict(list),
        "by_arm_condition": defaultdict(list),
        "by_arm_query_type": defaultdict(list),
    }
    for row in rows:
        arm = str(row["arm"])
        groups["by_arm"][arm].append(row)
        groups["by_arm_condition"][f"{arm}::{row['condition']}"].append(row)
        groups["by_arm_query_type"][f"{arm}::{row['query_type']}"].append(row)
    result: dict[str, Any] = {
        "schema": "experiment_c_visual_suite_v2",
        "summarizer_sha256": _sha256_file(Path(__file__)),
        "inputs": [{"path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in inputs],
        "rows": len(rows),
        "episodes": episodes,
        "conditions": conditions,
        "primary_metric": "choice_accuracy",
        "chance_accuracy": 0.25,
        "future_event_observed": any(not row["forecast_card_hidden"] for row in rows),
        "protocol_validation": "passed",
        "manifest_semantics_revalidated": manifest is not None,
        "raw_responses_reparsed": manifest is not None,
        "run_protocol_metadata_complete": metadata_complete,
        "legacy_metadata_override": bool(allow_legacy_metadata and not metadata_complete),
        "error_policy": "intention-to-evaluate: error and parse-failure rows count as incorrect",
        "independence_grouping": (
            "merge episodes with an identical observed event stream in any formal condition"
            if independence_groups else "episode"
        ),
        "independence_groups": len({independence_groups.get(item, item) for item in episodes})
        if independence_groups else len(episodes),
    }
    for name, mapping in groups.items():
        result[name] = {key: _metric(value) for key, value in sorted(mapping.items())}
    result["layouts_by_arm"] = {
        arm: sorted({str(row["mcq_layout"]) for row in arm_rows})
        for arm, arm_rows in sorted(groups["by_arm"].items())
    }
    result["choice_counts_by_arm"] = {
        arm: {
            "gold": dict(sorted(Counter(str(row.get("gold_choice")) for row in arm_rows).items())),
            "predicted": dict(sorted(Counter(str(row.get("predicted_choice")) for row in arm_rows).items())),
        }
        for arm, arm_rows in sorted(groups["by_arm"].items())
    }
    result["paired_arm_deltas"] = {}
    for left, right in pair_specs:
        metric = _paired_delta(rows, left, right, independence_groups)
        if metric["matched"]:
            metric["run_protocol_metadata_complete"] = bool(
                left in protocol_by_arm and right in protocol_by_arm
            )
            metric["inference_note"] = (
                "content-group sign-flip is primary; row-level McNemar is exploratory"
                if independence_groups else
                "episode sign-flip is primary; row-level McNemar is exploratory"
            )
            result["paired_arm_deltas"][f"{left}_minus_{right}"] = metric
    result["relocated_minus_same_scene"] = {}
    for arm in sorted(groups["by_arm"]):
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_condition = defaultdict(dict)
        for row in arm_rows:
            condition_key = (str(row["episode_id"]), str(row["query_id"]))
            by_condition[str(row["condition"])][condition_key] = (
                row.get("status") == "ok" and bool(row["score"]["choice_correct"])
            )
        left = by_condition["relocated"]
        right = by_condition["same_scene_control"]
        keys = sorted(set(left) & set(right))
        if keys:
            result["relocated_minus_same_scene"][arm] = {
                "matched": len(keys),
                "accuracy_delta": round(sum(float(left[key]) - float(right[key]) for key in keys)
                                        / len(keys), 4),
            }
    return result


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Experiment C visual MCQ summary", "",
        "Every row passed the C-PNG protocol checks: future targets are withheld and every "
        "question has four unique choices. Error and parse-failure rows count as incorrect.", "",
        f"Run-protocol fingerprints complete: **{summary['run_protocol_metadata_complete']}**. "
        f"Legacy metadata override: **{summary['legacy_metadata_override']}**.", "",
        f"Manifest question/gold/choice reconstruction: **{summary['manifest_semantics_revalidated']}**. "
        f"Raw-response reparsing: **{summary['raw_responses_reparsed']}**.", "",
        f"Byte-level model artifact manifests: **{bool(summary.get('model_artifacts'))}**.", "",
        f"Inference clusters: **{summary['independence_groups']}** content groups. Episodes "
        "whose observed streams match under any formal condition share one cluster. This removes "
        "exact duplicates; it does not assert that generator-derived clusters are IID.", "",
        "| arm | n | errors | parse rate | choice accuracy |", "|---|---:|---:|---:|---:|",
    ]
    for arm, metric in summary["by_arm"].items():
        lines.append(f"| {arm} | {metric['n']} | {metric['errors']} | {metric['parse_rate']} | "
                     f"{metric['choice_accuracy']} |")
    lines += ["", "## By condition", "",
              "| arm | condition | n | parse rate | choice accuracy |", "|---|---|---:|---:|---:|"]
    for key, metric in summary["by_arm_condition"].items():
        arm, condition = key.split("::")
        lines.append(f"| {arm} | {condition} | {metric['n']} | {metric['parse_rate']} | "
                     f"{metric['choice_accuracy']} |")
    lines += ["", "## By query type", "",
              "| arm | query type | n | parse rate | choice accuracy |",
              "|---|---|---:|---:|---:|"]
    for key, metric in summary["by_arm_query_type"].items():
        arm, query_type = key.split("::")
        lines.append(f"| {arm} | {query_type} | {metric['n']} | {metric['parse_rate']} | "
                     f"{metric['choice_accuracy']} |")
    lines += ["", "## Choice distributions", "",
              "Gold counts are reported rather than assumed. A large prediction skew is a readout bias.", "",
              "| arm | layouts | gold A/B/C/D | predicted A/B/C/D |",
              "|---|---|---|---|"]
    for arm, counts in summary["choice_counts_by_arm"].items():
        gold = "/".join(str(counts["gold"].get(letter, 0)) for letter in "ABCD")
        predicted = "/".join(str(counts["predicted"].get(letter, 0)) for letter in "ABCD")
        layouts = ", ".join(summary["layouts_by_arm"][arm])
        lines.append(f"| {arm} | {layouts} | {gold} | {predicted} |")
    lines += ["", "## Paired deltas", ""]
    for name, metric in summary["paired_arm_deltas"].items():
        cluster_p = metric["cluster_signflip_p"]
        cluster_text = f"{cluster_p:.8f}" if cluster_p is not None else "not enumerated"
        lines.append(
            f"- {name}: {metric['accuracy_delta']:+.4f} over {metric['matched']} matched rows; "
            f"left-only/right-only correct={metric['left_only_correct']}/{metric['right_only_correct']}; "
            f"{metric['independence_cluster_unit']}-cluster sign-flip p={cluster_text}; "
            f"row-level McNemar p={metric['exact_mcnemar_p']:.6f} (exploratory)"
        )
    for arm, metric in summary["relocated_minus_same_scene"].items():
        lines.append(f"- {arm} relocated minus same-scene: {metric['accuracy_delta']:+.4f} "
                     f"over {metric['matched']} matched rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_c_visual_v2.summary.json"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--allow-legacy-metadata", action="store_true",
                        help="accept older rows without run_protocol; the report marks this limitation")
    parser.add_argument("--expected-episodes", type=int, default=12,
                        help="required number of complete episodes; 0 disables the count check")
    parser.add_argument("--expected-conditions", nargs="+", choices=sorted(CONDITIONS),
                        default=sorted(CONDITIONS))
    parser.add_argument("--expected-observed-frames", type=int, default=16,
                        help="required per-row observation budget; 0 disables the check")
    parser.add_argument("--expected-arms", nargs="+", choices=sorted(ARM_MEMORY_STATES),
                        default=sorted(ARM_MEMORY_STATES))
    parser.add_argument("--data", type=Path, default=Path("data/experiment_c/manifest.json"),
                        help="manifest used to verify fingerprints and group duplicate content")
    parser.add_argument("--hash-model-artifacts", action="store_true",
                        help="hash every base-model/processor file and verify Spatial checkpoint bytes")
    args = parser.parse_args()
    if args.expected_episodes < 0 or args.expected_observed_frames < 0:
        raise SystemExit("expected episode and frame counts must be nonnegative")
    rows = []
    for path in args.inputs:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    manifest = json.loads(args.data.read_text(encoding="utf-8"))
    if manifest.get("schema") != "experiment_c_relocation_v2":
        raise ValueError("formal visual summary requires fixture schema v2")
    manifest_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    row_manifest_hashes = {
        row.get("run_protocol", {}).get("manifest_sha256")
        for row in rows if isinstance(row.get("run_protocol"), dict)
    }
    if row_manifest_hashes and row_manifest_hashes != {manifest_sha}:
        raise ValueError("result rows do not match --data manifest")
    independence_groups = _content_independence_groups(manifest)
    summary = summarize(
        rows,
        args.inputs,
        allow_legacy_metadata=args.allow_legacy_metadata,
        expected_episodes=args.expected_episodes,
        expected_conditions=set(args.expected_conditions),
        expected_observed_frames=args.expected_observed_frames,
        independence_groups=independence_groups,
        expected_arms=set(args.expected_arms),
        manifest=manifest,
    )
    summary["manifest"] = {"path": str(args.data), "sha256": manifest_sha}
    if args.hash_model_artifacts:
        protocols = [row["run_protocol"] for row in rows
                     if isinstance(row.get("run_protocol"), dict)]
        model_paths = sorted({str(item["model_path"]) for item in protocols if item.get("model_path")})
        checkpoint_paths = sorted({str(item["spatial_checkpoint"]) for item in protocols
                                   if item.get("spatial_checkpoint")})
        artifacts = {
            str(Path(path).resolve()): _artifact_tree(Path(path))
            for path in [*model_paths, *checkpoint_paths]
        }
        for protocol in protocols:
            checkpoint = protocol.get("spatial_checkpoint")
            expected = protocol.get("spatial_checkpoint_hash")
            if checkpoint and expected:
                entry = artifacts[str(Path(checkpoint).resolve())]
                actual = entry["files"][0]["sha256"] if len(entry["files"]) == 1 else entry["tree_sha256"]
                if actual != expected:
                    raise ValueError("Spatial checkpoint bytes do not match run_protocol")
        summary["model_artifacts"] = artifacts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report:
        write_report(args.report, summary)
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "arms": sorted(summary["by_arm"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
