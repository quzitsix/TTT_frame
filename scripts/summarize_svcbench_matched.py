#!/usr/bin/env python3
"""Strictly merge and audit matched real-video SVCBench result shards.

The matched runner writes five rows for every query point.  This summarizer
reconstructs the expected grid from the SVCBench subset, verifies that every
arm used the same observable video prefix, reparses every raw answer, and only
then computes metrics.  Parse and runtime failures stay in the denominator as
wrong answers.  Official GPA/MoC/UDA are reported only for complete
trajectories; invalid predictions are never removed from a sequence.

Example::

    conda run --no-capture-output -n meowbench python \
      scripts/summarize_svcbench_matched.py \
      runs/svcbench_real_matched_v1/shard0.jsonl \
      runs/svcbench_real_matched_v1/shard1.jsonl \
      runs/svcbench_real_matched_v1/shard2.jsonl \
      --annotations /data/quzitsix/svcbench_real_v1/subset/svcbench_subset.json \
      --selection data/svcbench_real_ego4d_pilot_v1.json \
      --output runs/svcbench_real_matched_v1/summary.json \
      --report runs/svcbench_real_matched_v1/summary.md
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_svcbench import (  # noqa: E402
    _load_subset,
    _normalize_subtype,
    _parse_integer,
)
from scripts.evaluate_svcbench_matched import (  # noqa: E402
    _chunks as _runner_chunks,
    _processor_frame_count,
    _timed_question,
)


ROW_SCHEMA = "ttt_frame.svcbench_matched/1"
SUMMARY_SCHEMA = "ttt_frame.svcbench_matched_aggregate/1"
SELECTION_SCHEMA = "ttt_frame.svcbench_real_selection/1"
CONDITIONS = ("blind", "icl", "spatial", "spatial_slow_only", "spatial_base")
CONDITION_SET = set(CONDITIONS)
MOC_SUBTYPES = {"O2-Unique", "E1-Action", "E1-Transit", "E2-Periodic", "E2-Episode"}
UDA_SUBTYPES = {"O1-Snap", *MOC_SUBTYPES}
MEMORY_STATES = {
    "blind": "no_visual_input_at_position_zero",
    "icl": "matched_visual_context",
    "spatial": "episode_fast_state",
    "spatial_slow_only": "official_learned_initial_state",
    "spatial_base": "spatial_branch_bypassed_after_ingest",
}
COMMON_ARM_FIELDS = (
    "schema", "episode_id", "query_episode_independent", "question_id", "q_id",
    "query_index", "source_dataset", "video_path", "video_file", "video_sha256",
    "question", "prompt", "counting_subtype", "query_time", "cutoff_time", "gold",
    "sampled_prefix_frame_count", "selected_frame_count", "selected_sample_indices",
    "frame_timestamps", "frame_bundle_sha256", "future_frames_filtered",
    "max_decoded_timestamp", "run_protocol",
)
SPATIAL_SHARED_BUDGET_FIELDS = (
    "spatial_frames_ingested", "spatial_frames_encoded", "spatial_padding_frames",
    "spatial_write_calls", "spatial_write_calls_planned", "shared_spatial_write",
)
COMPARISON_PROTOCOL_FIELDS = (
    "schema", "runner_sha256", "annotations_sha256", "model_artifact_sha256",
    "base_config_fingerprint", "spatial_checkpoint_sha256", "conditions", "dtype",
    "max_new_tokens", "sampling", "num_shards",
)


class ValidationError(ValueError):
    """Raised when result rows cannot support a valid matched comparison."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _mean(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-8) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _sign(value: int) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _gpa_point(prediction: int, gold: int) -> float:
    sigma = 0.05 * max(gold, 1)
    return math.exp(-((prediction - gold) ** 2) / (2 * sigma ** 2))


def _trajectory_metrics_strict(
    predictions: Sequence[int | None], gts: Sequence[int], subtype: str,
) -> dict[str, Any]:
    """Compute aligned trajectory metrics without dropping invalid positions.

    The official implementation assumes every prediction is an integer.  A
    trajectory containing a parse/runtime failure therefore receives no
    official GPA/MoC/UDA value here.  ``gpa_failure_zero`` is an explicitly
    non-official, conservative companion where an invalid position contributes
    zero.  Exact accuracy always includes every requested query point.
    """

    if len(predictions) != len(gts) or not gts:
        raise ValidationError("trajectory predictions/golds must have equal non-zero length")
    if any(not _is_int(gold) for gold in gts):
        raise ValidationError("trajectory gold counts must be integers")
    if any(prediction is not None and not _is_int(prediction) for prediction in predictions):
        raise ValidationError("trajectory predictions must be integers or null")

    subtype = _normalize_subtype(subtype)
    n_queries = len(gts)
    valid_mask = [prediction is not None for prediction in predictions]
    n_valid = sum(valid_mask)
    complete = n_valid == n_queries
    correct = [prediction is not None and prediction == gold
               for prediction, gold in zip(predictions, gts)]
    valid_correct = sum(correct)
    failure_zero_terms = [
        _gpa_point(int(prediction), int(gold)) if prediction is not None else 0.0
        for prediction, gold in zip(predictions, gts)
    ]
    gpa = None
    moc = None
    uda = None
    moc_applicable = subtype in MOC_SUBTYPES and n_queries >= 2
    uda_applicable = subtype in UDA_SUBTYPES and n_queries >= 2
    if complete:
        integer_predictions = [int(value) for value in predictions if value is not None]
        gpa = sum(_gpa_point(prediction, gold)
                  for prediction, gold in zip(integer_predictions, gts)) / n_queries
        if moc_applicable:
            violation = next(
                (index for index in range(n_queries - 1)
                 if integer_predictions[index + 1] < integer_predictions[index]),
                None,
            )
            moc = 1.0 if violation is None else violation / (n_queries - 1)
        if uda_applicable:
            uda = sum(
                _sign(integer_predictions[index + 1] - integer_predictions[index])
                == _sign(gts[index + 1] - gts[index])
                for index in range(n_queries - 1)
            ) / (n_queries - 1)
    return {
        "n_queries": n_queries,
        "n_valid": n_valid,
        "n_invalid": n_queries - n_valid,
        "coverage": n_valid / n_queries,
        "exact_correct": valid_correct,
        "exact_accuracy_all_queries": valid_correct / n_queries,
        "exact_accuracy_valid_only": valid_correct / n_valid if n_valid else None,
        "official_complete_trajectory": complete,
        "gpa": gpa,
        "gpa_failure_zero": sum(failure_zero_terms) / n_queries,
        "moc": moc,
        "uda": uda,
        "moc_applicable": moc_applicable,
        "uda_applicable": uda_applicable,
        "counting_subtype": subtype,
    }


def _selection_maps(selection: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]], dict[str, str], set[str]
]:
    _require(selection.get("schema") == SELECTION_SCHEMA,
             f"selection must use schema {SELECTION_SCHEMA}")
    videos = selection.get("videos")
    _require(isinstance(videos, list) and videos, "selection must contain a non-empty videos list")
    by_video: dict[str, dict[str, Any]] = {}
    families: dict[str, str] = {}
    question_ids: set[str] = set()
    for entry in videos:
        _require(isinstance(entry, dict), "selection video entries must be objects")
        video = str(entry.get("video", ""))
        _require(video and video not in by_video, f"duplicate or empty selection video: {video!r}")
        digest = str(entry.get("sha256", ""))
        _require(len(digest) == 64, f"selection video {video} lacks a SHA-256")
        ids = [str(value) for value in entry.get("question_ids", [])]
        family_map = entry.get("semantic_families")
        _require(ids and len(ids) == len(set(ids)), f"invalid question_ids for selection video {video}")
        _require(isinstance(family_map, dict) and set(map(str, family_map)) == set(ids),
                 f"semantic_families do not cover question_ids for {video}")
        overlap = question_ids & set(ids)
        _require(not overlap, f"question IDs occur in multiple selection videos: {sorted(overlap)}")
        question_ids.update(ids)
        for question_id in ids:
            family = str(family_map[question_id]).strip()
            _require(bool(family), f"question {question_id} has an empty semantic family")
            families[question_id] = family
        by_video[video] = entry
    _require(int(selection.get("n_videos", len(by_video))) == len(by_video),
             "selection n_videos is inconsistent")
    _require(int(selection.get("n_questions", len(question_ids))) == len(question_ids),
             "selection n_questions is inconsistent")
    _require(int(selection.get("n_semantic_families", len(set(families.values()))))
             == len(set(families.values())), "selection n_semantic_families is inconsistent")
    return by_video, families, question_ids


def _annotation_maps(
    annotation_items: list[dict[str, Any]], selection_questions: set[str],
    selection_videos: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_question: dict[str, dict[str, Any]] = {}
    by_point: dict[tuple[str, str], dict[str, Any]] = {}
    point_count = 0
    for item in annotation_items:
        question_id = str(item.get("id", ""))
        _require(question_id and question_id not in by_question,
                 f"duplicate or empty annotation question ID: {question_id!r}")
        video_key = f"{item.get('source_dataset', '')}/{item.get('video_path', '')}"
        _require(video_key in selection_videos,
                 f"annotation question {question_id} references unselected video {video_key}")
        _require(question_id in {str(value) for value in selection_videos[video_key]["question_ids"]},
                 f"annotation question {question_id} is assigned to the wrong selection video")
        points = item.get("query_points")
        _require(isinstance(points, list) and points,
                 f"annotation question {question_id} has no query points")
        seen_indices: set[int] = set()
        for point in points:
            q_id = str(point.get("q_id", ""))
            index = point.get("query_index")
            _require(q_id and (question_id, q_id) not in by_point,
                     f"duplicate or empty q_id for annotation question {question_id}: {q_id!r}")
            _require(_is_int(index) and int(index) not in seen_indices,
                     f"invalid query_index for annotation point {question_id}/{q_id}")
            _require(_is_int(point.get("count")),
                     f"non-integer gold count for annotation point {question_id}/{q_id}")
            _require(math.isfinite(float(point.get("query_time"))),
                     f"invalid query_time for annotation point {question_id}/{q_id}")
            seen_indices.add(int(index))
            by_point[(question_id, q_id)] = point
            point_count += 1
        by_question[question_id] = item
    _require(set(by_question) == selection_questions,
             "annotation question IDs do not exactly match the selection: "
             f"missing={sorted(selection_questions - set(by_question))}, "
             f"extra={sorted(set(by_question) - selection_questions)}")
    for video_key, entry in selection_videos.items():
        declared_counts = entry.get("question_point_counts")
        if declared_counts is None:  # Backward compatibility with older manifests.
            continue
        expected_ids = {str(value) for value in entry["question_ids"]}
        _require(isinstance(declared_counts, dict),
                 f"selection video {video_key} question_point_counts must be an object")
        declared_by_id = {str(key): value for key, value in declared_counts.items()}
        _require(set(declared_by_id) == expected_ids,
                 f"selection video {video_key} question_point_counts do not exactly cover "
                 f"question_ids: missing={sorted(expected_ids - set(declared_by_id))}, "
                 f"extra={sorted(set(declared_by_id) - expected_ids)}")
        for question_id in sorted(expected_ids):
            declared = declared_by_id[question_id]
            actual = len(by_question[question_id]["query_points"])
            _require(_is_int(declared) and declared == actual,
                     f"selection video {video_key} question_point_counts[{question_id!r}] "
                     f"is {declared!r}, annotation contains {actual}")
    return by_question, by_point


def _protocol_payload(protocol: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in COMPARISON_PROTOCOL_FIELDS if field not in protocol]
    _require(not missing, f"run_protocol is missing comparison fields: {missing}")
    return {field: protocol[field] for field in COMPARISON_PROTOCOL_FIELDS}


def _validate_protocols(
    rows: list[dict[str, Any]], annotation_sha256: str | None,
    selection_videos: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    protocols_by_shard: dict[int, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for row in rows:
        protocol = row.get("run_protocol")
        _require(isinstance(protocol, dict), "every row must contain run_protocol")
        payload = _protocol_payload(protocol)
        computed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        _require(protocol.get("comparison_fingerprint") == computed,
                 "run_protocol comparison_fingerprint does not match its payload")
        _require(payload["schema"] == ROW_SCHEMA, "run_protocol has the wrong row schema")
        _require(tuple(payload["conditions"]) == CONDITIONS,
                 "run_protocol does not declare the exact five-arm condition order")
        if annotation_sha256 is not None:
            _require(payload["annotations_sha256"] == annotation_sha256,
                     "run_protocol annotation hash differs from --annotations")
        _require(isinstance(payload["sampling"], dict), "run_protocol sampling must be an object")
        num_shards = payload["num_shards"]
        shard_index = protocol.get("shard_index")
        _require(_is_int(num_shards) and num_shards >= 1,
                 "run_protocol num_shards must be a positive integer")
        _require(_is_int(shard_index) and 0 <= shard_index < num_shards,
                 "run_protocol shard_index is outside num_shards")
        previous = protocols_by_shard.setdefault(int(shard_index), protocol)
        _require(_canonical(previous) == _canonical(protocol),
                 f"shard {shard_index} mixes run protocols")
        fingerprints.add(str(protocol["comparison_fingerprint"]))
    _require(len(fingerprints) == 1, "input shards have different comparison fingerprints")
    reference = next(iter(protocols_by_shard.values()))
    num_shards = int(reference["num_shards"])
    _require(set(protocols_by_shard) == set(range(num_shards)),
             f"incomplete shard set: expected {list(range(num_shards))}, "
             f"found {sorted(protocols_by_shard)}")
    reference_payload = _protocol_payload(reference)
    for index, protocol in protocols_by_shard.items():
        _require(_canonical(_protocol_payload(protocol)) == _canonical(reference_payload),
                 f"shard {index} comparison payload differs from the other shards")

    ordered_videos = sorted(selection_videos)
    expected_shards = {video: index % num_shards for index, video in enumerate(ordered_videos)}
    for row in rows:
        video_key = f"{row.get('source_dataset', '')}/{row.get('video_path', '')}"
        actual_shard = int(row["run_protocol"]["shard_index"])
        _require(expected_shards.get(video_key) == actual_shard,
                 f"video {video_key} is in shard {actual_shard}, expected {expected_shards.get(video_key)}")
    return reference_payload, protocols_by_shard


def _expected_grid(
    by_question: dict[str, dict[str, Any]], sampling: dict[str, Any],
) -> set[tuple[str, str]]:
    stride = sampling.get("query_stride")
    maximum = sampling.get("max_query_points")
    _require(_is_int(stride) and stride >= 1, "sampling query_stride must be a positive integer")
    _require(_is_int(maximum) and maximum >= 0, "sampling max_query_points must be nonnegative")
    expected: set[tuple[str, str]] = set()
    for question_id, item in by_question.items():
        points = list(item["query_points"])[::int(stride)]
        if maximum:
            points = points[:int(maximum)]
        for point in points:
            expected.add((question_id, str(point["q_id"])))
    return expected


def _validate_row_shape(row: dict[str, Any], row_number: int) -> None:
    required = set(COMMON_ARM_FIELDS) | {
        "condition", "answer", "prediction", "correct", "status", "error", "elapsed_sec",
        "visual_frames_provided_to_answer", "visual_frames_encoded",
        "visual_padding_frames", "visual_encode_passes", "visual_encode_attempted",
        "spatial_frames_ingested", "spatial_frames_encoded", "spatial_padding_frames",
        "spatial_write_calls", "spatial_write_calls_planned", "shared_spatial_write",
        "episode_memory_state", "episode_elapsed_sec",
    }
    missing = required - set(row)
    _require(not missing, f"row {row_number} is missing fields: {sorted(missing)}")
    _require(row.get("schema") == ROW_SCHEMA, f"row {row_number} has unsupported schema")
    _require(str(row.get("condition")) in CONDITION_SET,
             f"row {row_number} has unknown condition {row.get('condition')!r}")
    integer_budget_fields = (
        "visual_frames_provided_to_answer", "visual_encode_passes",
        "spatial_frames_ingested", "spatial_frames_encoded", "spatial_padding_frames",
        "spatial_write_calls", "spatial_write_calls_planned",
    )
    for field in integer_budget_fields:
        _require(_is_int(row[field]) and row[field] >= 0,
                 f"row {row_number} field {field} must be a nonnegative integer")
    for field in ("visual_frames_encoded", "visual_padding_frames"):
        _require(row[field] is None or (_is_int(row[field]) and row[field] >= 0),
                 f"row {row_number} field {field} must be null or a nonnegative integer")
    for field in ("visual_encode_attempted", "shared_spatial_write"):
        _require(isinstance(row[field], bool),
                 f"row {row_number} field {field} must be boolean")


def _validate_prefix_and_budget(row: dict[str, Any]) -> None:
    context = f"{row['condition']}:{row['question_id']}/{row['q_id']}"
    _require(row["query_episode_independent"] is True,
             f"{context} is not marked as an independent query episode")
    _require(row["future_frames_filtered"] is True,
             f"{context} is not marked future-filtered")
    sampled = row["sampled_prefix_frame_count"]
    selected = row["selected_frame_count"]
    _require(_is_int(sampled) and _is_int(selected) and sampled >= selected >= 0,
             f"{context} has invalid sampled/selected frame counts")
    timestamps = row["frame_timestamps"]
    indices = row["selected_sample_indices"]
    _require(isinstance(timestamps, list) and len(timestamps) == selected,
             f"{context} frame_timestamps do not match selected_frame_count")
    _require(isinstance(indices, list) and len(indices) == selected
             and all(_is_int(index) for index in indices),
             f"{context} selected_sample_indices are invalid")
    if indices:
        _require(indices == sorted(set(indices)) and indices[0] >= 0 and indices[-1] < sampled,
                 f"{context} selected_sample_indices are not increasing within the prefix")
    _require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(float(value)) for value in timestamps),
             f"{context} contains invalid frame timestamps")
    _require(all(float(left) <= float(right) + 1e-9
                 for left, right in zip(timestamps, timestamps[1:])),
             f"{context} frame timestamps are not chronological")
    cutoff = float(row["cutoff_time"])
    _require(all(float(timestamp) <= cutoff for timestamp in timestamps),
             f"{context} contains a frame after cutoff_time")
    digest = row["frame_bundle_sha256"]
    _require(
        (selected > 0 and isinstance(digest, str) and len(digest) == 64)
        or (selected == 0 and digest is None),
        f"{context} has an invalid frame-bundle SHA-256",
    )

    protocol = row["run_protocol"]
    sampling = protocol["sampling"]
    max_frames = sampling.get("max_frames")
    write_chunk = sampling.get("write_chunk_frames")
    cutoff_margin = sampling.get("cutoff_margin")
    _require(_is_int(max_frames) and max_frames >= 1,
             "run_protocol max_frames must be a positive integer")
    _require(_is_int(write_chunk) and write_chunk >= 4 and write_chunk % 2 == 0,
             "run_protocol write_chunk_frames must be even and at least four")
    _require(isinstance(cutoff_margin, (int, float)) and not isinstance(cutoff_margin, bool)
             and float(cutoff_margin) >= 0,
             "run_protocol cutoff_margin must be nonnegative")
    _require(selected == min(sampled, int(max_frames)),
             f"{context} selected frame count does not honor max_frames")
    _require(_same_number(row["cutoff_time"], float(row["query_time"]) - float(cutoff_margin)),
             f"{context} cutoff_time is inconsistent with cutoff_margin")

    condition = str(row["condition"])
    _require(row["episode_memory_state"] == MEMORY_STATES[condition],
             f"{context} has the wrong episode_memory_state")
    planned_chunks = [
        len(chunk) for chunk in _runner_chunks(list(range(selected)), int(write_chunk))
    ] if selected else []
    planned_writes = len(planned_chunks)

    if condition == "blind":
        expected = {
            "visual_frames_provided_to_answer": 0,
            "visual_frames_encoded": None,
            "visual_padding_frames": None,
            "visual_encode_passes": 0,
            "visual_encode_attempted": False,
            "spatial_frames_ingested": 0,
            "spatial_frames_encoded": 0,
            "spatial_padding_frames": 0,
            "spatial_write_calls": 0,
            "spatial_write_calls_planned": 0,
            "shared_spatial_write": False,
        }
    elif condition == "icl":
        expected = {
            "spatial_frames_ingested": 0,
            "spatial_frames_encoded": 0,
            "spatial_padding_frames": 0,
            "spatial_write_calls": 0,
            "spatial_write_calls_planned": 0,
            "shared_spatial_write": False,
        }
        for field, value in expected.items():
            _require(row[field] == value,
                     f"{context} has {field}={row[field]!r}, expected {value!r}")
        attempted = row["visual_encode_attempted"]
        _require(isinstance(attempted, bool),
                 f"{context} visual_encode_attempted must be boolean")
        if row["status"] != "error":
            encoded = _processor_frame_count(selected)
            expected = {
                "visual_frames_provided_to_answer": selected,
                "visual_frames_encoded": encoded,
                "visual_padding_frames": encoded - selected,
                "visual_encode_passes": 1,
                "visual_encode_attempted": True,
            }
            _require(selected > 0, f"{context} completed without a selected visual frame")
        elif attempted:
            expected = {
                "visual_frames_provided_to_answer": selected,
                "visual_frames_encoded": None,
                "visual_padding_frames": None,
                "visual_encode_passes": 0,
                "visual_encode_attempted": True,
            }
            _require(selected > 0, f"{context} attempted visual encoding with no frames")
        else:
            expected = {
                "visual_frames_provided_to_answer": 0,
                "visual_frames_encoded": None,
                "visual_padding_frames": None,
                "visual_encode_passes": 0,
                "visual_encode_attempted": False,
            }
            _require(selected == 0,
                     f"{context} skipped visual encoding despite an available prefix")
    else:
        expected = {
            "visual_frames_provided_to_answer": 0,
            "visual_frames_encoded": None,
            "visual_padding_frames": None,
            "visual_encode_passes": 0,
            "visual_encode_attempted": False,
            "spatial_write_calls_planned": planned_writes,
        }
        completed = row["spatial_write_calls"]
        _require(_is_int(completed) and 0 <= completed <= planned_writes,
                 f"{context} spatial_write_calls must be within the planned writes")
        if row["status"] != "error":
            _require(completed == planned_writes and selected > 0,
                     f"{context} successful Spatial read requires every planned write")
        completed_chunks = planned_chunks[:int(completed)]
        real_frames = sum(completed_chunks)
        encoded_frames = sum(_processor_frame_count(size) for size in completed_chunks)
        expected.update({
            "spatial_frames_ingested": real_frames,
            "spatial_frames_encoded": encoded_frames,
            "spatial_padding_frames": encoded_frames - real_frames,
            "shared_spatial_write": completed > 0,
        })
    for field, value in expected.items():
        _require(row[field] == value,
                 f"{context} has {field}={row[field]!r}, expected {value!r}")


def _validate_annotation_row(
    row: dict[str, Any], item: dict[str, Any], point: dict[str, Any],
    selection_entry: dict[str, Any], semantic_family: str,
) -> None:
    context = f"{row['condition']}:{row['question_id']}/{row['q_id']}"
    source = str(item.get("source_dataset", ""))
    video_path = str(item.get("video_path", ""))
    video_key = f"{source}/{video_path}"
    expected_episode = f"{video_key}::{item['id']}::{point['q_id']}"
    cutoff_margin = float(row["run_protocol"]["sampling"]["cutoff_margin"])
    query_time = float(point["query_time"])
    expected_values = {
        "episode_id": expected_episode,
        "query_index": int(point["query_index"]),
        "source_dataset": source,
        "video_path": video_path,
        "video_sha256": str(selection_entry["sha256"]),
        "question": str(item["question"]),
        "prompt": _timed_question(
            str(item["question"]), query_time=query_time,
            cutoff_time=query_time - cutoff_margin,
        ),
        "counting_subtype": item.get(
            "counting_subtype_normalized", _normalize_subtype(item.get("counting_subtype")),
        ),
        "gold": int(point["count"]),
    }
    for field, value in expected_values.items():
        _require(row[field] == value,
                 f"{context} {field} differs from annotation/selection: "
                 f"{row[field]!r} != {value!r}")
    _require(_same_number(row["query_time"], point["query_time"]),
             f"{context} query_time differs from annotation")
    _require(bool(semantic_family), f"{context} has no semantic family")


def _audit_answer(row: dict[str, Any]) -> dict[str, Any]:
    context = f"{row['condition']}:{row['question_id']}/{row['q_id']}"
    _require(isinstance(row["answer"], str), f"{context} answer must be a string")
    parsed = _parse_integer(row["answer"])
    stored = row["prediction"]
    _require(stored is None or _is_int(stored), f"{context} stored prediction is invalid")
    _require(stored == parsed,
             f"{context} stored prediction {stored!r} differs from reparsed answer {parsed!r}")
    expected_correct = parsed is not None and parsed == int(row["gold"])
    _require(isinstance(row["correct"], bool) and row["correct"] == expected_correct,
             f"{context} stored correctness differs from reparsed answer")
    status = row["status"]
    _require(status in {"ok", "parse_error", "error"}, f"{context} has unknown status {status!r}")
    if status == "ok":
        _require(parsed is not None and row["error"] is None,
                 f"{context} ok status is inconsistent with answer/error")
    elif status == "parse_error":
        _require(parsed is None and row["error"] is None,
                 f"{context} parse_error status is inconsistent with answer/error")
    else:
        _require(parsed is None and isinstance(row["error"], str) and bool(row["error"]),
                 f"{context} error status must contain an error and no parsed integer")
    audited = dict(row)
    audited["_audited_prediction"] = parsed
    audited["_audited_correct"] = expected_correct
    return audited


def _exact_group_metrics(
    rows: list[dict[str, Any]], key_function: Callable[[dict[str, Any]], str],
) -> tuple[float | None, dict[str, dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_function(row)].append(row)
    details: dict[str, dict[str, Any]] = {}
    for key, values in sorted(groups.items()):
        n = len(values)
        valid = sum(row["_audited_prediction"] is not None for row in values)
        correct = sum(bool(row["_audited_correct"]) for row in values)
        question_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            question_values[str(row["question_id"])].append(row)
        question_accuracies = {
            question_id: sum(bool(row["_audited_correct"]) for row in question_rows)
            / len(question_rows)
            for question_id, question_rows in sorted(question_values.items())
        }
        details[key] = {
            "n_queries": n,
            "n_questions": len(question_values),
            "n_valid": valid,
            "coverage": valid / n,
            "exact_correct": correct,
            "exact_accuracy_query_micro": correct / n,
            "exact_accuracy_question_macro": _mean(question_accuracies.values()),
            "by_question_exact_accuracy": question_accuracies,
        }
    return _mean(value["exact_accuracy_question_macro"] for value in details.values()), details


def _count_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return descriptive pointwise count calibration with explicit coverage.

    The primary MAE is defined only for a complete arm. A valid-only companion
    is retained for debugging partial failures, but its name makes clear that it
    must not be compared as though failed predictions had disappeared.
    """

    predictions = [
        int(row["_audited_prediction"])
        for row in rows if row["_audited_prediction"] is not None
    ]
    absolute_errors = [
        abs(int(row["_audited_prediction"]) - int(row["gold"]))
        for row in rows if row["_audited_prediction"] is not None
    ]
    total = len(rows)
    valid = len(predictions)
    valid_only_mae = _mean(absolute_errors)
    histogram = Counter(predictions)
    return {
        "n_queries": total,
        "n_valid": valid,
        "n_invalid": total - valid,
        "coverage": valid / total if total else 0.0,
        "mae": valid_only_mae if valid == total else None,
        "mae_requires_complete_coverage": True,
        "mae_valid_only_diagnostic": valid_only_mae,
        "prediction_statistics_scope": "valid_predictions_only",
        "prediction_min": min(predictions) if predictions else None,
        "prediction_max": max(predictions) if predictions else None,
        "prediction_histogram": {
            str(value): int(histogram[value]) for value in sorted(histogram)
        },
    }


def _condition_metrics(
    rows: list[dict[str, Any]], trajectory_metrics: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    n = len(rows)
    valid = sum(row["_audited_prediction"] is not None for row in rows)
    correct = sum(bool(row["_audited_correct"]) for row in rows)
    question_macro = _mean(
        metric["exact_accuracy_all_queries"] for metric in trajectory_metrics.values()
    )
    video_macro, by_video = _exact_group_metrics(
        rows, lambda row: f"{row['source_dataset']}/{row['video_path']}",
    )
    family_macro, by_family = _exact_group_metrics(
        rows, lambda row: str(row["_semantic_family"]),
    )
    official_gpa = [metric["gpa"] for metric in trajectory_metrics.values()
                    if metric["gpa"] is not None]
    applicable_moc = [metric for metric in trajectory_metrics.values()
                      if metric["moc_applicable"]]
    applicable_uda = [metric for metric in trajectory_metrics.values()
                      if metric["uda_applicable"]]
    evaluated_moc = [metric["moc"] for metric in applicable_moc if metric["moc"] is not None]
    evaluated_uda = [metric["uda"] for metric in applicable_uda if metric["uda"] is not None]
    summary = {
        "n_questions": len(trajectory_metrics),
        "n_queries": n,
        "n_valid": valid,
        "n_parse_or_runtime_failures": n - valid,
        "n_runtime_errors": sum(row["status"] == "error" for row in rows),
        "coverage_query_micro": valid / n,
        "exact_accuracy_query_micro": correct / n,
        "exact_accuracy_question_macro": question_macro,
        "exact_accuracy_source_video_macro": video_macro,
        "exact_accuracy_semantic_family_macro": family_macro,
        "count_calibration": _count_calibration(rows),
        "official_metrics": {
            "gpa_question_macro": _mean(official_gpa),
            "gpa_complete_trajectories": len(official_gpa),
            "gpa_applicable_trajectories": len(trajectory_metrics),
            "gpa_trajectory_coverage": len(official_gpa) / len(trajectory_metrics),
            "gpa_failure_zero_question_macro_non_official": _mean(
                metric["gpa_failure_zero"] for metric in trajectory_metrics.values()
            ),
            "moc_question_macro": _mean(evaluated_moc),
            "moc_evaluated_trajectories": len(evaluated_moc),
            "moc_applicable_trajectories": len(applicable_moc),
            "moc_trajectory_coverage": (
                len(evaluated_moc) / len(applicable_moc) if applicable_moc else None
            ),
            "uda_question_macro": _mean(evaluated_uda),
            "uda_evaluated_trajectories": len(evaluated_uda),
            "uda_applicable_trajectories": len(applicable_uda),
            "uda_trajectory_coverage": (
                len(evaluated_uda) / len(applicable_uda) if applicable_uda else None
            ),
        },
        "frame_budget_totals": {
            "selected_real_frames": sum(int(row["selected_frame_count"]) for row in rows),
            "visual_frames_provided_to_answer": sum(
                int(row["visual_frames_provided_to_answer"]) for row in rows
            ),
            "visual_frames_encoded_completed": sum(
                int(row["visual_frames_encoded"] or 0) for row in rows
            ),
            "visual_padding_frames_completed": sum(
                int(row["visual_padding_frames"] or 0) for row in rows
            ),
            "visual_encode_attempts": sum(bool(row["visual_encode_attempted"]) for row in rows),
            "visual_encode_passes_completed": sum(
                int(row["visual_encode_passes"]) for row in rows
            ),
            "spatial_real_frames_ingested": sum(
                int(row["spatial_frames_ingested"]) for row in rows
            ),
            "spatial_frames_encoded": sum(
                int(row["spatial_frames_encoded"]) for row in rows
            ),
            "spatial_padding_frames": sum(
                int(row["spatial_padding_frames"]) for row in rows
            ),
            "spatial_write_calls_completed": sum(
                int(row["spatial_write_calls"]) for row in rows
            ),
            "spatial_write_calls_planned": sum(
                int(row["spatial_write_calls_planned"]) for row in rows
            ),
        },
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
    }
    return summary, {"source_video": by_video, "semantic_family": by_family}


def _paired_exact_delta(
    by_condition: dict[str, dict[tuple[str, str], dict[str, Any]]],
    left: str,
    right: str,
) -> dict[str, Any]:
    left_rows = by_condition[left]
    right_rows = by_condition[right]
    keys = sorted(left_rows)
    _require(set(keys) == set(right_rows), f"paired arms {left}/{right} have different grids")
    outcome = [
        (bool(left_rows[key]["_audited_correct"]), bool(right_rows[key]["_audited_correct"]))
        for key in keys
    ]

    def grouped_delta(group: Callable[[dict[str, Any]], str]) -> tuple[float | None, dict[str, float]]:
        questions: dict[str, list[int]] = defaultdict(list)
        representative: dict[str, dict[str, Any]] = {}
        for key in keys:
            left_row, right_row = left_rows[key], right_rows[key]
            question_id = str(left_row["question_id"])
            representative[question_id] = left_row
            questions[question_id].append(
                int(bool(left_row["_audited_correct"]))
                - int(bool(right_row["_audited_correct"]))
            )
        question_delta = {
            question_id: sum(items) / len(items)
            for question_id, items in questions.items()
        }
        values: dict[str, list[float]] = defaultdict(list)
        for question_id, delta in question_delta.items():
            values[group(representative[question_id])].append(delta)
        detail = {name: sum(items) / len(items) for name, items in sorted(values.items())}
        return _mean(detail.values()), detail

    question_macro, by_question = grouped_delta(lambda row: str(row["question_id"]))
    video_macro, by_video = grouped_delta(
        lambda row: f"{row['source_dataset']}/{row['video_path']}"
    )
    family_macro, by_family = grouped_delta(lambda row: str(row["_semantic_family"]))
    left_only = sum(left_ok and not right_ok for left_ok, right_ok in outcome)
    right_only = sum(right_ok and not left_ok for left_ok, right_ok in outcome)
    return {
        "left": left,
        "right": right,
        "matched_queries": len(keys),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_correct": sum(left_ok and right_ok for left_ok, right_ok in outcome),
        "both_wrong": sum(not left_ok and not right_ok for left_ok, right_ok in outcome),
        "exact_accuracy_query_micro_delta": (left_only - right_only) / len(keys),
        "exact_accuracy_question_macro_delta": question_macro,
        "exact_accuracy_source_video_macro_delta": video_macro,
        "exact_accuracy_semantic_family_macro_delta": family_macro,
        "by_question_delta": by_question,
        "by_source_video_delta": by_video,
        "by_semantic_family_delta": by_family,
    }


def _trajectory_order_sensitivity(
    audited_grid: dict[str, dict[tuple[str, str], dict[str, Any]]],
    by_question: dict[str, dict[str, Any]],
    official_trajectories: dict[str, dict[str, dict[str, Any]]],
    condition_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare official source order with chronological query-time order.

    SVCBench's released evaluator preserves the source JSONL order, represented
    here by ``query_index``. Most trajectories are already chronological, but a
    source annotation can contain a different timestamp order. The official
    metrics remain primary; this companion exposes UDA sensitivity without
    silently changing benchmark semantics.
    """

    mismatches: dict[str, dict[str, Any]] = {}
    chronological_metrics: dict[str, dict[str, dict[str, Any]]] = {
        condition: {} for condition in CONDITIONS
    }
    for question_id, item in sorted(by_question.items()):
        reference_rows = [
            row for (candidate_id, _), row in audited_grid[CONDITIONS[0]].items()
            if candidate_id == question_id
        ]
        official_rows = sorted(
            reference_rows,
            key=lambda row: (int(row["query_index"]), float(row["query_time"])),
        )
        chronological_rows = sorted(
            reference_rows,
            key=lambda row: (float(row["query_time"]), int(row["query_index"])),
        )
        official_ids = [str(row["q_id"]) for row in official_rows]
        chronological_ids = [str(row["q_id"]) for row in chronological_rows]
        if official_ids != chronological_ids:
            def order_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [
                    {
                        "q_id": str(row["q_id"]),
                        "query_index": int(row["query_index"]),
                        "query_time": float(row["query_time"]),
                        "gold": int(row["gold"]),
                    }
                    for row in rows
                ]

            mismatches[question_id] = {
                "source_video": f"{item['source_dataset']}/{item['video_path']}",
                "question": str(item["question"]),
                "counting_subtype": item.get(
                    "counting_subtype_normalized",
                    _normalize_subtype(item.get("counting_subtype")),
                ),
                "official_query_index_order": order_payload(official_rows),
                "chronological_query_time_order": order_payload(chronological_rows),
            }

        subtype = item.get(
            "counting_subtype_normalized", _normalize_subtype(item.get("counting_subtype")),
        )
        for condition in CONDITIONS:
            rows = [
                row for (candidate_id, _), row in audited_grid[condition].items()
                if candidate_id == question_id
            ]
            rows.sort(key=lambda row: (float(row["query_time"]), int(row["query_index"])))
            chronological_metrics[condition][question_id] = _trajectory_metrics_strict(
                [row["_audited_prediction"] for row in rows],
                [int(row["gold"]) for row in rows],
                str(subtype),
            )

    by_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        chronological_values = [
            metric["uda"] for metric in chronological_metrics[condition].values()
            if metric["uda"] is not None
        ]
        official_value = condition_summaries[condition]["official_metrics"][
            "uda_question_macro"
        ]
        chronological_value = _mean(chronological_values)
        details: dict[str, dict[str, Any]] = {}
        for question_id in sorted(mismatches):
            official_uda = official_trajectories[condition][question_id]["uda"]
            chronological_uda = chronological_metrics[condition][question_id]["uda"]
            details[question_id] = {
                "official_uda": official_uda,
                "chronological_uda": chronological_uda,
                "chronological_minus_official": (
                    chronological_uda - official_uda
                    if chronological_uda is not None and official_uda is not None
                    else None
                ),
            }
        by_condition[condition] = {
            "official_uda_question_macro": official_value,
            "chronological_uda_question_macro": chronological_value,
            "chronological_minus_official": (
                chronological_value - official_value
                if chronological_value is not None and official_value is not None
                else None
            ),
            "official_evaluated_trajectories": condition_summaries[condition][
                "official_metrics"
            ]["uda_evaluated_trajectories"],
            "chronological_evaluated_trajectories": len(chronological_values),
            "order_mismatch_details": details,
        }

    return {
        "primary_order": "official source order: query_index, then query_time",
        "sensitivity_order": "chronological order: query_time, then query_index",
        "primary_metrics_unchanged": True,
        "n_order_mismatch_trajectories": len(mismatches),
        "order_mismatch_question_ids": sorted(mismatches),
        "order_mismatches": mismatches,
        "by_condition": by_condition,
    }


def _validate_and_summarize(
    rows: list[dict[str, Any]],
    annotation_items: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    annotation_sha256: str | None = None,
    input_provenance: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate an in-memory result suite and return its audited summary."""

    _require(bool(rows), "no result rows supplied")
    selection_videos, semantic_families, selection_questions = _selection_maps(selection)
    by_question, by_point = _annotation_maps(
        annotation_items, selection_questions, selection_videos,
    )
    reference_protocol, protocols_by_shard = _validate_protocols(
        rows, annotation_sha256, selection_videos,
    )
    expected_grid = _expected_grid(by_question, reference_protocol["sampling"])

    raw_grid: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    audited_grid: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row_number, row in enumerate(rows, 1):
        _require(isinstance(row, dict), f"row {row_number} is not a JSON object")
        _validate_row_shape(row, row_number)
        condition = str(row["condition"])
        question_id = str(row["question_id"])
        q_id = str(row["q_id"])
        key = (question_id, q_id)
        _require(key not in raw_grid[condition],
                 f"duplicate row for {(condition, question_id, q_id)}")
        _require(question_id in by_question and key in by_point,
                 f"row references unknown annotation point {question_id}/{q_id}")
        item, point = by_question[question_id], by_point[key]
        video_key = f"{item.get('source_dataset', '')}/{item.get('video_path', '')}"
        _validate_annotation_row(
            row, item, point, selection_videos[video_key], semantic_families[question_id],
        )
        _validate_prefix_and_budget(row)
        audited = _audit_answer(row)
        audited["_semantic_family"] = semantic_families[question_id]
        raw_grid[condition][key] = row
        audited_grid[condition][key] = audited

    _require(set(raw_grid) == CONDITION_SET,
             f"expected conditions {list(CONDITIONS)}, found {sorted(raw_grid)}")
    for condition in CONDITIONS:
        actual = set(raw_grid[condition])
        _require(actual == expected_grid,
                 f"condition {condition} has an incomplete/mismatched grid; "
                 f"missing={sorted(expected_grid - actual)[:10]}, "
                 f"extra={sorted(actual - expected_grid)[:10]}")

    reference_condition = CONDITIONS[0]
    for key in sorted(expected_grid):
        reference = raw_grid[reference_condition][key]
        reference_common = {field: reference[field] for field in COMMON_ARM_FIELDS}
        for condition in CONDITIONS[1:]:
            candidate = raw_grid[condition][key]
            candidate_common = {field: candidate[field] for field in COMMON_ARM_FIELDS}
            _require(_canonical(candidate_common) == _canonical(reference_common),
                     f"arms {reference_condition}/{condition} have mismatched prefix, "
                     f"question, gold, video, or run protocol for {key}")
        spatial_reference = raw_grid["spatial"][key]
        shared_budget = {
            field: spatial_reference[field] for field in SPATIAL_SHARED_BUDGET_FIELDS
        }
        for condition in ("spatial_slow_only", "spatial_base"):
            candidate_budget = {
                field: raw_grid[condition][key][field]
                for field in SPATIAL_SHARED_BUDGET_FIELDS
            }
            _require(_canonical(candidate_budget) == _canonical(shared_budget),
                     f"Spatial arms do not share the same completed write budget for {key}")

    # The independent-episode protocol still yields one ordered trajectory per
    # question and arm for metric computation.
    trajectories: dict[str, dict[str, dict[str, Any]]] = {}
    condition_summaries: dict[str, dict[str, Any]] = {}
    by_source_video: dict[str, dict[str, dict[str, Any]]] = {}
    by_semantic_family: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        condition_trajectories: dict[str, dict[str, Any]] = {}
        for question_id, item in sorted(by_question.items()):
            question_rows = [row for (candidate_id, _), row in audited_grid[condition].items()
                             if candidate_id == question_id]
            question_rows.sort(key=lambda row: (int(row["query_index"]), float(row["query_time"])))
            subtype = item.get(
                "counting_subtype_normalized", _normalize_subtype(item.get("counting_subtype")),
            )
            metric = _trajectory_metrics_strict(
                [row["_audited_prediction"] for row in question_rows],
                [int(row["gold"]) for row in question_rows],
                str(subtype),
            )
            metric.update({
                "question_id": question_id,
                "question": str(item["question"]),
                "source_video": f"{item['source_dataset']}/{item['video_path']}",
                "semantic_family": semantic_families[question_id],
            })
            condition_trajectories[question_id] = metric
        condition_rows = list(audited_grid[condition].values())
        condition_summary, grouped = _condition_metrics(
            condition_rows, condition_trajectories,
        )
        trajectories[condition] = condition_trajectories
        condition_summaries[condition] = condition_summary
        by_source_video[condition] = grouped["source_video"]
        by_semantic_family[condition] = grouped["semantic_family"]

    pairs = (
        ("icl", "blind"),
        ("spatial", "spatial_slow_only"),
        ("spatial", "spatial_base"),
    )
    paired = {
        f"{left}_minus_{right}": _paired_exact_delta(audited_grid, left, right)
        for left, right in pairs
    }
    order_sensitivity = _trajectory_order_sensitivity(
        audited_grid, by_question, trajectories, condition_summaries,
    )
    expected_query_points = len(expected_grid)
    selection_declared_points = selection.get("n_query_points")
    if (int(reference_protocol["sampling"]["query_stride"]) == 1
            and int(reference_protocol["sampling"]["max_query_points"]) == 0
            and selection_declared_points is not None):
        _require(int(selection_declared_points) == expected_query_points,
                 "selection n_query_points is inconsistent with annotations")

    return {
        "schema": SUMMARY_SCHEMA,
        "protocol_validation": "passed",
        "raw_answers_reparsed": True,
        "error_policy": "intention-to-evaluate: parse and runtime failures count as incorrect",
        "invalid_trajectory_policy": (
            "official GPA/MoC/UDA are null for a trajectory containing any invalid prediction; "
            "positions are never removed or compressed"
        ),
        "official_metric_applicability": {
            "gpa": "all complete trajectories",
            "moc_subtypes": sorted(MOC_SUBTYPES),
            "uda_subtypes": sorted(UDA_SUBTYPES),
        },
        "inputs": input_provenance or [],
        "selection": {
            "name": selection.get("name"),
            "dataset_repo": selection.get("dataset_repo"),
            "dataset_revision": selection.get("dataset_revision"),
            "source_type": selection.get("source_type"),
            "selection_policy": selection.get("selection_policy"),
            "source_annotation_sha256": selection.get("annotation_sha256"),
            "n_videos": len(selection_videos),
            "n_questions": len(selection_questions),
            "n_semantic_families": len(set(semantic_families.values())),
            "n_query_points": expected_query_points,
        },
        "comparison_fingerprint": next(iter(protocols_by_shard.values()))[
            "comparison_fingerprint"
        ],
        "run_protocol_comparison_payload": reference_protocol,
        "shards": {
            "expected": int(reference_protocol["num_shards"]),
            "found": sorted(protocols_by_shard),
            "devices": {str(index): protocol.get("device")
                        for index, protocol in sorted(protocols_by_shard.items())},
        },
        "n_rows": len(rows),
        "expected_rows": expected_query_points * len(CONDITIONS),
        "conditions": list(CONDITIONS),
        "by_condition": condition_summaries,
        "trajectories": trajectories,
        "by_source_video": by_source_video,
        "by_semantic_family": by_semantic_family,
        "paired_exact_deltas": paired,
        "trajectory_order_sensitivity": order_sensitivity,
        "limitations": [
            (
                f"This evaluation contains {len(selection_videos)} source videos, "
                f"{len(selection_questions)} question trajectories, and "
                f"{len(set(semantic_families.values()))} semantic families."
            ),
            (
                "Query points and semantically overlapping questions from one source video "
                f"are correlated; the {expected_query_points} points are not IID."
            ),
            "Question-, source-video-, and semantic-family macro accuracy are descriptive at this dataset size.",
            "The task probes observed-prefix state/counting, not future-action prediction.",
        ],
    }


def _verify_source_annotation(
    annotation_path: Path, annotation_payload: Any, selection: dict[str, Any],
) -> dict[str, Any] | None:
    expected = selection.get("annotation_sha256")
    if not expected:
        return None
    _require(isinstance(annotation_payload, dict),
             "selection provenance requires a grouped annotation JSON object")
    source_value = annotation_payload.get("source_annotation")
    _require(bool(source_value), "subset annotation does not identify source_annotation")
    source_path = Path(str(source_value)).expanduser()
    if not source_path.is_absolute():
        source_path = annotation_path.parent / source_path
    _require(source_path.is_file(), f"source annotation does not exist: {source_path}")
    actual = _sha256_file(source_path)
    _require(actual == expected,
             "selection source annotation SHA-256 differs from the referenced source file")
    return {"path": str(source_path.resolve()), "sha256": actual}


def _verify_video_artifacts(
    rows: list[dict[str, Any]], selection: dict[str, Any],
) -> list[dict[str, Any]]:
    selection_videos, _, _ = _selection_maps(selection)
    paths: dict[str, set[str]] = defaultdict(set)
    row_hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        video_key = f"{row.get('source_dataset', '')}/{row.get('video_path', '')}"
        paths[video_key].add(str(row.get("video_file", "")))
        row_hashes[video_key].add(str(row.get("video_sha256", "")))
    artifacts = []
    for video_key, entry in sorted(selection_videos.items()):
        _require(len(paths.get(video_key, set())) == 1,
                 f"video {video_key} has missing or inconsistent video_file paths")
        _require(row_hashes.get(video_key) == {str(entry["sha256"])},
                 f"video {video_key} row hashes differ from the selection")
        path = Path(next(iter(paths[video_key]))).expanduser()
        _require(path.is_file(), f"result video file does not exist: {path}")
        actual = _sha256_file(path)
        _require(actual == entry["sha256"], f"video bytes differ from selection: {video_key}")
        artifacts.append({
            "video": video_key,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": actual,
        })
    return artifacts


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _format_histogram(value: dict[str, int]) -> str:
    return "`" + json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "`"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    selection = summary["selection"]
    selection_name = str(selection.get("name") or "selection manifest")
    source_type = selection.get("source_type")
    lines = [
        "# SVCBench real-video matched evaluation", "",
        "All five arms passed the matched-prefix audit. Raw answers were reparsed, and parse or "
        "runtime failures count as wrong in every exact-accuracy denominator.", "",
        f"Data: **{selection['n_videos']} source videos** in **{selection_name}**, "
        f"**{selection['n_questions']} question trajectories**, "
        f"**{selection['n_semantic_families']} semantic families**, and "
        f"**{selection['n_query_points']} query points**. The suite contains "
        f"**{summary['n_rows']} rows** across five arms.", "",
    ]
    if source_type:
        lines += [f"Selection scope from the manifest: {source_type}.", ""]
    lines += [
        "## Main results", "",
        "| arm | exact (queries) | question macro | video macro | family macro | parse coverage | GPA | MoC | UDA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metric = summary["by_condition"][condition]
        official = metric["official_metrics"]
        lines.append(
            f"| {condition} | {_format_metric(metric['exact_accuracy_query_micro'])} | "
            f"{_format_metric(metric['exact_accuracy_question_macro'])} | "
            f"{_format_metric(metric['exact_accuracy_source_video_macro'])} | "
            f"{_format_metric(metric['exact_accuracy_semantic_family_macro'])} | "
            f"{_format_metric(metric['coverage_query_micro'])} | "
            f"{_format_metric(official['gpa_question_macro'])} | "
            f"{_format_metric(official['moc_question_macro'])} | "
            f"{_format_metric(official['uda_question_macro'])} |"
        )
    lines += [
        "", "GPA is applicable to all question trajectories. MoC and UDA follow the official "
        "SVCBench subtype sets. An official metric is omitted for a trajectory if any answer "
        "failed to parse; invalid positions are never deleted from the sequence.", "",
        "## Count calibration (descriptive)", "",
        "MAE is query-micro absolute count error and is reported only when every requested "
        "answer in that arm parsed successfully. With incomplete coverage it is shown as `—`, "
        "rather than silently deleting failures. Prediction min/max and the histogram describe "
        "parsed predictions only, with their coverage shown in the same row.", "",
        "| arm | parsed / queries | coverage | MAE (complete coverage) | prediction min | prediction max | prediction histogram |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for condition in CONDITIONS:
        calibration = summary["by_condition"][condition]["count_calibration"]
        lines.append(
            f"| {condition} | {calibration['n_valid']} / {calibration['n_queries']} | "
            f"{_format_metric(calibration['coverage'])} | "
            f"{_format_metric(calibration['mae'])} | "
            f"{_format_metric(calibration['prediction_min'], digits=0)} | "
            f"{_format_metric(calibration['prediction_max'], digits=0)} | "
            f"{_format_histogram(calibration['prediction_histogram'])} |"
        )
    lines += [
        "", "A valid-only MAE is retained in JSON as `mae_valid_only_diagnostic` for debugging "
        "partial runs; it is not a replacement for the full-coverage MAE.", "",
        "## Paired exact-accuracy deltas", "",
        "| comparison | query delta | question macro | video macro | family macro | left-only / right-only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metric in summary["paired_exact_deltas"].items():
        lines.append(
            f"| {name} | {_format_metric(metric['exact_accuracy_query_micro_delta'])} | "
            f"{_format_metric(metric['exact_accuracy_question_macro_delta'])} | "
            f"{_format_metric(metric['exact_accuracy_source_video_macro_delta'])} | "
            f"{_format_metric(metric['exact_accuracy_semantic_family_macro_delta'])} | "
            f"{metric['left_only_correct']} / {metric['right_only_correct']} |"
        )

    lines += ["", "## Source-video exact accuracy (question macro within video)", "",
              "| source video | " + " | ".join(CONDITIONS) + " |",
              "|---|" + "---:|" * len(CONDITIONS)]
    videos = sorted(summary["by_source_video"][CONDITIONS[0]])
    for video in videos:
        values = [summary["by_source_video"][condition][video]["exact_accuracy_question_macro"]
                  for condition in CONDITIONS]
        lines.append(f"| {video} | " + " | ".join(_format_metric(value) for value in values) + " |")

    lines += ["", "## Semantic-family exact accuracy (question macro within family)", "",
              "| semantic family | " + " | ".join(CONDITIONS) + " |",
              "|---|" + "---:|" * len(CONDITIONS)]
    families = sorted(summary["by_semantic_family"][CONDITIONS[0]])
    for family in families:
        values = [summary["by_semantic_family"][condition][family]["exact_accuracy_question_macro"]
                  for condition in CONDITIONS]
        lines.append(f"| {family} | " + " | ".join(_format_metric(value) for value in values) + " |")

    lines += ["", "## Coverage", "",
              "| arm | valid / queries | complete GPA trajectories | MoC evaluated / applicable | UDA evaluated / applicable |",
              "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        metric = summary["by_condition"][condition]
        official = metric["official_metrics"]
        lines.append(
            f"| {condition} | {metric['n_valid']} / {metric['n_queries']} | "
            f"{official['gpa_complete_trajectories']} / {official['gpa_applicable_trajectories']} | "
            f"{official['moc_evaluated_trajectories']} / {official['moc_applicable_trajectories']} | "
            f"{official['uda_evaluated_trajectories']} / {official['uda_applicable_trajectories']} |"
        )

    sensitivity = summary["trajectory_order_sensitivity"]
    lines += [
        "", "## Query-order sensitivity", "",
        "The primary GPA/MoC/UDA values above retain SVCBench's released source order "
        "(`query_index`, then `query_time`). The sensitivity below sorts each trajectory "
        "chronologically (`query_time`, then `query_index`) and changes no primary metric.", "",
        f"Trajectories whose two orders differ: **{sensitivity['n_order_mismatch_trajectories']}**.", "",
    ]
    if sensitivity["order_mismatches"]:
        lines += [
            "| question | source video | official time order | chronological time order |",
            "|---|---|---|---|",
        ]
        for question_id, detail in sensitivity["order_mismatches"].items():
            official_times = ", ".join(
                _format_metric(point["query_time"], digits=2)
                for point in detail["official_query_index_order"]
            )
            chronological_times = ", ".join(
                _format_metric(point["query_time"], digits=2)
                for point in detail["chronological_query_time_order"]
            )
            lines.append(
                f"| {question_id} | {detail['source_video']} | {official_times} | "
                f"{chronological_times} |"
            )
        lines += [
            "", "| arm | official UDA | chronological UDA | chronological − official |",
            "|---|---:|---:|---:|",
        ]
        for condition in CONDITIONS:
            metric = sensitivity["by_condition"][condition]
            lines.append(
                f"| {condition} | {_format_metric(metric['official_uda_question_macro'])} | "
                f"{_format_metric(metric['chronological_uda_question_macro'])} | "
                f"{_format_metric(metric['chronological_minus_official'])} |"
            )
        lines += [
            "", "Per-trajectory UDA values for the order mismatches:", "",
            "| question | arm | official UDA | chronological UDA | chronological − official |",
            "|---|---|---:|---:|---:|",
        ]
        for question_id in sensitivity["order_mismatch_question_ids"]:
            for condition in CONDITIONS:
                detail = sensitivity["by_condition"][condition][
                    "order_mismatch_details"
                ][question_id]
                lines.append(
                    f"| {question_id} | {condition} | "
                    f"{_format_metric(detail['official_uda'])} | "
                    f"{_format_metric(detail['chronological_uda'])} | "
                    f"{_format_metric(detail['chronological_minus_official'])} |"
                )
    lines += ["", "## Limits", ""]
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines += ["", "The paired deltas are descriptive. No row-level significance test is reported "
              "because query points and questions sharing a source video are correlated."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="matched runner shard JSONL files")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="grouped SVCBench subset JSON used by the runner")
    parser.add_argument("--selection", type=Path,
                        default=REPO_ROOT / "data/svcbench_real_ego4d_pilot_v1.json")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/svcbench_real_matched_v1/summary.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("runs/svcbench_real_matched_v1/summary.md"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _require(args.annotations.is_file(), f"annotations do not exist: {args.annotations}")
    _require(args.selection.is_file(), f"selection does not exist: {args.selection}")
    _require(len({str(path.resolve()) for path in args.inputs}) == len(args.inputs),
             "the same input shard was supplied more than once")
    for path in args.inputs:
        _require(path.is_file(), f"input shard does not exist: {path}")
    input_paths = {str(path.resolve()) for path in args.inputs}
    _require(args.output.resolve() != args.report.resolve(),
             "--output and --report must be different files")
    _require(str(args.output.resolve()) not in input_paths
             and str(args.report.resolve()) not in input_paths,
             "summary destinations cannot overwrite an input shard")
    for destination in (args.output, args.report):
        if destination.exists() and not args.overwrite:
            raise SystemExit(f"output already exists (pass --overwrite): {destination}")

    annotation_bytes = args.annotations.read_bytes()
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    annotation_payload = json.loads(annotation_bytes)
    annotation_items = _load_subset(args.annotations)
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    source_annotation = _verify_source_annotation(
        args.annotations, annotation_payload, selection,
    )

    rows: list[dict[str, Any]] = []
    input_provenance: list[dict[str, Any]] = []
    for path in args.inputs:
        shard_rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            _require(isinstance(row, dict), f"non-object JSON row in {path}:{line_number}")
            row = dict(row)
            row["_input_path"] = str(path.resolve())
            shard_rows.append(row)
        _require(bool(shard_rows), f"input shard is empty: {path}")
        shard_indices = {
            row.get("run_protocol", {}).get("shard_index")
            for row in shard_rows if isinstance(row.get("run_protocol"), dict)
        }
        _require(len(shard_indices) == 1,
                 f"input file {path} contains multiple or missing shard indices")
        rows.extend(shard_rows)
        input_provenance.append({
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "rows": len(shard_rows),
            "shard_index": next(iter(shard_indices)),
        })

    summary = _validate_and_summarize(
        rows, annotation_items, selection,
        annotation_sha256=annotation_sha256,
        input_provenance=input_provenance,
    )
    summary["summarizer_sha256"] = _sha256_file(Path(__file__))
    summary["annotations"] = {
        "path": str(args.annotations.resolve()),
        "sha256": annotation_sha256,
        "source_annotation": source_annotation,
    }
    summary["selection_manifest"] = {
        "path": str(args.selection.resolve()),
        "sha256": hashlib.sha256(selection_bytes).hexdigest(),
    }
    summary["video_artifacts"] = _verify_video_artifacts(rows, selection)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, summary)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report": str(args.report.resolve()),
        "rows": summary["n_rows"],
        "query_points": summary["selection"]["n_query_points"],
        "conditions": summary["conditions"],
        "protocol_validation": summary["protocol_validation"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
