"""Candidate mining utilities for causal HomeSentinel temporal tasks.

This module deliberately does *not* create evaluation labels.  HomeSentinel's
merged captions and ``benchmark_tmp`` object files are automatic intermediate
artifacts.  They are useful for finding moments that a human should inspect,
but their prose must not become a question's gold answer without reviewing the
source video.

The evidence index is built from complete identifiers instead of splitting on
``-``.  This matters because valid HomeSentinel video IDs can themselves end in
suffixes such as ``-1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "homesentinel_temporal_candidate_manifest_v1"
DEFAULT_DATA_ROOT = Path("/data/HomeSentinel/asuka")
DEFAULT_OUTPUT = Path("data/homesentinel_temporal_pilot_v1.json")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def parse_time_seconds(value: Any) -> float:
    """Parse ``MM:SS`` or ``HH:MM:SS`` timestamps into seconds."""

    text = str(value or "").strip()
    fields = text.split(":")
    if len(fields) not in (2, 3):
        raise ValueError(f"unsupported timestamp: {value!r}")
    try:
        numbers = [float(field) for field in fields]
    except ValueError as exc:
        raise ValueError(f"unsupported timestamp: {value!r}") from exc
    if any(number < 0 for number in numbers) or numbers[-1] >= 60:
        raise ValueError(f"unsupported timestamp: {value!r}")
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    if minutes >= 60:
        raise ValueError(f"unsupported timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def make_evidence_id(video_id: str, scene_id: str, segment_id: str) -> str:
    """Return the canonical evidence ID without parsing the video ID."""

    return f"{video_id}-{scene_id}-{segment_id}"


def _caption_event_span(segment: dict[str, Any]) -> tuple[float, float] | None:
    times: list[float] = []
    for event in segment.get("events", []):
        if not isinstance(event, dict) or event.get("t") in (None, ""):
            continue
        try:
            times.append(parse_time_seconds(event["t"]))
        except ValueError:
            continue
    if not times:
        return None
    return min(times), max(times)


def _scene_ranges(data_root: Path, video_id: str) -> tuple[dict[str, tuple[float, float]], str]:
    metadata_path = data_root / video_id / "dataset_meta.json"
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"dataset metadata must be an object: {metadata_path}")
    ranges: dict[str, tuple[float, float]] = {}
    for scene in metadata.get("valid_scenes", []):
        if not isinstance(scene, dict) or not scene.get("scene_id"):
            continue
        try:
            start, end = float(scene["start"]), float(scene["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        ranges[str(scene["scene_id"])] = (start, end)
    return ranges, str(metadata_path.resolve())


def build_evidence_index(
    data_root: Path,
    merged_captions_path: Path | None = None,
    *,
    video_filename: str = "raw_video.mp4",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Map caption evidence IDs to ranges on the source-media timeline.

    ``dataset_meta.valid_scenes`` supplies the authoritative media envelope.
    Automatic caption event timestamps optionally give a narrower span for
    candidate ordering.  The latter remains explicitly marked as automatic
    and is never presented as verified annotation.
    """

    data_root = Path(data_root).resolve()
    merged_path = Path(merged_captions_path or data_root / "merged_captions.json").resolve()
    merged = _load_json(merged_path)
    if not isinstance(merged, list):
        raise ValueError(f"merged captions must be a list: {merged_path}")

    index: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    for video in merged:
        if not isinstance(video, dict) or not video.get("video_id"):
            issues.append({"reason": "invalid_merged_video_record", "record": video})
            continue
        video_id = str(video["video_id"])
        try:
            video_idx = int(video["video_idx"])
        except (KeyError, TypeError, ValueError):
            issues.append({"reason": "invalid_video_idx", "video_id": video_id})
            continue
        try:
            scenes, metadata_path = _scene_ranges(data_root, video_id)
        except ValueError as exc:
            issues.append({
                "reason": "unavailable_dataset_metadata",
                "video_id": video_id,
                "detail": str(exc),
            })
            continue
        video_path = (data_root / video_id / video_filename).resolve()
        if not video_path.is_file():
            issues.append({
                "reason": "missing_video_file",
                "video_id": video_id,
                "video_path": str(video_path),
            })

        segments = video.get("segments", [])
        if not isinstance(segments, list):
            issues.append({"reason": "invalid_segments", "video_id": video_id})
            continue
        for segment in segments:
            if not isinstance(segment, dict):
                issues.append({"reason": "invalid_segment_record", "video_id": video_id})
                continue
            scene_id = str(segment.get("scene_id") or "")
            segment_id = str(segment.get("segment_id") or "")
            if not scene_id or not segment_id:
                issues.append({
                    "reason": "missing_scene_or_segment_id",
                    "video_id": video_id,
                })
                continue
            evidence_id = make_evidence_id(video_id, scene_id, segment_id)
            if evidence_id in index:
                issues.append({"reason": "duplicate_evidence_id", "evidence_id": evidence_id})
                continue
            if scene_id not in scenes:
                issues.append({
                    "reason": "scene_missing_from_dataset_meta",
                    "evidence_id": evidence_id,
                })
                continue
            scene_start, scene_end = scenes[scene_id]
            automatic_span = _caption_event_span(segment)
            if automatic_span is not None:
                # Caption event timestamps are on the original video timeline.
                # Clamp rounding spillover to the real scene envelope.
                automatic_start = max(scene_start, automatic_span[0])
                automatic_end = min(scene_end, automatic_span[1])
                if automatic_end < automatic_start:
                    automatic_span = None
                else:
                    automatic_span = (automatic_start, automatic_end)
            ordering_span = automatic_span or (scene_start, scene_end)
            index[evidence_id] = {
                "evidence_id": evidence_id,
                "video_id": video_id,
                "video_idx": video_idx,
                "scene_id": scene_id,
                "segment_id": segment_id,
                "video_path": str(video_path),
                "dataset_meta_path": metadata_path,
                "media_scene_range_seconds": {
                    "start": round(scene_start, 6),
                    "end": round(scene_end, 6),
                    "source": "dataset_meta.valid_scenes",
                },
                "caption_event_span_seconds": (
                    {
                        "start": round(automatic_span[0], 6),
                        "end": round(automatic_span[1], 6),
                        "source": "automatic_merged_caption_events",
                    }
                    if automatic_span is not None
                    else None
                ),
                "candidate_ordering_range_seconds": {
                    "start": round(ordering_span[0], 6),
                    "end": round(ordering_span[1], 6),
                    "source": (
                        "automatic_caption_event_span_clipped_to_media_scene"
                        if automatic_span is not None
                        else "dataset_meta_scene_envelope"
                    ),
                },
                "human_verified": False,
            }
    return index, issues


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalise_key(value: Any) -> str:
    return _normalise_text(value).casefold()


def _object_file_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"video_(\d+)_object\.json", path.name)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def _observation_for_manifest(
    observation: dict[str, Any],
    evidence: dict[str, Any],
    *,
    source_file: Path,
    source_record: int,
    source_observation: int,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence["evidence_id"],
        "location_state_text": _normalise_text(observation.get("location/state")),
        "candidate_ordering_range_seconds": evidence["candidate_ordering_range_seconds"],
        "media_scene_range_seconds": evidence["media_scene_range_seconds"],
        "caption_event_span_seconds": evidence["caption_event_span_seconds"],
        "source": {
            "object_record_path": str(source_file.resolve()),
            "object_record_index": source_record,
            "observation_index": source_observation,
            "text_provenance": "automatic_object_intermediate",
        },
        "human_verified": False,
    }


def _candidate_id(video_idx: int, video_id: str, object_name: str) -> str:
    digest = hashlib.sha1(f"{video_id}\0{object_name}".encode()).hexdigest()[:10]
    return f"hs-asuka-v{video_idx:03d}-{digest}"


def _candidate_tasks(object_name: str) -> dict[str, dict[str, Any]]:
    return {
        "C2_latest_state": {
            "intent": "Verify a cutoff flip and ask for the object's latest visible state.",
            "draft_prompt": (
                f"After reviewing the marked video interval, what is the latest visible "
                f"location or state of {object_name}?"
            ),
            "human_question": None,
            "human_answer": None,
            "human_evidence_times_seconds": None,
        },
        "C3_delayed_binding_revisit": {
            "intent": "Verify whether the same object is revisited after an intervening gap.",
            "draft_prompt": (
                f"How did the visible location or state of {object_name} change between "
                "the two marked observations?"
            ),
            "human_question": None,
            "human_answer": None,
            "human_evidence_times_seconds": None,
        },
    }


def _best_pair(
    observations: Iterable[dict[str, Any]], min_gap_seconds: float
) -> tuple[list[dict[str, Any]], float] | None:
    ordered = sorted(
        observations,
        key=lambda row: (
            row["candidate_ordering_range_seconds"]["start"],
            row["candidate_ordering_range_seconds"]["end"],
            row["evidence_id"],
        ),
    )
    choices: list[tuple[float, int, int]] = []
    for left_idx, left in enumerate(ordered):
        for right_idx in range(left_idx + 1, len(ordered)):
            right = ordered[right_idx]
            if _normalise_key(left["location_state_text"]) == _normalise_key(
                right["location_state_text"]
            ):
                continue
            gap = (
                float(right["candidate_ordering_range_seconds"]["start"])
                - float(left["candidate_ordering_range_seconds"]["end"])
            )
            if gap >= min_gap_seconds:
                choices.append((gap, left_idx, right_idx))
    if not choices:
        return None
    gap, left_idx, right_idx = max(choices, key=lambda item: (item[0], -item[1], item[2]))
    return [ordered[left_idx], ordered[right_idx]], gap


def build_homesentinel_temporal_manifest(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    merged_captions_path: Path | None = None,
    objects_dir: Path | None = None,
    min_gap_seconds: float = 10.0,
    limit: int = 0,
    video_filename: str = "raw_video.mp4",
) -> dict[str, Any]:
    """Mine human-review candidates from automatic HomeSentinel artifacts."""

    if (
        isinstance(min_gap_seconds, bool)
        or not isinstance(min_gap_seconds, (int, float))
        or not math.isfinite(min_gap_seconds)
        or min_gap_seconds < 0
    ):
        raise ValueError("min_gap_seconds must be a finite nonnegative number")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    data_root = Path(data_root).resolve()
    merged_path = Path(merged_captions_path or data_root / "merged_captions.json").resolve()
    object_root = Path(objects_dir or data_root / "benchmark_tmp").resolve()
    if not object_root.is_dir():
        raise ValueError(f"objects directory does not exist: {object_root}")
    evidence_index, index_issues = build_evidence_index(
        data_root, merged_path, video_filename=video_filename
    )

    unresolved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    object_files = sorted(object_root.glob("video_*_object.json"), key=_object_file_key)
    for object_path in object_files:
        match = re.fullmatch(r"video_(\d+)_object\.json", object_path.name)
        if match is None:
            continue
        file_video_idx = int(match.group(1))
        records = _load_json(object_path)
        if not isinstance(records, list):
            skipped.append({
                "reason": "object_file_not_a_list",
                "source_file": str(object_path),
            })
            continue

        grouped: dict[str, dict[str, Any]] = {}
        for record_idx, record in enumerate(records):
            if not isinstance(record, dict):
                skipped.append({
                    "reason": "invalid_object_record",
                    "source_file": str(object_path),
                    "record_index": record_idx,
                })
                continue
            object_name = _normalise_text(record.get("object"))
            if not object_name:
                skipped.append({
                    "reason": "missing_object_name",
                    "source_file": str(object_path),
                    "record_index": record_idx,
                })
                continue
            group = grouped.setdefault(
                _normalise_key(object_name), {"object": object_name, "observations": []}
            )
            observations = record.get("observations", [])
            if not isinstance(observations, list):
                skipped.append({
                    "reason": "observations_not_a_list",
                    "source_file": str(object_path),
                    "record_index": record_idx,
                    "object": object_name,
                })
                continue
            for observation_idx, observation in enumerate(observations):
                if not isinstance(observation, dict):
                    skipped.append({
                        "reason": "invalid_observation_record",
                        "source_file": str(object_path),
                        "record_index": record_idx,
                        "observation_index": observation_idx,
                        "object": object_name,
                    })
                    continue
                ref = _normalise_text(observation.get("timestamp"))
                state_text = _normalise_text(observation.get("location/state"))
                try:
                    observation_video_idx = int(observation.get("video_idx"))
                except (TypeError, ValueError):
                    observation_video_idx = -1
                if not ref or not state_text:
                    skipped.append({
                        "reason": "observation_missing_reference_or_state_text",
                        "source_file": str(object_path),
                        "record_index": record_idx,
                        "observation_index": observation_idx,
                        "object": object_name,
                    })
                    continue
                evidence = evidence_index.get(ref)
                if evidence is None:
                    unresolved.append({
                        "reason": "evidence_id_not_found",
                        "evidence_id": ref,
                        "source_file": str(object_path),
                        "record_index": record_idx,
                        "observation_index": observation_idx,
                        "object": object_name,
                    })
                    continue
                if (
                    observation_video_idx != file_video_idx
                    or evidence["video_idx"] != file_video_idx
                ):
                    unresolved.append({
                        "reason": "video_idx_mismatch",
                        "evidence_id": ref,
                        "source_file": str(object_path),
                        "file_video_idx": file_video_idx,
                        "observation_video_idx": observation_video_idx,
                        "evidence_video_idx": evidence["video_idx"],
                        "object": object_name,
                    })
                    continue
                group["observations"].append(
                    _observation_for_manifest(
                        observation,
                        evidence,
                        source_file=object_path,
                        source_record=record_idx,
                        source_observation=observation_idx,
                    )
                )

        for group in grouped.values():
            object_name = group["object"]
            observations = group["observations"]
            if len(observations) < 2:
                skipped.append({
                    "reason": "fewer_than_two_resolved_observations",
                    "source_file": str(object_path),
                    "video_idx": file_video_idx,
                    "object": object_name,
                    "resolved_observations": len(observations),
                })
                continue
            if len({_normalise_key(row["location_state_text"]) for row in observations}) < 2:
                skipped.append({
                    "reason": "state_text_not_distinct",
                    "source_file": str(object_path),
                    "video_idx": file_video_idx,
                    "object": object_name,
                })
                continue
            selected = _best_pair(observations, min_gap_seconds)
            if selected is None:
                skipped.append({
                    "reason": "no_time_sortable_pair_meeting_min_gap",
                    "source_file": str(object_path),
                    "video_idx": file_video_idx,
                    "object": object_name,
                    "min_gap_seconds": min_gap_seconds,
                })
                continue
            pair, gap = selected
            first_evidence = evidence_index[pair[0]["evidence_id"]]
            candidate_rows.append({
                "candidate_id": _candidate_id(
                    file_video_idx, first_evidence["video_id"], object_name
                ),
                "human_verified": False,
                "status": "candidate_pending_human_review",
                "video_id": first_evidence["video_id"],
                "video_idx": file_video_idx,
                "video_path": first_evidence["video_path"],
                "object": object_name,
                "ordered_observations": pair,
                "gap_seconds": round(gap, 6),
                "suggested_tasks": _candidate_tasks(object_name),
                "human_annotation": {
                    "approved": None,
                    "object_identity_verified": None,
                    "temporal_order_verified": None,
                    "states_verified": None,
                    "question": None,
                    "answer": None,
                    "answer_type": None,
                    "distractors": None,
                    "evidence_times_seconds": None,
                    "notes": None,
                },
            })

    candidate_rows.sort(key=lambda row: (row["video_idx"], row["object"].casefold()))
    before_limit = len(candidate_rows)
    if limit:
        candidate_rows = candidate_rows[:limit]
    referenced_ids = {
        observation["evidence_id"]
        for candidate in candidate_rows
        for observation in candidate["ordered_observations"]
    }
    referenced_evidence = {
        evidence_id: evidence_index[evidence_id] for evidence_id in sorted(referenced_ids)
    }
    return {
        "schema": SCHEMA,
        "human_verified": False,
        "label_policy": {
            "candidate_mining_only": True,
            "automatic_caption_is_gold": False,
            "automatic_object_record_is_gold": False,
            "required_next_step": (
                "A human must inspect the source video, verify identity and order, and fill "
                "the annotation fields before any candidate enters evaluation."
            ),
        },
        "sources": {
            "data_root": str(data_root),
            "merged_captions": str(merged_path),
            "objects_dir": str(object_root),
            "video_filename": video_filename,
            "caption_and_object_role": "candidate_mining_only",
        },
        "config": {
            "min_gap_seconds": min_gap_seconds,
            "limit": limit,
        },
        "summary": {
            "object_files": len(object_files),
            "evidence_index_entries_total": len(evidence_index),
            "evidence_entries_referenced_by_candidates": len(referenced_evidence),
            "candidates_before_limit": before_limit,
            "candidates_written": len(candidate_rows),
            "unresolved_references": len(unresolved),
            "skipped_records_or_objects": len(skipped),
            "evidence_index_issues": len(index_issues),
        },
        "evidence_index": {
            evidence_id: evidence_index[evidence_id] for evidence_id in sorted(evidence_index)
        },
        "candidates": candidate_rows,
        "diagnostics": {
            "unresolved_references": unresolved,
            "skipped": skipped,
            "evidence_index_issues": index_issues,
        },
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
