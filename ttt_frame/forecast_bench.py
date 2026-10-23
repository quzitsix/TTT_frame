"""Strict past-only inputs and separate audits for EgoLife forecasting bench_v0.

Inference reads only ``inputs_mcq.jsonl``.  Audit/split helpers read private
metadata independently and never attach it to a :class:`ForecastCase`.
The fixed two-FPS sampling plan presents identical frames to every method.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


TASK = "egocentric_action_forecasting_5s"
SAMPLE_FPS = 2
_INPUT_FIELDS = {
    "context_video", "video", "question", "context_duration_seconds",
    "input_duration_seconds", "prediction_horizon_seconds", "options",
}


@dataclass(frozen=True)
class ForecastOption:
    label: str
    text: str


@dataclass(frozen=True)
class ForecastCase:
    id: str
    task: str
    context_path: Path
    recent_path: Path
    question: str
    options: tuple[ForecastOption, ...]
    context_seconds: float = 1800.0
    recent_seconds: float = 30.0
    horizon_seconds: float = 5.0

    @property
    def prefix_seconds(self) -> float:
        return self.context_seconds + self.recent_seconds


@dataclass(frozen=True)
class FrameSample:
    stream_step: int
    prefix_seconds: float
    video_seconds: float
    role: str
    path: Path


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _exact_fields(value: object, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly the public fields {sorted(fields)}")
    return value


def _duration(value: object, expected: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or float(value) != expected
    ):
        raise ValueError(f"{name} must equal {expected}")
    return float(value)


def _past_path(root: Path, case_id: str, value: object, filename: str) -> Path:
    expected = f"samples/{case_id}/{filename}"
    if value != expected:
        raise ValueError(f"past media path must be exactly {expected}")
    path = (root / expected).resolve()
    if not path.is_relative_to(root):
        raise ValueError("past media path escapes benchmark root")
    if path != root / expected:
        raise ValueError("past media path may not redirect to another file")
    if not path.is_file():
        raise ValueError(f"missing past media: {path}")
    return path


def load_forecast_cases(
    root: str | Path,
    *,
    ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[ForecastCase]:
    """Load only public MCQ inputs, rejecting hidden fields and media escapes.

    The whole input manifest is validated even when a subset is requested.
    Neither answers, annotation files nor metadata are opened by this function.
    """

    root = Path(root).resolve()
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    if isinstance(ids, str):
        raise ValueError("ids must be a collection of case IDs, not a single string")
    selected = None if ids is None else set(ids)
    cases: list[ForecastCase] = []
    seen: set[str] = set()
    for row in _jsonl(root / "inputs_mcq.jsonl"):
        _exact_fields(row, {"id", "task", "input"}, "case")
        case_id = row["id"]
        if not isinstance(case_id, str) or not re.fullmatch(r"egolife_\d{3}", case_id):
            raise ValueError("case id must follow egolife_NNN")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        if row["task"] != TASK:
            raise ValueError(f"unsupported forecasting task: {row['task']}")
        inputs = _exact_fields(row["input"], _INPUT_FIELDS, "input")
        question = inputs["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        options = inputs["options"]
        if not isinstance(options, list) or len(options) != 6:
            raise ValueError("options must contain exactly six public choices")
        parsed_options = []
        for option in options:
            _exact_fields(option, {"label", "text"}, "option")
            if not isinstance(option["text"], str) or not option["text"].strip():
                raise ValueError("option text must be nonempty")
            if not isinstance(option["label"], str):
                raise ValueError("option label must be a string")
            parsed_options.append(ForecastOption(option["label"], option["text"]))
        if {option.label for option in parsed_options} != set("ABCDEF"):
            raise ValueError("option labels must be unique A through F")
        if len({option.text.strip() for option in parsed_options}) != 6:
            raise ValueError("option texts must be distinct")
        cases.append(ForecastCase(
            id=case_id,
            task=row["task"],
            context_path=_past_path(root, case_id, inputs["context_video"], "context.mp4"),
            recent_path=_past_path(root, case_id, inputs["video"], "input.mp4"),
            question=question,
            options=tuple(parsed_options),
            context_seconds=_duration(inputs["context_duration_seconds"], 1800, "context"),
            recent_seconds=_duration(inputs["input_duration_seconds"], 30, "recent"),
            horizon_seconds=_duration(inputs["prediction_horizon_seconds"], 5, "horizon"),
        ))
    if selected is not None:
        missing = selected - seen
        if missing:
            raise ValueError(f"requested case ids not present: {sorted(missing)}")
        cases = [case for case in cases if case.id in selected]
    return cases if limit is None else cases[:limit]


def make_sampling_plan(case: ForecastCase) -> tuple[FrameSample, ...]:
    """Return the two-FPS grid [0, 1830), with no post-cutoff sample.

    Context covers steps 0..3599.  Recent input covers steps 3600..3659.
    Times are requested media presentation times, not claims about the actual
    decoded PTS.  A decoder must choose a frame no later than each request.
    """

    if (case.context_seconds, case.recent_seconds, case.horizon_seconds) != (1800, 30, 5):
        raise ValueError("sampling requires the 1800s + 30s -> 5s protocol")
    context_steps = int(case.context_seconds * SAMPLE_FPS)
    total_steps = int(case.prefix_seconds * SAMPLE_FPS)
    return tuple(
        FrameSample(
            stream_step=step,
            prefix_seconds=step / SAMPLE_FPS,
            video_seconds=(step if step < context_steps else step - context_steps) / SAMPLE_FPS,
            role="context" if step < context_steps else "recent",
            path=case.context_path if step < context_steps else case.recent_path,
        )
        for step in range(total_steps)
    )


def _audit_metadata(root: Path, cases: list[ForecastCase]) -> list[dict]:
    rows = _jsonl(root / "metadata.jsonl")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != {case.id for case in cases}:
        raise ValueError("metadata ids do not match public input ids")
    for row in rows:
        meta = row["metadata"]
        start, boundary, cutoff, end = (
            meta["context_start_seconds"], meta["input_start_seconds"],
            meta["cutoff_seconds"], meta["future_end_seconds"],
        )
        if (
            not all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (start, boundary, cutoff, end))
            or boundary - start != 1800
            or cutoff - boundary != 30
            or end - cutoff != 5
            or meta["context_end_seconds"] != boundary
        ):
            raise ValueError(f"invalid private source timeline for {row['id']}")
        if type(meta["day"]) is not int or not 1 <= meta["day"] <= 7:
            raise ValueError(f"invalid recording day for {row['id']}")
    return rows


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _overlap_counts(rows: list[dict], *, same_participant: bool) -> dict:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs = 0
    for (i, left), (j, right) in combinations(enumerate(rows), 2):
        a, b = left["metadata"], right["metadata"]
        if a["day"] != b["day"]:
            continue
        if same_participant and a["participant"] != b["participant"]:
            continue
        if _overlap(
            (a["context_start_seconds"], a["future_end_seconds"]),
            (b["context_start_seconds"], b["future_end_seconds"]),
        ):
            pairs += 1
            parent[find(j)] = find(i)
    components = Counter(find(index) for index in range(len(rows)))
    return {
        "overlapping_pairs": pairs,
        "connected_components": len(components),
        "component_sizes": sorted(components.values(), reverse=True),
    }


def audit_forecast_bench(root: str | Path) -> dict:
    """Audit private label quality and source grouping; never an inference input."""

    root = Path(root).resolve()
    cases = load_forecast_cases(root)
    rows = _audit_metadata(root, cases)
    same_person_recent_overlap = 0
    future_in_other_past = 0
    for left, right in combinations(rows, 2):
        a, b = left["metadata"], right["metadata"]
        if (a["participant"], a["day"]) != (b["participant"], b["day"]):
            continue
        same_person_recent_overlap += int(_overlap(
            (a["input_start_seconds"], a["future_end_seconds"]),
            (b["input_start_seconds"], b["future_end_seconds"]),
        ))
        for earlier, later in ((a, b), (b, a)):
            future_in_other_past += int(_overlap(
                (earlier["cutoff_seconds"], earlier["future_end_seconds"]),
                (later["context_start_seconds"], later["cutoff_seconds"]),
            ))
    verification = [row["verification"] for row in rows]
    return {
        "schema": "forecast_bench_audit_v1",
        "audit_only": True,
        "root": str(root),
        "sample_count": len(rows),
        "task_counts": dict(Counter(case.task for case in cases)),
        "participants": dict(Counter(row["metadata"]["participant"] for row in rows)),
        "days": dict(sorted(Counter(row["metadata"]["day"] for row in rows).items())),
        "person_days": len({(row["metadata"]["participant"], row["metadata"]["day"])
                            for row in rows}),
        "release_splits": dict(Counter(row["metadata"]["split"] for row in rows)),
        "categories": dict(Counter(row["output"]["answer_category"] for row in rows)),
        "gt_consistency": dict(Counter(row["gt_consistency"] for row in verification)),
        "uncertain_gt": sum(bool(row["gt_uncertain"]) for row in verification),
        "human_verified": sum(bool(row["human_verified"]) for row in verification),
        "plausible_distractor_counts": dict(Counter(
            row["output"]["option_type_counts"]["plausible_not_observed"] for row in rows
        )),
        "correct_position_counts": dict(Counter(row["output"]["correct_answer"] for row in rows)),
        "target_intervals": dict(Counter(
            str(row["metadata"]["target_interval_in_future"]) for row in rows
        )),
        "sampling": {
            "fps": SAMPLE_FPS, "context_frames": 3600, "recent_frames": 60,
            "first_prefix_seconds": 0.0, "last_prefix_seconds": 1829.5,
            "cutoff_prefix_seconds": 1830.0, "future_input_frames": 0,
        },
        "overlap": {
            "same_participant_day": _overlap_counts(rows, same_participant=True),
            "day_across_participants": _overlap_counts(rows, same_participant=False),
            "same_person_recent_and_future_pairs": same_person_recent_overlap,
            "same_person_future_in_another_past_pairs": future_in_other_past,
        },
        "interpretation": [
            "Silver labels from automatic screening; human verification is recorded separately.",
            "All released cases form one benchmark split; no official training split exists.",
            "Overlapping pasts and shared scenes require day grouping across all participants.",
            "A source action is one reference event, not an exhaustive future event list.",
        ],
    }


def make_research_day_split(root: str | Path) -> dict:
    """Opt-in nonofficial split: days 1–4 / 5 / 6–7 across every participant."""

    root = Path(root).resolve()
    rows = _audit_metadata(root, load_forecast_cases(root))
    assignment = {1: "train", 2: "train", 3: "train", 4: "train",
                  5: "validation", 6: "test", 7: "test"}
    groups: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        groups[assignment[row["metadata"]["day"]]].append(row["id"])
    return {
        "schema": "forecast_bench_research_day_split_v1",
        "official_split": False,
        "research_only": True,
        "grouping": "recording_day_all_participants",
        "description": (
            "Research split of an evaluation-only release; results are not the official "
            "200-case benchmark score. Recording day is held out across all participants."
        ),
        "root": str(root),
        "inputs_manifest_sha256": hashlib.sha256(
            (root / "inputs_mcq.jsonl").read_bytes()
        ).hexdigest(),
        "metadata_manifest_sha256": hashlib.sha256(
            (root / "metadata.jsonl").read_bytes()
        ).hexdigest(),
        "days": {"train": [1, 2, 3, 4], "validation": [5], "test": [6, 7]},
        "splits": groups,
        "id_to_day": {row["id"]: row["metadata"]["day"] for row in rows},
        "counts": {name: len(ids) for name, ids in groups.items()},
        "training_policy": (
            "Fit normalization, recent predictor, projectors and readout on train only; "
            "choose hyperparameters on validation; reset online state for each test case. "
            "Test updates may use observed past frames only, never test future or answers."
        ),
    }
