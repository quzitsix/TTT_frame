"""Shared past-only sampling and answer contracts for bench_v0 MCQ baselines.

This module never opens benchmark answers, private metadata, or future media.
The default sparse schedule covers the whole available history, but it does
not encode every source frame and is independent of the annotation schedule.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence, TypeVar

from ttt_frame.forecast_bench import ForecastCase, ForecastOption, FrameSample, TASK
from ttt_frame.forecast_features import causal_video_frames


ARMS = (
    "qwen_blind", "qwen_recent_icl", "qwen_full_icl",
    "blind", "recent_icl", "full_icl", "spatial", "spatial_slow_only", "spatial_base",
)
QWEN_ARMS = ARMS[:3]
OFFICIAL_ARMS = ARMS[3:]
_Value = TypeVar("_Value")


def _validate_case(case: ForecastCase) -> None:
    if case.task != TASK or (
        case.context_seconds, case.recent_seconds, case.horizon_seconds
    ) != (1800, 30, 5):
        raise ValueError("expected the bench_v0 1800s + 30s -> 5s protocol")
    if not re.fullmatch(r"egolife_\d{3}", case.id):
        raise ValueError("invalid bench_v0 sample ID")
    context, recent = Path(case.context_path), Path(case.recent_path)
    if (
        context.name != "context.mp4" or recent.name != "input.mp4"
        or context.parent != recent.parent or context.parent.name != case.id
        or context.parent.parent.name != "samples"
        or context.resolve() != context.absolute()
        or recent.resolve() != recent.absolute()
    ):
        raise ValueError("media must be this sample's unredirected context.mp4 and input.mp4")


def _frame_count(value: int, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 4 or value % 2:
        raise ValueError(f"{name} must be an even integer of at least four")
    if value > maximum:
        raise ValueError(f"{name} exceeds the number of source frames")


def make_frame_plan(
    case: ForecastCase, *, context_frames: int = 30, recent_frames: int = 60,
) -> tuple[FrameSample, ...]:
    """Sample each past clip from its start on an endpoint-excluding grid.

    Defaults: context 0,60,...,1740; recent 0,.5,...,29.5. Each clip has an
    even frame count so its temporal tubelets never need to cross the boundary.
    Prefix timestamps are relative to context start; cutoff is 1830 seconds.
    """
    _validate_case(case)
    _frame_count(context_frames, "context_frames", 3600)
    _frame_count(recent_frames, "recent_frames", 600)
    result = []
    for role, path, duration, offset, count in (
        ("context", case.context_path, case.context_seconds, 0.0, context_frames),
        ("recent", case.recent_path, case.recent_seconds, case.context_seconds, recent_frames),
    ):
        for index in range(count):
            local_time = duration * index / count
            result.append(FrameSample(
                stream_step=len(result), prefix_seconds=offset + local_time,
                video_seconds=local_time, role=role, path=Path(path),
            ))
    return tuple(result)


def frame_bundle_hash(
    images: Sequence[Any], actual_seconds: Sequence[float],
    requested_seconds: Sequence[float],
) -> str:
    """Hash decoded RGB pixels and both timestamp lists, not compressed files."""
    if not (len(images) == len(actual_seconds) == len(requested_seconds)):
        raise ValueError("frame images and timestamps do not match")
    digest = hashlib.sha256(b"ttt_frame.forecast_mcq.frame_bundle/1\0")
    for image, actual, requested in zip(images, actual_seconds, requested_seconds):
        if not math.isfinite(actual) or not math.isfinite(requested) or actual > requested + 1e-9:
            raise ValueError("invalid or post-request frame timestamp")
        rgb = image.convert("RGB")
        digest.update(
            f"{float(actual):.12f}\0{float(requested):.12f}\0{rgb.width}x{rgb.height}\0RGB\0".encode()
        )
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def _media_stats(case: ForecastCase) -> dict:
    result = {}
    for role, path in (("context", case.context_path), ("recent", case.recent_path)):
        stat = Path(path).stat()
        result[role] = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def load_frame_bundle(
    case: ForecastCase, *, context_frames: int = 30, recent_frames: int = 60,
    max_side: int = 224, decode: bool = True,
) -> dict:
    """Decode the same causal evidence once for reuse by every architecture.

    ``*_requested_seconds`` and ``*_actual_seconds`` both use prefix time.
    ``decode=False`` returns a plan and source file stats without decoding any
    media; actual times and pixel hashes are then None. ``source_media`` records
    size/mtime, not a source content hash. Only ``*_images`` are non-JSON values.
    """
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 64 or max_side % 32:
        raise ValueError("max_side must be a multiple of 32 and at least 64")
    plan = make_frame_plan(case, context_frames=context_frames, recent_frames=recent_frames)
    source_media = _media_stats(case)
    plan_payload = {
        "sample_id": case.id, "max_side": max_side,
        "frames": [{"role": sample.role, "requested_prefix_seconds": sample.prefix_seconds,
                    "requested_video_seconds": sample.video_seconds} for sample in plan],
    }
    result = {
        "schema": "bench_v0_mcq_past_frame_bundle_v1", "sample_id": case.id,
        "decoded": bool(decode), "audio_used": False, "source_media": source_media,
        "sampling": {"context_frames": context_frames, "recent_frames": recent_frames,
                     "max_side": max_side, "grid": "clip_start_inclusive_end_exclusive",
                     "cutoff_prefix_seconds": case.prefix_seconds,
                     "annotation_sampling_reused": False},
        "frame_plan_sha256": hashlib.sha256(json.dumps(
            plan_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        "frame_bundle_sha256": None,
    }
    for role in ("context", "recent"):
        samples = [sample for sample in plan if sample.role == role]
        requested = [sample.prefix_seconds for sample in samples]
        result[f"{role}_requested_seconds"] = requested
        result[f"{role}_images"] = []
        result[f"{role}_actual_seconds"] = None
        result[f"{role}_frame_bundle_sha256"] = None
        if not decode:
            continue
        offset = 0.0 if role == "context" else case.context_seconds
        decoded = list(causal_video_frames(
            samples[0].path, [sample.video_seconds for sample in samples], max_side=max_side,
        ))
        if len(decoded) != len(samples):
            raise ValueError(f"decoder returned the wrong number of {role} frames")
        images = [image for image, _ in decoded]
        actual = [offset + timestamp for _, timestamp in decoded]
        if any(
            not math.isfinite(t) or t < offset or t > requested[i] + 1e-9
            or (i > 0 and t < actual[i - 1]) for i, t in enumerate(actual)
        ):
            raise ValueError(f"noncausal {role} frame timestamps")
        result[f"{role}_images"] = images
        result[f"{role}_actual_seconds"] = actual
        result[f"{role}_frame_bundle_sha256"] = frame_bundle_hash(images, actual, requested)
    if decode:
        result["frame_bundle_sha256"] = frame_bundle_hash(
            result["context_images"] + result["recent_images"],
            result["context_actual_seconds"] + result["recent_actual_seconds"],
            result["context_requested_seconds"] + result["recent_requested_seconds"],
        )
        if _media_stats(case) != source_media:
            raise ValueError("past media changed during decoding")
    return result


def build_mcq_prompt(case: ForecastCase) -> str:
    """Use the identical question and choice order for every baseline."""
    _validate_case(case)
    if len(case.options) != 6 or {option.label for option in case.options} != set("ABCDEF"):
        raise ValueError("expected six unique A-F options")
    if not case.question.strip() or any(not option.text.strip() for option in case.options):
        raise ValueError("question and option text must be nonempty")
    options = "\n".join(f"{option.label}. {option.text}" for option in case.options)
    return (
        "这是第一人称视频动作预测任务。以历史视频开始为 0 秒，预测截止点为 1830 秒。\n"
        "历史区间为 [0, 1800) 秒，最近区间为 [1800, 1830) 秒；需要预测 [1830, 1835) 秒内的一个具体动作。\n"
        "只能依据实际提供的截止前信息、问题和选项作答。视频帧按时间先后排列，不提供截止后的画面。\n"
        f"问题：{case.question}\n选项：\n{options}\n"
        "只输出一个选项字母 A、B、C、D、E 或 F，不要解释，不要输出其他文字。"
    )


def parse_mcq_prediction(response: object, options: Sequence[ForecastOption]) -> str | None:
    """Accept a single A-F letter or exact choice text, never search a rationale."""
    if not isinstance(response, str):
        return None
    candidate = response.strip()
    if re.fullmatch(r"[A-Fa-f]", candidate):
        return candidate.upper()
    matches = [option.label for option in options if candidate == option.text.strip()]
    return matches[0] if len(matches) == 1 else None


def write_chunks(values: Sequence[_Value], size: int) -> list[list[_Value]]:
    """Keep whole temporal pairs and merge one/two-frame write remainders.

    Call separately for context and recent, preserving their boundary. The
    same real frames enter ICL and Spatial without small-tail video padding.
    """
    _frame_count(size, "write chunk size", 4200)
    if len(values) < 4 or len(values) % 2:
        raise ValueError("write streams must contain an even number of at least four frames")
    result, start = [], 0
    while len(values) - start > size + 2:
        result.append(list(values[start:start + size]))
        start += size
    result.append(list(values[start:]))
    return result
