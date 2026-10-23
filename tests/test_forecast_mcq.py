"""Causal evidence, temporal boundaries, and strict MCQ answer contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from ttt_frame.forecast_bench import ForecastCase, ForecastOption, TASK
from ttt_frame.forecast_mcq import (
    ARMS, OFFICIAL_ARMS, QWEN_ARMS, build_mcq_prompt, frame_bundle_hash,
    load_frame_bundle, make_frame_plan, parse_mcq_prediction, write_chunks,
)


def _case(root: Path) -> ForecastCase:
    sample = root / "samples" / "egolife_001"
    sample.mkdir(parents=True)
    for name in ("context.mp4", "input.mp4"):
        (sample / name).write_bytes(b"stub decoder input")
    return ForecastCase(
        id="egolife_001", task=TASK, context_path=sample / "context.mp4",
        recent_path=sample / "input.mp4", question="接下来会做什么？",
        options=tuple(ForecastOption(label, f"动作 {label}") for label in "ABCDEF"),
    )


def test_default_plan_and_checkpoint_families(tmp_path):
    case = _case(tmp_path)
    plan = make_frame_plan(case)
    assert len(plan) == 90
    assert [s.prefix_seconds for s in plan[:30]] == list(range(0, 1800, 60))
    assert [s.video_seconds for s in plan[30:]] == [i * .5 for i in range(60)]
    assert plan[30].prefix_seconds == 1800 and plan[-1].prefix_seconds == 1829.5
    assert {s.path.name for s in plan} == {"context.mp4", "input.mp4"}
    assert len(ARMS) == len(set(ARMS)) == 9
    assert len(QWEN_ARMS) == 3 and len(OFFICIAL_ARMS) == 6


@pytest.mark.parametrize("counts", [
    {"context_frames": 3}, {"recent_frames": 2}, {"recent_frames": 5},
    {"context_frames": True}, {"recent_frames": 60.0}, {"context_frames": 3602},
    {"recent_frames": 602},
])
def test_frame_counts_reject_temporal_padding_and_oversampling(tmp_path, counts):
    with pytest.raises(ValueError):
        make_frame_plan(_case(tmp_path), **counts)


def test_future_or_redirected_media_are_rejected(tmp_path):
    case = _case(tmp_path)
    future = case.recent_path.with_name("future.mp4")
    future.write_bytes(b"private")
    with pytest.raises(ValueError, match="unredirected"):
        make_frame_plan(replace(case, recent_path=future))
    case.recent_path.unlink()
    case.recent_path.symlink_to(future)
    with pytest.raises(ValueError, match="unredirected"):
        load_frame_bundle(case, decode=False)


def test_dry_plan_does_not_decode_or_open_private_metadata(tmp_path, monkeypatch):
    import ttt_frame.forecast_mcq as module

    case = _case(tmp_path)
    (tmp_path / "metadata.jsonl").write_text("INVALID PRIVATE DATA")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry plan must not decode")

    monkeypatch.setattr(module, "causal_video_frames", forbidden)
    bundle = load_frame_bundle(case, decode=False)
    assert not bundle["decoded"] and not bundle["audio_used"]
    assert bundle["context_images"] == [] and bundle["recent_images"] == []
    assert bundle["frame_bundle_sha256"] is None
    assert bundle["recent_actual_seconds"] is None
    assert len(bundle["frame_plan_sha256"]) == 64
    assert bundle["source_media"]["recent"]["size_bytes"] == len(b"stub decoder input")


def test_decodes_separate_clips_preserving_actual_prefix_pts(tmp_path, monkeypatch):
    import ttt_frame.forecast_mcq as module

    case, calls = _case(tmp_path), []

    def decoder(path, requested_seconds, *, max_side):
        calls.append((path.name, requested_seconds))
        for requested in requested_seconds:
            # Real decoder's preceding-PTS behavior is covered independently.
            yield Image.new("RGB", (64, 64)), max(0, requested - .025)

    monkeypatch.setattr(module, "causal_video_frames", decoder)
    bundle = load_frame_bundle(case, context_frames=4, recent_frames=4, max_side=64)
    assert [name for name, _ in calls] == ["context.mp4", "input.mp4"]
    assert calls[1][1] == [0, 7.5, 15, 22.5]
    assert bundle["recent_actual_seconds"] == [1800, 1807.475, 1814.975, 1822.475]
    assert len(bundle["frame_bundle_sha256"]) == 64
    assert bundle["context_frame_bundle_sha256"] != bundle["recent_frame_bundle_sha256"]
    assert bundle["frame_bundle_sha256"] == load_frame_bundle(
        case, context_frames=4, recent_frames=4, max_side=64,
    )["frame_bundle_sha256"]


def test_future_pts_and_missing_frames_fail_before_model_input(tmp_path, monkeypatch):
    import ttt_frame.forecast_mcq as module

    case = _case(tmp_path)

    def bad_pts(path, requested_seconds, *, max_side):
        for requested in requested_seconds:
            yield Image.new("RGB", (64, 64)), requested + .05

    monkeypatch.setattr(module, "causal_video_frames", bad_pts)
    with pytest.raises(ValueError, match="noncausal"):
        load_frame_bundle(case, context_frames=4, recent_frames=4)
    monkeypatch.setattr(module, "causal_video_frames", lambda *a, **kw: iter(()))
    with pytest.raises(ValueError, match="wrong number"):
        load_frame_bundle(case, context_frames=4, recent_frames=4)


def test_bundle_hash_detects_pixel_and_time_changes():
    image = Image.new("RGB", (64, 64))
    original = frame_bundle_hash([image], [0], [.25])
    assert original != frame_bundle_hash([image], [0], [.3])
    assert original != frame_bundle_hash([image], [.1], [.25])
    assert original != frame_bundle_hash([Image.new("RGB", (64, 64), "red")], [0], [.25])
    with pytest.raises(ValueError):
        frame_bundle_hash([image], [.5], [.25])


def test_prompt_has_cutoff_and_public_choices_but_no_answer(tmp_path):
    case = _case(tmp_path)
    prompt = build_mcq_prompt(case)
    assert "[1830, 1835)" in prompt and case.question in prompt
    assert all(f"{o.label}. {o.text}" in prompt for o in case.options)
    assert "正确答案" not in prompt and "future.mp4" not in prompt


@pytest.mark.parametrize("response,expected", [
    ("A", "A"), (" f ", "F"), ("动作 C", "C"), ("动作", None),
    ("答案是 A", None), ("A or B", None), ("A.", None), (None, None),
])
def test_strict_parse_does_not_extract_letters_from_explanations(tmp_path, response, expected):
    assert parse_mcq_prediction(response, _case(tmp_path).options) == expected


@pytest.mark.parametrize("count", [4, 6, 8, 10, 12, 14, 30, 60, 90])
def test_write_chunks_preserve_all_frames_without_short_tails(count):
    values = list(range(count))
    chunks = write_chunks(values, 8)
    assert [value for chunk in chunks for value in chunk] == values
    assert all(len(chunk) >= 4 and len(chunk) % 2 == 0 for chunk in chunks)
    assert max(map(len, chunks)) <= 10


@pytest.mark.parametrize("count,size", [(2, 8), (9, 8), (8, 3), (8, 2)])
def test_invalid_write_stream_rejected(count, size):
    with pytest.raises(ValueError):
        write_chunks(list(range(count)), size)
