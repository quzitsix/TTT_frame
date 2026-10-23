"""Regression tests for the forecasting runner's split and cache contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_forecast_residual import (  # noqa: E402
    cache_signature,
    load_cache,
    main,
    read_split,
    select_ids,
)


def _split() -> dict:
    return {
        "schema": "forecast_bench_research_day_split_v1",
        "official_split": False,
        "research_only": True,
        "grouping": "recording_day_all_participants",
        "inputs_manifest_sha256": "a" * 64,
        "days": {"train": [1, 2, 3, 4], "validation": [5], "test": [6, 7]},
        "splits": {
            "train": ["egolife_001", "egolife_002"],
            "validation": ["egolife_003"],
            "test": ["egolife_004"],
        },
        "id_to_day": {
            "egolife_001": 1, "egolife_002": 4,
            "egolife_003": 5, "egolife_004": 6,
        },
    }


def _cache() -> dict:
    requested = torch.tensor([1828.0, 1828.5, 1829.0, 1829.5], dtype=torch.float64)
    return {
        "kind": "bench_v0_causal_visual_features",
        "format_version": 1,
        "sample_id": "egolife_001",
        "input_manifest_sha256": "a" * 64,
        "features": torch.zeros(4, 8, 256),
        "requested_prefix_seconds": requested,
        "actual_prefix_seconds": requested.clone(),
        "sample_fps": 2.0,
        "cropped": True,
        "complete_prefix": False,
        "encoder": "qwen3_vl_per_image",
        "audio_used": False,
        "model_fingerprint": {"sha256": "b" * 64},
        "config": {
            "max_side": 224,
            "pool_shape": [2, 4],
            "projection_seed": 0,
            "context_tail_seconds": 0,
            "limit_frames": 4,
            "batch_size": 2,
            "device": "cpu",
        },
    }


def _load_fixture(tmp_path: Path, cache: dict, *, allow_partial: bool = True):
    path = tmp_path / "features.pt"
    torch.save(cache, path)
    return load_cache(path, "egolife_001", _split(), allow_partial)


def test_read_split_and_id_selection_preserve_the_assigned_roles(tmp_path):
    split = _split()
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    restored = read_split(path)
    assert select_ids(restored, "train", None) == ["egolife_001", "egolife_002"]
    assert select_ids(restored, "train", ["egolife_002"]) == ["egolife_002"]
    for requested in (["egolife_004"], ["egolife_001", "egolife_001"]):
        with pytest.raises(ValueError, match="distinct members"):
            select_ids(restored, "train", requested)


@pytest.mark.parametrize("problem", ["same_day", "same_id", "wrong_day", "official"])
def test_read_split_rejects_leaking_or_inconsistent_assignments(tmp_path, problem):
    split = _split()
    if problem == "same_day":
        split["days"]["test"].append(1)
    elif problem == "same_id":
        split["splits"]["test"].append("egolife_001")
    elif problem == "wrong_day":
        split["id_to_day"]["egolife_004"] = 1
    else:
        split["official_split"] = True
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError):
        read_split(path)


def test_partial_cache_requires_an_explicit_opt_in(tmp_path):
    cache = _cache()
    with pytest.raises(ValueError, match="partial cache requires"):
        _load_fixture(tmp_path, cache, allow_partial=False)
    loaded = _load_fixture(tmp_path, cache)
    assert torch.equal(loaded["features"], cache["features"])


def test_a_partial_cache_cannot_claim_to_cover_the_complete_prefix(tmp_path):
    cache = _cache()
    cache.update(cropped=False, complete_prefix=True)
    with pytest.raises(ValueError, match="entire 1830-second past"):
        _load_fixture(tmp_path, cache, allow_partial=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_id", "egolife_004"),
        ("input_manifest_sha256", "c" * 64),
        ("format_version", 2),
        ("kind", "future_visual_features"),
    ],
)
def test_cache_identity_is_bound_to_its_sample_and_input_manifest(tmp_path, field, value):
    cache = _cache()
    cache[field] = value
    with pytest.raises(ValueError, match="identity or source manifest mismatch"):
        _load_fixture(tmp_path, cache)


@pytest.mark.parametrize(
    "problem",
    ["integer_grid", "lookahead", "reverse_actual", "stale_frame", "irregular_grid",
     "post_cutoff", "nan_time", "nan_feature"],
)
def test_cache_rejects_noncausal_or_invalid_temporal_evidence(tmp_path, problem):
    cache = _cache()
    if problem == "integer_grid":
        # int full_like(..., 0.5) truncates to zero; this once accepted a stuck grid.
        cache["requested_prefix_seconds"] = torch.zeros(4, dtype=torch.int64)
        cache["actual_prefix_seconds"] = torch.zeros(4, dtype=torch.int64)
    elif problem == "lookahead":
        cache["actual_prefix_seconds"][1] += 0.01
    elif problem == "reverse_actual":
        cache["actual_prefix_seconds"][2] = 1827.9
    elif problem == "stale_frame":
        cache["actual_prefix_seconds"] -= 2.0
    elif problem == "irregular_grid":
        cache["requested_prefix_seconds"][1] += 0.1
    elif problem == "post_cutoff":
        cache["requested_prefix_seconds"] += 0.5
        cache["actual_prefix_seconds"] += 0.5
    elif problem == "nan_time":
        cache["actual_prefix_seconds"][1] = float("nan")
    else:
        cache["features"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="noncausal/irregular time grid"):
        _load_fixture(tmp_path, cache)


@pytest.mark.parametrize(
    ("field", "value"), [("encoder", "full_video_transformer"), ("audio_used", True)]
)
def test_cache_requires_independent_images_without_audio(tmp_path, field, value):
    cache = _cache()
    cache[field] = value
    with pytest.raises(ValueError, match="independent RGB image features"):
        _load_fixture(tmp_path, cache)


def test_cache_signature_ignores_runtime_options_but_preserves_representation():
    cache = _cache()
    runtime_changed = deepcopy(cache)
    runtime_changed["config"].update(
        context_tail_seconds=10, limit_frames=20, batch_size=8, device="cuda:2"
    )
    assert cache_signature(cache) == cache_signature(runtime_changed)
    for field, value in (("max_side", 448), ("projection_seed", 7), ("pool_shape", [1, 8])):
        changed = deepcopy(cache)
        changed["config"][field] = value
        assert cache_signature(cache) != cache_signature(changed)
    changed = deepcopy(cache)
    changed["model_fingerprint"] = {"sha256": "d" * 64}
    assert cache_signature(cache) != cache_signature(changed)


def test_cli_fits_only_train_cache_and_writes_finite_prequential_report(tmp_path):
    split = _split()
    split["root"] = str(tmp_path / "source_benchmark")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    cache_dir = tmp_path / "features"
    cache_dir.mkdir()
    generator = torch.Generator().manual_seed(101)
    for sample_id in ("egolife_001", "egolife_003"):
        cache = _cache()
        times = torch.arange(1820, 1830, 0.5, dtype=torch.float64)
        cache.update(sample_id=sample_id, features=torch.randn(20, 8, 256, generator=generator),
                     requested_prefix_seconds=times, actual_prefix_seconds=times.clone())
        torch.save(cache, cache_dir / f"{sample_id}.pt")
    output = tmp_path / "result"
    assert main([
        "--cache-dir", str(cache_dir), "--split-manifest", str(split_path),
        "--output-dir", str(output), "--train-ids", "egolife_001",
        "--eval-ids", "egolife_003", "--allow-partial-prefix",
        "--score-last-seconds", "5", "--threads", "1",
    ]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["train_ids"] == ["egolife_001"]
    assert report["eval_ids"] == ["egolife_003"]
    assert report["future_media_used"] is False and report["is_mcq_accuracy"] is False
    assert all(row["count"] == 23 for row in report["aggregate"].values())
    assert report["cases"][0]["last_prediction_target_seconds_after_cutoff"] == [0, .5, 1.5, 4.5]
    state = torch.load(output / "recent_predictor.pt", weights_only=True)
    assert state["train_ids"] == ["egolife_001"]
    forecasts = torch.load(output / "egolife_003_forecasts.pt", weights_only=True)
    assert len(forecasts) == 6
    assert all(value.shape == (4, 8, 256) for value in forecasts.values())
