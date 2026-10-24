"""Full-prefix cache isolation, sharding and strict resumability checks."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from scripts.cache_idea1_features import (
    atomic_save, build_protocol, canonical_sha256, exclusive_directory_lock,
    feature_sha256, install_protocol, media_snapshot, select_shard, validate_existing,
)
from ttt_frame.forecast_bench import ForecastCase
from ttt_frame.forecast_features import FeatureConfig


def make_case(root, number=1):
    directory = root / f"egolife_{number:03d}"
    directory.mkdir(exist_ok=True)
    context, recent = directory / "context.mp4", directory / "input.mp4"
    context.write_bytes(b"observed context")
    recent.write_bytes(b"observed recent input")
    return ForecastCase(
        id=directory.name, task="egocentric_action_forecasting_5s",
        context_path=context, recent_path=recent, question="not encoded", options=(),
    )


@pytest.fixture
def cached(tmp_path):
    case = make_case(tmp_path)
    config = FeatureConfig(max_side=224)
    protocol = build_protocol(config, {"sha256": "model"}, "public_manifest", 64)
    sources = media_snapshot(case)
    times = torch.arange(3660, dtype=torch.float64) / 2
    features = torch.zeros(3660, 8, 256)
    cache = {
        "kind": "bench_v0_causal_visual_features", "format_version": 1,
        "sample_id": case.id, "complete_prefix": True, "cropped": False,
        "protocol_sha256": canonical_sha256(protocol),
        "input_manifest_sha256": protocol["input_manifest_sha256"],
        "model_fingerprint": protocol["model_fingerprint"],
        "source_media": sources, "audio_used": False,
        "execution": protocol["execution"], "config": asdict(config),
        "features": features, "requested_prefix_seconds": times,
        "actual_prefix_seconds": times.clone(), "feature_sha256": feature_sha256(features),
    }
    return case, protocol, sources, cache


def validate(data):
    case, protocol, sources, cache = data
    validate_existing(cache, case, protocol_sha=canonical_sha256(protocol),
                      protocol=protocol, sources=sources)


def test_shards_are_disjoint_and_cover_public_case_order():
    cases = list(range(200))
    shards = [select_shard(cases, i, 4) for i in range(4)]
    assert all(len(shard) == 50 for shard in shards)
    assert sorted(value for shard in shards for value in shard) == cases
    assert shards[1] == cases[1::4]
    for index, count in ((-1, 4), (4, 4), (0, 0), (False, 4)):
        with pytest.raises(ValueError, match="shard index"):
            select_shard(cases, index, count)


def test_protocol_rejects_non_full_or_non_two_fps_features():
    for config in (FeatureConfig(context_tail_seconds=60), FeatureConfig(limit_frames=8),
                   FeatureConfig(sample_fps=1)):
        with pytest.raises(ValueError):
            build_protocol(config, {}, "manifest", 64)


def test_protocol_install_is_repeatable_and_rejects_config_source_or_model_drift(tmp_path):
    protocol = build_protocol(FeatureConfig(max_side=224), {"sha256": "one"}, "manifest", 64)
    assert install_protocol(tmp_path, protocol) == install_protocol(tmp_path, protocol)
    variants = []
    for section, key, value in (("feature_config", "max_side", 256),
                                ("model_fingerprint", "sha256", "two"),
                                ("source_sha256", "ttt_frame/forecast_features.py", "changed"),
                                ("execution", "batch_size", 32)):
        altered = deepcopy(protocol)
        altered[section][key] = value
        variants.append(altered)
    for altered in variants:
        with pytest.raises(ValueError, match="conflicting cache protocol"):
            install_protocol(tmp_path, altered)
    saved = json.loads((tmp_path / "cache_protocol.json").read_text())
    assert saved["protocol_sha256"] == canonical_sha256(protocol)


def test_only_public_media_hashed_and_content_change_detected(tmp_path):
    case = make_case(tmp_path)
    before = media_snapshot(case)
    assert set(before) == {"context", "recent"}
    assert all(set(row) == {"path", "size_bytes", "mtime_ns", "sha256"}
               for row in before.values())
    case.recent_path.write_bytes(b"different observed content")
    after = media_snapshot(case)
    assert before["context"] == after["context"]
    assert before["recent"]["sha256"] != after["recent"]["sha256"]


def test_complete_cache_roundtrip_is_valid(cached, tmp_path):
    validate(cached)
    destination = tmp_path / "cache.pt"
    atomic_save(cached[-1], destination)
    reloaded = torch.load(destination, weights_only=True)
    validate((*cached[:3], reloaded))
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("field,value", [
    ("sample_id", "egolife_002"), ("complete_prefix", False),
    ("protocol_sha256", "stale"), ("input_manifest_sha256", "changed"),
    ("source_media", {}), ("execution", {}), ("config", {}),
    ("model_fingerprint", {"sha256": "changed"}), ("feature_sha256", "corrupted"),
])
def test_resume_rejects_conflicting_or_incomplete_cache(cached, field, value):
    cached[-1][field] = value
    with pytest.raises(ValueError):
        validate(cached)


def test_resume_rejects_future_pts_and_incorrect_grid(cached):
    cached[-1]["actual_prefix_seconds"][1] = 0.75
    with pytest.raises(ValueError, match="noncausal"):
        validate(cached)
    cached[-1]["actual_prefix_seconds"][1] = 0.5
    cached[-1]["requested_prefix_seconds"][0] = 0.1
    with pytest.raises(ValueError, match="requested time"):
        validate(cached)


def test_case_lock_excludes_overlap_and_cleans_up_on_error(tmp_path):
    lock = tmp_path / ".case.lock"
    with pytest.raises(RuntimeError):
        with exclusive_directory_lock(lock):
            with pytest.raises(ValueError, match="overlapping workers"):
                with exclusive_directory_lock(lock, timeout=0):
                    pass
            raise RuntimeError("interrupted")
    assert not lock.exists()
