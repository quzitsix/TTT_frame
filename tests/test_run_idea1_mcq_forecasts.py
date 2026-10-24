"""Runner integration, input isolation and strict resumption checks."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from scripts import run_idea1_mcq_forecasts as runner
from ttt_frame.forecast_features import FeatureConfig


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("idea1_runner")
    root, cache_dir = directory / "source", directory / "cache"
    root.mkdir()
    cache_dir.mkdir()
    ids = ["egolife_001", "egolife_002", "egolife_003"]
    manifest = root / "inputs_mcq.jsonl"
    manifest.write_text("".join(json.dumps({"id": sample_id}) + "\n" for sample_id in ids))
    manifest_sha = runner.sha256_file(manifest)
    split = {
        "schema": "forecast_bench_research_day_split_v1", "official_split": False,
        "grouping": "recording_day_all_participants", "root": str(root),
        "inputs_manifest_sha256": manifest_sha,
        "days": {"train": [1], "validation": [5], "test": [6]},
        "splits": {role: [sample_id] for role, sample_id in zip(runner.ROLES, ids)},
        "id_to_day": dict(zip(ids, (1, 5, 6))),
    }
    split_path = directory / "split.json"
    split_path.write_text(json.dumps(split))
    config = asdict(FeatureConfig(max_side=224))
    protocol = {
        "kind": "idea01_full_prefix_visual_cache_protocol", "format_version": 1,
        "feature_config": config, "model_fingerprint": {"sha256": "b" * 64},
        "input_manifest_sha256": manifest_sha,
        "execution": {"images_attend_independently": True, "batch_size": 64},
        "source_sha256": {"extractor.py": "c" * 64},
        "storage_dtype": "float32", "audio_used": False, "future_media_used": False,
        "questions_or_choices_encoded": False,
    }
    feature_record = {"protocol_sha256": runner.canonical_sha256(protocol), "protocol": protocol}
    (cache_dir / "cache_protocol.json").write_text(json.dumps(feature_record))
    times = torch.arange(3660, dtype=torch.float64) / 2
    # Shared benign variation keeps the frozen predictor well conditioned and
    # the test inexpensive; validation/test differ without entering the fit.
    sequence = torch.sin(torch.arange(3660, dtype=torch.float32) / 37).view(-1, 1, 1).expand(-1, 8, 256).clone()
    for index, sample_id in enumerate(ids):
        features = sequence + index * 0.01
        sources = {}
        for role in ("context", "recent"):
            path = root / f"{sample_id}_{role}.mp4"
            path.write_bytes(b"public past media placeholder")
            sources[role] = {"path": str(path), "size_bytes": path.stat().st_size,
                             "mtime_ns": path.stat().st_mtime_ns, "sha256": runner.sha256_file(path)}
        cache = {
            "kind": "bench_v0_causal_visual_features", "format_version": 1,
            "sample_id": sample_id, "input_manifest_sha256": manifest_sha,
            "complete_prefix": True, "cropped": False, "sample_fps": 2.0,
            "encoder": "qwen3_vl_per_image", "audio_used": False,
            "features": features, "requested_prefix_seconds": times,
            "actual_prefix_seconds": times.clone(), "feature_sha256": runner.tensor_sha256(features),
            "model_fingerprint": protocol["model_fingerprint"], "config": config,
            "execution": protocol["execution"], "protocol_sha256": feature_record["protocol_sha256"],
            "source_media": sources,
        }
        torch.save(cache, cache_dir / f"{sample_id}.pt")
    return directory, cache_dir, split_path, split


def test_preflight_validates_every_split_and_retains_only_train(dataset):
    _, cache_dir, split_path, split = dataset
    _, protocol, training = runner.prepare_inputs(cache_dir, split_path)
    assert len(training) == 1
    expected = torch.load(cache_dir / "egolife_001.pt", weights_only=True)["features"]
    assert torch.equal(training[0], expected)
    assert set(protocol["cache_sources"]) == set(split["id_to_day"])
    assert protocol["recent_predictor"] == {
        "horizons": [1, 2, 4, 10], "window_steps": 4, "ridge": 0.01,
        "standardize": True, "max_fit_rows": 20000,
    }
    assert protocol["answer_keys_used"] is False and protocol["future_media_used"] is False


def test_missing_test_cache_blocks_everything_before_output_or_fit(dataset, tmp_path):
    _, cache_dir, split_path, _ = dataset
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    for name in ("egolife_001.pt", "egolife_002.pt", "cache_protocol.json"):
        (incomplete / name).symlink_to(cache_dir / name)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="not ready"):
        runner.main(["--cache-dir", str(incomplete), "--split-manifest", str(split_path),
                     "--output-dir", str(output), "--workers", "1", "--threads", "1"])
    assert not output.exists()


@pytest.mark.parametrize("problem", ["feature_checksum", "protocol", "future_time", "partial", "source_changed"])
def test_corrupt_or_noncausal_cache_is_rejected(dataset, tmp_path, problem):
    _, cache_dir, _, split = dataset
    cache = torch.load(cache_dir / "egolife_001.pt", weights_only=True)
    record = runner.read_feature_protocol(cache_dir, split)
    if problem == "feature_checksum":
        cache["features"][0, 0, 0] += 1
    elif problem == "protocol":
        cache["execution"]["batch_size"] = 1
    elif problem == "future_time":
        cache["actual_prefix_seconds"][1] += 0.1
    elif problem == "source_changed":
        cache["source_media"]["context"]["mtime_ns"] -= 1
    else:
        cache["cropped"] = True
    path = tmp_path / "changed.pt"
    torch.save(cache, path)
    with pytest.raises(ValueError):
        runner.validated_cache(path, "egolife_001", split, record)


def test_changed_public_manifest_is_rejected(dataset, tmp_path):
    _, cache_dir, _, split = dataset
    root = tmp_path / "different_root"
    root.mkdir()
    (root / "inputs_mcq.jsonl").write_text('{"id":"different"}\n')
    changed = deepcopy(split)
    changed["root"] = str(root)
    path = tmp_path / "changed_split.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="public input manifest checksum"):
        runner.prepare_inputs(cache_dir, path)


def test_full_synthetic_cli_fit_parallel_forecast_and_resume(dataset, tmp_path, monkeypatch):
    _, cache_dir, split_path, split = dataset
    output = tmp_path / "forecasts"
    args = ["--cache-dir", str(cache_dir), "--split-manifest", str(split_path),
            "--output-dir", str(output), "--workers", "2", "--threads", "1"]
    original_fit = runner.RidgeRecentPredictor.fit
    fits = []

    def observed_fit(sequences, **kwargs):
        fits.append(len(sequences))
        assert len(sequences) == 1
        return original_fit(sequences, **kwargs)

    monkeypatch.setattr(runner.RidgeRecentPredictor, "fit", observed_fit)
    assert runner.main(args) == 0
    assert fits == [1]
    predictor = torch.load(output / "recent_predictor.pt", weights_only=True)
    assert predictor["train_ids"] == split["splits"]["train"]
    expected_train = torch.load(cache_dir / "egolife_001.pt", weights_only=True)["features"]
    torch.testing.assert_close(predictor["baseline"]["mean"], expected_train.mean((0, 1)), atol=1e-6, rtol=1e-5)
    protocol_record = json.loads((output / "protocol.json").read_text())
    for sample_id in split["id_to_day"]:
        result = runner.validate_forecast(output / f"{sample_id}_forecasts.pt", sample_id, protocol_record)
        assert result["sample_id"] == sample_id
    before = (output / "egolife_003_forecasts.pt").stat().st_mtime_ns
    assert runner.main(args + ["--resume"]) == 0
    assert fits == [1]
    assert (output / "egolife_003_forecasts.pt").stat().st_mtime_ns == before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] and summary["reused_cases"] == 3
    assert summary["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert not (output / ".runner.lock").exists()
    # A changed implementation/model/source protocol cannot reuse the run.
    changed_record = deepcopy(protocol_record)
    changed_record["protocol"]["feature_protocol"]["protocol"]["model_fingerprint"]["sha256"] = "changed"
    (output / "protocol.json").write_text(json.dumps(changed_record))
    with pytest.raises(ValueError, match="resume rejected"):
        runner.main(args + ["--resume"])
    (output / "protocol.json").write_text(json.dumps(protocol_record))
    # Tensor checksums also reject damaged result artifacts.
    path = output / "egolife_003_forecasts.pt"
    damaged = torch.load(path, weights_only=True)
    damaged["last_forecasts"]["residual"][0, 0, 0] += 1
    torch.save(damaged, path)
    with pytest.raises(ValueError, match="corrupt forecast"):
        runner.main(args + ["--resume"])
