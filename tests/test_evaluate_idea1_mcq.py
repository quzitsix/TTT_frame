"""Readout orchestration: split isolation and frozen, answer-free test prediction."""
import hashlib
import json

import pytest
import torch

from scripts.cache_idea1_text import (
    canonical_sha256 as text_protocol_sha256, case_text_sha256, tensor_sha256 as text_tensor_sha256,
)
from scripts.evaluate_idea1_mcq import choose_lambda, load_fit_labels, main
from scripts.run_idea1_mcq_forecasts import canonical_sha256, sha256_file, tensor_sha256
from ttt_frame.forecast_bench import load_forecast_cases
from ttt_frame.forecast_residual import RidgeRecentPredictor
from ttt_frame.idea1_mcq_memory import ARMS
from ttt_frame.idea1_mcq_readout import SharedRidgeReadout


def test_fit_label_reader_excludes_test_fields(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('\n'.join(json.dumps(row) for row in [
        {"id": "train", "correct_answer": "B"},
        {"id": "validation", "correct_answer": "C"},
        {"id": "test", "correct_answer": {"deliberately": "invalid"}},
    ]))
    split = {"splits": {"train": ["train"], "validation": ["validation"], "test": ["test"]}}
    assert load_fit_labels(path, split, "train") == {"train": "B"}
    assert load_fit_labels(path, split, "validation") == {"validation": "C"}
    with pytest.raises(ValueError, match="test labels are forbidden"):
        load_fit_labels(path, split, "test")


def test_selection_uses_shared_validation_objective_and_fixed_tie_rule():
    assert choose_lambda([
        {"ridge_lambda": 0.01, "correct_sum": 10},
        {"ridge_lambda": 100, "correct_sum": 10},
        {"ridge_lambda": 10, "correct_sum": 9},
    ]) == 100


def setup_small_run(tmp_path):
    root = tmp_path / "bench"
    root.mkdir()
    ids = [f"egolife_{index:03d}" for index in range(1, 5)]
    days = [1, 1, 5, 6]
    public = []
    for sample_id in ids:
        sample = root / "samples" / sample_id
        sample.mkdir(parents=True)
        (sample / "context.mp4").touch()
        (sample / "input.mp4").touch()
        public.append({"id": sample_id, "task": "egocentric_action_forecasting_5s", "input": {
            "context_video": f"samples/{sample_id}/context.mp4",
            "video": f"samples/{sample_id}/input.mp4", "question": "What happens next?",
            "context_duration_seconds": 1800, "input_duration_seconds": 30,
            "prediction_horizon_seconds": 5,
            "options": [{"label": label, "text": f"Action {label}"} for label in "ABCDEF"],
        }})
    manifest = root / "inputs_mcq.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in public))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    split = {"schema": "forecast_bench_research_day_split_v1", "official_split": False,
             "root": str(root),
             "grouping": "recording_day_all_participants", "inputs_manifest_sha256": sha,
             "days": {"train": [1], "validation": [5], "test": [6]},
             "splits": {"train": ids[:2], "validation": ids[2:3], "test": ids[3:]},
             "id_to_day": dict(zip(ids, days))}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    # There is intentionally NO held-out answer row or metadata file.
    (root / "answers.jsonl").write_text("".join(json.dumps({"id": sid, "correct_answer": gold})
                                               + "\n" for sid, gold in zip(ids[:3], "BCD")))
    forecasts = tmp_path / "forecasts"
    forecasts.mkdir()
    protocol = {
        "schema": "idea01_mcq_forecasts_protocol_v1",
        "split_sha256": sha256_file(split_path), "splits": split["splits"],
        "input_manifest_sha256": sha, "arms": list(ARMS),
        "cache_sources": {
            sid: {"cache_sha256": hashlib.sha256((sid + "cache").encode()).hexdigest(),
                  "feature_sha256": hashlib.sha256((sid + "features").encode()).hexdigest()}
            for sid in ids
        },
        "sample_fps": 2, "prefix_seconds": 1830, "sequence_steps": 3660,
        "recent_predictor": {"horizons": [1, 2, 4, 10], "window_steps": 4,
                             "ridge": 0.01, "standardize": True, "max_fit_rows": 20000},
        "future_media_used": False, "answer_keys_used": False,
    }
    protocol_sha = canonical_sha256(protocol)
    predictor = RidgeRecentPredictor(
        horizons=(1, 2, 4, 10), window_steps=4, mean=torch.zeros(256), std=torch.ones(256),
        coefficients=torch.zeros(4, 513, 256), tokens=8, training_rows=(8, 8, 8, 8),
        standardize=True, ridge=0.01,
    )
    predictor_path = forecasts / "recent_predictor.pt"
    torch.save({"kind": "idea01_frozen_recent_ridge", "baseline": predictor.to_dict(),
                "protocol_sha256": protocol_sha, "train_ids": ids[:2]}, predictor_path)
    predictor_sha = sha256_file(predictor_path)
    (forecasts / "protocol.json").write_text(json.dumps({
        "protocol_sha256": protocol_sha, "protocol": protocol,
        "predictor_sha256": predictor_sha, "fit_seconds": 0.0,
    }))
    generator = torch.Generator().manual_seed(2)
    # A Chinese template makes canonical Unicode escaping part of the fixture.
    text_protocol = {
        "input_manifest_sha256": sha,
        "prompt_template": "问题：{question}\\n候选动作：{candidate}",
        "pooling": "last_nonpadding_token_after_final_decoder_norm",
        "answer_keys_used": False, "videos_used": False, "candidate_order_features": False,
    }
    text = {"schema": "idea01_candidate_text_features_v1", "input_manifest_sha256": sha,
            "protocol": text_protocol, "protocol_sha256": text_protocol_sha256(text_protocol),
            "records": {}}
    for case in load_forecast_cases(root):
        sample_id = case.id
        predictions = {arm: torch.randn(4, 8, 256, generator=generator) for arm in ARMS}
        artifact = {
            "schema": "idea01_mcq_case_forecasts_v1", "sample_id": sample_id,
            "split_role": next(role for role, members in split["splits"].items() if sample_id in members),
            "protocol_sha256": protocol_sha, "predictor_sha256": predictor_sha,
            **protocol["cache_sources"][sample_id],
            "last_forecasts": predictions, "stats": {arm: {} for arm in ARMS},
            "forecast_sha256": {arm: tensor_sha256(value) for arm, value in predictions.items()},
            "elapsed_seconds": 0.0, "last_actual_prefix_seconds": 1829.5,
            "forecast_target_seconds_after_cutoff": [0.0, 0.5, 1.5, 4.5],
        }
        artifact["metadata_sha256"] = canonical_sha256({
            key: value for key, value in artifact.items() if key != "last_forecasts"
        })
        torch.save(artifact, forecasts / f"{sample_id}_forecasts.pt")
        embeddings = torch.randn(6, 2048, generator=generator)
        text["records"][sample_id] = {
            "labels": [option.label for option in case.options], "embeddings": embeddings,
            "text_sha256": case_text_sha256(case), "embedding_sha256": text_tensor_sha256(embeddings),
        }
    text_path = tmp_path / "text.pt"
    torch.save(text, text_path)
    return root, forecasts, text_path, split_path, ids


def test_end_to_end_prediction_needs_no_test_answers(tmp_path, monkeypatch):
    root, forecasts, text_path, split_path, ids = setup_small_run(tmp_path)
    fit = SharedRidgeReadout.fit
    seen_fit_ids = []

    def observed_fit(train_features, train_options, labels, **kwargs):
        assert train_features.shape == (2, 7, 1024)
        assert train_options.shape == (2, 6, 2048)
        assert labels.tolist() == [1, 2]
        assert kwargs["train_ids"] == ids[:2]
        assert tuple(kwargs["arm_names"]) == ARMS
        seen_fit_ids.append(tuple(kwargs["train_ids"]))
        return fit(train_features, train_options, labels, **kwargs)

    monkeypatch.setattr(SharedRidgeReadout, "fit", observed_fit)
    output = tmp_path / "output"
    assert main(["--root", str(root), "--forecast-dir", str(forecasts),
                 "--text-cache", str(text_path), "--split-manifest", str(split_path),
                 "--output-dir", str(output), "--threads", "1"]) == 0
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * 8
    assert all(row["prediction"] in "ABCDEF" for row in rows)
    assert {row["research_split"] for row in rows if row["id"] == ids[-1]} == {"test"}
    protocol = json.loads((output / "protocol.json").read_text())["protocol"]
    assert protocol["train_ids"] == ids[:2] and not protocol["test_labels_used"]
    assert not protocol["refit_validation"]
    complete = json.loads((output / "completion.json").read_text())
    assert complete["complete"] and not complete["test_answers_loaded"]
    assert seen_fit_ids == [tuple(ids[:2])] * 5
    checkpoint = torch.load(output / "shared_readout.pt", weights_only=True)
    assert checkpoint["independent_train_cases"] == 2 and checkpoint["train_rows"] == 14
    assert checkpoint["train_ids"] == ids[:2]


def test_mixed_forecast_protocol_rejected_before_fit(tmp_path):
    root, forecasts, text_path, split_path, ids = setup_small_run(tmp_path)
    artifact = torch.load(forecasts / f"{ids[-1]}_forecasts.pt", weights_only=True)
    artifact["protocol_sha256"] = "another protocol"
    # Keep metadata internally valid so the identity mismatch is exercised.
    artifact["metadata_sha256"] = canonical_sha256({
        key: value for key, value in artifact.items()
        if key not in ("last_forecasts", "metadata_sha256")
    })
    torch.save(artifact, forecasts / f"{ids[-1]}_forecasts.pt")
    with pytest.raises(ValueError, match="forecast identity/protocol mismatch"):
        main(["--root", str(root), "--forecast-dir", str(forecasts),
              "--text-cache", str(text_path), "--split-manifest", str(split_path),
              "--output-dir", str(tmp_path / "output"), "--threads", "1"])


@pytest.mark.parametrize("problem, message", [
    ("text_protocol", "corrupt text cache protocol"),
    ("text_content", "candidate content/embedding checksum mismatch"),
    ("text_embedding", "candidate content/embedding checksum mismatch"),
    ("forecast_tensor", "corrupt forecast"),
    ("forecast_metadata", "corrupt forecast metadata checksum"),
    ("predictor", "forecast predictor checksum mismatch"),
])
def test_corrupt_artifacts_are_rejected_before_supervised_fit(tmp_path, monkeypatch, problem, message):
    root, forecasts, text_path, split_path, ids = setup_small_run(tmp_path)
    if problem.startswith("text_"):
        cache = torch.load(text_path, weights_only=True)
        if problem == "text_protocol":
            cache["protocol"]["prompt_template"] = "changed"
        elif problem == "text_content":
            cache["records"][ids[-1]]["text_sha256"] = "changed"
        else:
            cache["records"][ids[-1]]["embeddings"][0, 0] += 1
        torch.save(cache, text_path)
    elif problem == "predictor":
        with (forecasts / "recent_predictor.pt").open("ab") as output:
            output.write(b"unexpected checkpoint modification")
    else:
        path = forecasts / f"{ids[-1]}_forecasts.pt"
        artifact = torch.load(path, weights_only=True)
        if problem == "forecast_tensor":
            artifact["last_forecasts"]["residual"][0, 0, 0] += 1
        else:
            artifact["last_actual_prefix_seconds"] = 1830.0
        torch.save(artifact, path)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("invalid data must be rejected before readout fitting")

    monkeypatch.setattr(SharedRidgeReadout, "fit", forbidden_fit)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        main(["--root", str(root), "--forecast-dir", str(forecasts),
              "--text-cache", str(text_path), "--split-manifest", str(split_path),
              "--output-dir", str(output), "--threads", "1"])
    assert not output.exists()


def test_changed_day_split_cannot_reuse_fitted_forecasts(tmp_path, monkeypatch):
    root, forecasts, text_path, split_path, _ = setup_small_run(tmp_path)
    split = json.loads(split_path.read_text())
    # The replacement is a valid, day-disjoint split but differs from the
    # split under which the immutable recent predictor/forecasts were created.
    for key in ("splits", "days"):
        split[key]["validation"], split[key]["test"] = split[key]["test"], split[key]["validation"]
    split_path.write_text(json.dumps(split))

    def forbidden_fit(*args, **kwargs):
        pytest.fail("a different split must fail before label loading or fitting")

    monkeypatch.setattr(SharedRidgeReadout, "fit", forbidden_fit)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="forecast protocol/split/public input mismatch"):
        main(["--root", str(root), "--forecast-dir", str(forecasts),
              "--text-cache", str(text_path), "--split-manifest", str(split_path),
              "--output-dir", str(output), "--threads", "1"])
    assert not output.exists()
