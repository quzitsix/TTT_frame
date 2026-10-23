"""CPU-only contracts for constrained forecasting MCQ inference and resumption."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_forecast_mcq as runner
from ttt_frame.forecast_mcq import OFFICIAL_ARMS, QWEN_ARMS


class FakeTokenizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {letter: [index + 1] for index, letter in enumerate("ABCDEF")}

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return self.mapping[text]

    def decode(self, token_ids):
        return f"token_{token_ids[0]}"


def test_choice_is_constrained_even_when_unrestricted_token_is_much_larger():
    tokenizer = FakeTokenizer()
    token_ids = runner.label_tokens(tokenizer)
    logits = torch.tensor([30.0, -1.0, 0.0, 4.0, 1.0, 2.0, 3.0, 40.0])
    result = runner.choose_label(logits, token_ids, tokenizer)
    assert result["prediction"] == result["raw_answer"] == "C"
    assert result["unrestricted_top_token_id"] == 7
    assert result["unrestricted_top_token"] == "token_7"
    assert result["label_logits"] == logits[1:7].tolist()
    assert sum(result["label_probabilities"]) == pytest.approx(1)
    assert 0 < result["label_probability_mass"] < 1e-10
    assert result["status"] == "ok"


def test_constrained_readout_respects_token_id_order_and_first_label_ties():
    tokens = [6, 4, 2, 5, 1, 3]
    logits = torch.tensor([-10.0, 0.0, 1.0, 2.0, 9.0, 3.0, 4.0])
    assert runner.choose_label(logits, tokens, FakeTokenizer())["prediction"] == "B"
    assert runner.choose_label(torch.zeros(8), tokens, FakeTokenizer())["prediction"] == "A"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_are_rejected_even_outside_choice_tokens(bad_value):
    scores = torch.zeros(8)
    scores[7] = bad_value
    with pytest.raises(ValueError, match="nonfinite"):
        runner.choose_label(scores, list(range(1, 7)), FakeTokenizer())


@pytest.mark.parametrize("replacement", [[], [1, 2], [2]])
def test_label_tokenization_must_be_single_and_distinct(replacement):
    mapping = {letter: [index + 1] for index, letter in enumerate("ABCDEF")}
    mapping["A"] = replacement
    with pytest.raises(ValueError, match="distinct single-token"):
        runner.label_tokens(FakeTokenizer(mapping))


@pytest.mark.parametrize("use_memory,mode", [(False, "base"), (True, "read")])
def test_text_answer_uses_single_read_only_forward_and_preserves_stream_offset(use_memory, mode):
    calls, cleanup, contexts = [], [], []

    class Processor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["text"] == "question"
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "rendered"

        def __call__(self, **kwargs):
            assert kwargs == {"text": ["rendered"], "return_tensors": "pt"}
            return {"input_ids": torch.tensor([[10, 11]]), "attention_mask": torch.ones(1, 2)}

    @contextmanager
    def context(requested_mode):
        contexts.append(requested_mode)
        yield

    def model(**kwargs):
        assert not torch.is_grad_enabled()
        calls.append(kwargs)
        return SimpleNamespace(logits=torch.tensor([[[0., 0., 1., 9., 0., 0., 0., 20.]]]))

    engine = SimpleNamespace(
        processor=Processor(), model=model, position_offset=100, device=torch.device("cpu"),
        controller=SimpleNamespace(context=context), _move=lambda batch: batch,
        _clear_generation_state=lambda: cleanup.append("cleared"),
    )
    result = runner.answer_mcq(engine, "question", token_ids=list(range(1, 7)), use_memory=use_memory)
    assert result["prediction"] == "C" and result["input_tokens"] == 2
    assert result["visual_tokens_at_answer"] == 0
    assert result["position_offset"] == engine.position_offset == 100
    assert contexts == [mode] and len(calls) == 1 and len(cleanup) == 2
    assert calls[0]["use_cache"] is False and calls[0]["logits_to_keep"] == 1
    assert calls[0]["position_ids"].tolist() == [[[100, 101]], [[100, 101]], [[100, 101]]]


def _prediction_rows(case_id="egolife_001", arms=QWEN_ARMS, fingerprint="protocol"):
    return [{"id": case_id, "arm": arm, "prediction": "A", "raw_answer": "A", "status": "ok",
             "schema": "forecast_mcq_prediction_v1", "protocol_fingerprint": fingerprint}
            for arm in arms]


def _write_rows(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_resume_only_skips_complete_independent_episodes(tmp_path):
    output = tmp_path / "predictions.jsonl"
    selected = {"egolife_001", "egolife_002", "egolife_003"}
    assert runner.completed_ids(output, QWEN_ARMS, selected, "protocol") == set()
    _write_rows(output, _prediction_rows() + _prediction_rows("egolife_003"))
    assert runner.completed_ids(output, QWEN_ARMS, selected, "protocol") == {
        "egolife_001", "egolife_003",
    }


@pytest.mark.parametrize("failure", ["partial", "duplicate", "error", "protocol", "id", "arm"])
def test_resume_rejects_incompatible_or_incomplete_rows(tmp_path, failure):
    rows = _prediction_rows()
    if failure == "partial":
        rows.pop()
    elif failure == "duplicate":
        rows.append(dict(rows[0]))
    elif failure == "error":
        rows[1]["status"] = "error"
    elif failure == "protocol":
        rows[0]["protocol_fingerprint"] = "different"
    elif failure == "id":
        rows[0]["id"] = "egolife_002"
    elif failure == "arm":
        rows[0]["arm"] = "spatial"
    output = tmp_path / "predictions.jsonl"
    _write_rows(output, rows)
    with pytest.raises(ValueError):
        runner.completed_ids(output, QWEN_ARMS, {"egolife_001"}, "protocol")


def test_resume_does_not_silently_repair_truncated_json(tmp_path):
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"id":"egolife_001"')
    with pytest.raises(json.JSONDecodeError):
        runner.completed_ids(output, QWEN_ARMS, {"egolife_001"}, "protocol")


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong_schema"), ("prediction", "G"), ("prediction", None),
    ("raw_answer", "B"), ("raw_answer", None),
])
def test_resume_rejects_malformed_successful_rows(tmp_path, field, value):
    rows = _prediction_rows()
    rows[0][field] = value
    output = tmp_path / "predictions.jsonl"
    _write_rows(output, rows)
    with pytest.raises(ValueError):
        runner.completed_ids(output, QWEN_ARMS, {"egolife_001"}, "protocol")


@pytest.fixture
def tiny_bench(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    rows = []
    for index in range(1, 5):
        sample_id = f"egolife_{index:03d}"
        sample = root / "samples" / sample_id
        sample.mkdir(parents=True)
        for filename in ("context.mp4", "input.mp4"):
            (sample / filename).write_bytes(b"not a video: never decode in protocol tests")
        rows.append({
            "id": sample_id, "task": "egocentric_action_forecasting_5s",
            "input": {"context_video": f"samples/{sample_id}/context.mp4",
                      "video": f"samples/{sample_id}/input.mp4", "question": "What happens next?",
                      "context_duration_seconds": 1800, "input_duration_seconds": 30,
                      "prediction_horizon_seconds": 5,
                      "options": [{"label": letter, "text": f"action {letter}"} for letter in "ABCDEF"]},
        })
    _write_rows(root / "inputs_mcq.jsonl", rows)
    (root / "answers.jsonl").write_text("PRIVATE AND INVALID JSON")
    (root / "metadata.jsonl").write_text("PRIVATE AND INVALID JSON")
    model = tmp_path / "fake_model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"fake weights: must never be loaded")
    (model / "config.json").write_text("{}")
    checkpoint = tmp_path / "fake_spatial.safetensors"
    checkpoint.write_bytes(b"fake official checkpoint")
    protocol = tmp_path / "protocol.json"
    monkeypatch.setattr(runner, "source_hashes", lambda: {"runner": "frozen-source"})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "fake-commit\n")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("CPU protocol tests must not load a model")

    monkeypatch.setattr(runner, "QwenEngine", forbidden_model)
    prepare = ["--root", str(root), "--protocol", str(protocol), "--prepare",
               "--model-path", str(model), "--spatial-checkpoint", str(checkpoint)]
    return {"root": root, "model": model, "checkpoint": checkpoint,
            "protocol": protocol, "prepare": prepare}


def _prepare(fixture, capsys):
    assert runner.main(fixture["prepare"]) == 0
    capsys.readouterr()
    return runner.read_protocol(fixture["protocol"])


def test_prepare_fingerprints_fake_artifacts_and_does_not_read_private_files(tiny_bench, capsys):
    protocol = _prepare(tiny_bench, capsys)
    payload = protocol["payload"]
    assert len(payload["case_ids"]) == 4
    assert payload["episode_reset"] is True
    assert payload["training_on_benchmark"] is False
    assert payload["sampling"]["audio_used"] is False
    assert set(payload["base_artifact_sha256"]) == {"config.json", "model.safetensors"}
    assert payload["spatial_checkpoint_sha256"] == runner.file_hash(tiny_bench["checkpoint"])
    assert protocol["protocol_fingerprint"] == runner.fingerprint(payload)
    assert protocol["git_revision"] == "fake-commit"
    with pytest.raises(ValueError, match="already exists"):
        runner.main(tiny_bench["prepare"])


@pytest.mark.parametrize("family,arms", [("qwen", QWEN_ARMS), ("spatial", OFFICIAL_ARMS)])
def test_dry_shards_are_disjoint_exhaustive_and_need_no_model(tiny_bench, capsys, family, arms):
    _prepare(tiny_bench, capsys)
    results = []
    for index in (0, 1):
        assert runner.main([
            "--root", str(tiny_bench["root"]), "--protocol", str(tiny_bench["protocol"]),
            "--family", family, "--dry-run", "--num-shards", "2", "--shard-index", str(index),
        ]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["arms"] == list(arms)
        assert result["expected_rows"] == 2 * len(arms)
        results.append(set(result["case_ids"]))
    assert not results[0].intersection(results[1])
    assert results[0] | results[1] == {f"egolife_{index:03d}" for index in range(1, 5)}


def test_subset_filter_then_shard_is_deterministic(tiny_bench, capsys):
    _prepare(tiny_bench, capsys)
    assert runner.main([
        "--root", str(tiny_bench["root"]), "--protocol", str(tiny_bench["protocol"]),
        "--dry-run", "--ids", "egolife_004", "egolife_002", "egolife_003",
        "--num-shards", "2", "--shard-index", "1",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["case_ids"] == ["egolife_003"]


@pytest.mark.parametrize("failure", ["payload", "manifest", "source"])
def test_protocol_rejects_tampering(tiny_bench, capsys, monkeypatch, failure):
    protocol = _prepare(tiny_bench, capsys)
    if failure == "payload":
        protocol["payload"]["seed"] = 9
        tiny_bench["protocol"].write_text(json.dumps(protocol))
    elif failure == "manifest":
        with (tiny_bench["root"] / "inputs_mcq.jsonl").open("a") as stream:
            stream.write("\n")
    else:
        monkeypatch.setattr(runner, "source_hashes", lambda: {"runner": "changed-source"})
    with pytest.raises(ValueError, match="changed|fingerprint"):
        runner.read_protocol(tiny_bench["protocol"])


def test_prepare_cannot_write_protocol_into_source_data(tiny_bench):
    target = tiny_bench["root"] / "protocol.json"
    args = list(tiny_bench["prepare"])
    args[args.index("--protocol") + 1] = str(target)
    with pytest.raises(SystemExit) as exc:
        runner.main(args)
    assert exc.value.code == 2 and not target.exists()


def test_inference_cannot_write_into_protocol_source_with_different_cli_root(tiny_bench, capsys, tmp_path):
    _prepare(tiny_bench, capsys)
    target = tiny_bench["root"] / "predictions.jsonl"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--root", str(tmp_path / "unrelated-root"),
                     "--protocol", str(tiny_bench["protocol"]), "--output", str(target)])
    assert exc.value.code == 2 and not target.exists()


def test_finished_resume_does_not_reload_model(tiny_bench, capsys, tmp_path):
    protocol = _prepare(tiny_bench, capsys)
    output = tmp_path / "completed.jsonl"
    _write_rows(output, [row for index in range(1, 5) for row in _prediction_rows(
        f"egolife_{index:03d}", fingerprint=protocol["protocol_fingerprint"],
    )])
    before = output.read_bytes()
    assert runner.main([
        "--root", str(tiny_bench["root"]), "--protocol", str(tiny_bench["protocol"]),
        "--family", "qwen", "--output", str(output), "--resume",
    ]) == 0
    assert "already complete" in capsys.readouterr().out
    assert output.read_bytes() == before
