"""Offline lifecycle/gradient tests. Tiny random weights are NOT a VideoQA evaluation."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
pytest.importorskip("av")

from ttt_frame.video import iter_video_chunks
from ttt_frame.videoqa import VideoTTTConfig, VideoTTTMemory, masked_example, parse_qa


class TinyProcessor:
    """An offline text tokenizer; visual delivery is tested separately with real weights."""

    def __init__(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        words = "[PAD] [UNK] [EOS] user assistant Where is mug blue shelf red table ? .".split()
        backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        parts = []
        for msg in messages:
            parts.append(msg["role"] + " " + " ".join(
                c["text"] for c in msg["content"] if c["type"] == "text"
            ))
        parts.append("assistant " if add_generation_prompt else "[EOS]")
        return " ".join(parts)

    def __call__(self, text, images=None, **kwargs):
        return self.tokenizer(text, return_token_type_ids=False, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


def tiny_engine(**overrides):
    torch.manual_seed(71)
    config = transformers.SmolVLMConfig(
        text_config=dict(model_type="llama", vocab_size=32, hidden_size=32,
                         intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, pad_token_id=0, eos_token_id=2,
                         max_position_embeddings=256),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=8),
        pad_token_id=0, image_token_id=31,
    )
    base = transformers.SmolVLMForConditionalGeneration(config)
    settings = dict(model_path="tiny-offline", device="cpu", dtype="float32", rank=2,
                    lora_alpha=4, learning_rate=0.02, steps_per_chunk=5,
                    max_length=128, chunk_seconds=1, frames_per_chunk=2, seed=3)
    settings.update(overrides)
    return VideoTTTMemory(VideoTTTConfig(**settings), model=base, processor=TinyProcessor())


def make_video(path):
    import av
    from PIL import Image

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for i in range(12):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), (i * 20, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_sampling_is_chronological_bounded_and_closes_handle(tmp_path):
    video = make_video(tmp_path / "video.mp4")
    chunks = list(iter_video_chunks(video, chunk_seconds=1, frames_per_chunk=2, max_side=16))
    assert [c.timestamps for c in chunks] == [[0, 0.5], [1, 1.5], [2, 2.5]]
    assert all(im.size == (16, 16) for c in chunks for im in c.images)
    means = [im.getpixel((0, 0))[0] for c in chunks for im in c.images]
    assert means == sorted(means) and means[-1] > means[0] + 100
    generator = iter_video_chunks(video, chunk_seconds=1, frames_per_chunk=2, max_chunks=1)
    assert len(list(generator)) == 1
    video.rename(tmp_path / "closed.mp4")


def test_mask_excludes_question_and_rejects_prefix_mismatch():
    prompt = {"input_ids": torch.tensor([[3, 4, 5]])}
    full = {"input_ids": torch.tensor([[3, 4, 5, 6, 2]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    batch, cut = masked_example(prompt, full, 4)
    assert batch["labels"].tolist() == [[-100, -100, -100, 6]] and cut
    with pytest.raises(ValueError, match="prefix"):
        masked_example({"input_ids": torch.tensor([[9]])}, full, 8)
    with pytest.raises(ValueError, match="fit"):
        masked_example(prompt, full, 3)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_only_lora_changes_and_teacher_and_reset_restore_base(checkpointing):
    engine = tiny_engine(gradient_checkpointing=checkpointing)
    frozen = {n: p.detach().clone() for n, p in engine.model.named_parameters() if not p.requires_grad}
    inputs = engine._encode("Where is mug ?")
    with torch.no_grad():
        initial_logits = engine.model(**inputs).logits.clone()
    metrics = engine._learn([("Where is mug ?", "blue shelf")])
    assert metrics["loss_last"] < metrics["loss_first"]
    assert any(not torch.equal(p, engine._initial[n]) for n, p in engine.trainable.items())
    assert all(torch.equal(p, frozen[n]) for n, p in engine.model.named_parameters() if n in frozen)
    engine.model.eval()
    with engine.model.disable_adapter(), torch.no_grad():
        assert torch.equal(engine.model(**inputs).logits, initial_logits)
    engine.finish_ingest()
    assert engine.optimizer is None and all(p.grad is None for p in engine.trainable.values())
    engine.reset()
    with torch.no_grad():
        assert torch.equal(engine.model(**inputs).logits, initial_logits)
    assert engine.stats["sessions"] == 0


def test_ingest_revoke_query_save_reload_contains_only_parameters(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    engine = tiny_engine(steps_per_chunk=1, max_chunks=1)
    video = make_video(tmp_path / "evidence.mp4")
    teacher = Mock(side_effect=["blue mug on shelf", '[{"question":"Where is mug ?","answer":"blue shelf"}]'])
    monkeypatch.setattr(engine, "_generate", teacher)
    stats = engine.ingest_video(video)
    assert stats["qa_pairs"] == 1 and stats["frames"] == 2
    assert teacher.call_args_list[0].kwargs["images"]
    assert all(c.kwargs["teacher"] for c in teacher.call_args_list)
    engine.finish_ingest()
    video.unlink()  # queries have neither a source file nor a retained handle
    # Restore real generation after testing observation/training in isolation.
    monkeypatch.setattr(engine, "_generate", VideoTTTMemory._generate.__get__(engine))
    assert not any(k in vars(engine) for k in ("notes", "frames", "observations", "video_path"))
    destination = tmp_path / "memory"
    engine.save(destination)
    assert {p.name for p in destination.iterdir()} == {"memory.json", "adapter.safetensors"}
    assert all("lora_" in k for k in load_file(str(destination / "adapter.safetensors")))
    text = (destination / "memory.json").read_text()
    assert "evidence.mp4" not in text and "blue mug" not in text
    restored = tiny_engine(steps_per_chunk=1, max_chunks=1)
    restored.load_memory(destination)
    with torch.no_grad():
        inputs = engine._encode("Where is mug ?")
        assert torch.equal(engine.model(**inputs).logits, restored.model(**inputs).logits)
    original = {n: p.detach().clone() for n, p in engine.trainable.items()}
    # The public query goes through generation and does not train or receive images.
    spy = Mock(return_value="blue shelf")
    monkeypatch.setattr(engine, "_generate", spy)
    assert engine.answer("Where is mug ?") == "blue shelf"
    assert spy.call_args.kwargs == {"teacher": False}
    assert all(torch.equal(p, original[n]) for n, p in engine.trainable.items())
    with pytest.raises(RuntimeError, match="reset"):
        engine.ingest_video(video)


def test_empty_video_failure_cannot_turn_into_valid_memory(tmp_path):
    engine = tiny_engine()
    with pytest.raises(Exception):
        engine.ingest_video(tmp_path / "missing.mp4")
    assert engine.state == "failed"
    with pytest.raises(RuntimeError, match="failed"):
        engine.finish_ingest()
    with pytest.raises(RuntimeError):
        engine.answer("Where is mug ?")
    engine.reset()
    assert engine.state == "ingesting"


def test_multiple_sessions_accumulate_until_sealed(tmp_path, monkeypatch):
    engine = tiny_engine(steps_per_chunk=1, max_chunks=1, qa_per_chunk=0)
    engine.trace_file = tmp_path / "audit.jsonl"
    engine.set_trace_context(env_id="home1")
    video = make_video(tmp_path / "day.mp4")
    monkeypatch.setattr(engine, "_generate", Mock(return_value="blue mug on shelf"))
    engine.ingest_video(video)
    after_first = {n: p.detach().clone() for n, p in engine.trainable.items()}
    with pytest.raises(RuntimeError, match="finish_ingest"):
        engine.answer("Where is mug ?")
    engine.ingest_video(video)
    summary = engine.finish_ingest()["stats"]
    assert (summary["sessions"], summary["frames"], summary["optimizer_steps"]) == (2, 4, 2)
    assert summary["last_chunk_loss_last"] > 0
    assert any(not torch.equal(p, after_first[n]) for n, p in engine.trainable.items())
    import json
    audit = [json.loads(line) for line in engine.trace_file.read_text().splitlines()]
    assert len(audit) == 2 and audit[0]["env_id"] == "home1"
    assert audit[0]["observation"] == "blue mug on shelf"
    engine.save(tmp_path / "memory")
    assert "blue mug" not in (tmp_path / "memory" / "memory.json").read_text()


def test_pixels_cannot_be_silently_dropped():
    from PIL import Image

    engine = tiny_engine()
    with pytest.raises(RuntimeError, match="no pixels"):
        engine._encode("Describe", [Image.new("RGB", (16, 16))])


def test_pseudo_qa_rejects_malformed_output_and_deduplicates():
    assert parse_qa("not JSON", 4) == []
    assert parse_qa('[{"question":"q","answer":4}]', 4) == []
    text = '```json\n[{"question":"q","answer":"a"},{"question":"Q","answer":"b"}]\n```'
    assert parse_qa(text, 4) == [("q", "a")]


def test_mismatched_checkpoint_rejected_before_mutation(tmp_path):
    engine = tiny_engine()
    engine.finish_ingest()
    engine.save(tmp_path / "saved")
    other = tiny_engine(rank=4)
    with pytest.raises(ValueError, match="incompatible"):
        other.load_memory(tmp_path / "saved")
    assert all(torch.equal(p, other._initial[n]) for n, p in other.trainable.items())
