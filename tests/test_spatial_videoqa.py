"""Teacher-free video lifecycle checks with real, tiny random Qwen3-VL weights."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from test_spatial_model import assert_same_state, different_state, settings, snapshot, tiny_qwen
from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyVideoProcessor:
    """Build native video patches offline; no learned tokenizer or processor files."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        content = messages[0]["content"]
        if any(item["type"] == "video" for item in content):
            return "<video> Observe the visible environment."
        return next(item["text"] for item in content if item["type"] == "text")

    def __call__(self, text, videos=None, **kwargs):
        if videos is None:
            tokens = [3] + [10 + ord(character) % 100 for character in text[0][:12]] + [4]
            ids = torch.tensor([tokens])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        # One temporal patch contains two frames; retain Qwen's 2x2 merge ordering.
        frames = videos[0]
        chosen = frames[[0, -1]]
        pixels = torch.from_numpy(chosen.copy()).float().permute(0, 3, 1, 2) / 255
        pixels = torch.nn.functional.interpolate(pixels, size=(64, 64), mode="bilinear")
        patches = pixels.reshape(2, 3, 2, 2, 16, 2, 2, 16).permute(
            2, 5, 3, 6, 1, 0, 4, 7,
        ).reshape(16, 3 * 2 * 16 * 16)
        ids = torch.tensor([[123, 126, 126, 126, 126, 124, 1, 2]])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values_videos": patches,
            "video_grid_thw": torch.tensor([[1, 4, 4]]),
        }

    def batch_decode(self, token_ids, **kwargs):
        return [" ".join(map(str, row.tolist())) for row in token_ids]


def tiny_video_engine(**overrides):
    options = dict(
        model_path="tiny-offline-qwen",
        device="cpu",
        dtype="float32",
        chunk_seconds=1,
        frames_per_chunk=2,
        max_side=64,
        max_chunks=1,
        max_new_tokens=3,
        spatial=settings(use_conv=True, ttt_scale_init=0.3),
    )
    options.update(overrides)
    model = tiny_qwen()
    model.generate = Mock(side_effect=AssertionError("ingestion must not ask a caption/QA teacher"))
    return SpatialVideoMemory(
        SpatialVideoConfig(**options), model=model, processor=TinyVideoProcessor(),
    )


def frames(color):
    images = [Image.new("RGB", (32, 32), color) for _ in range(2)]
    ImageDraw.Draw(images[1]).rectangle((4, 4, 16, 20), fill="white")
    return images


def assert_no_observation_history(engine):
    forbidden = {"notes", "frames", "images", "observations", "video_path", "teacher", "optimizer"}
    assert not forbidden.intersection(vars(engine))
    for layer in engine.controller.layers.values():
        assert layer._video_mask is None and layer._video_grid is None
        if layer.state is not None:
            assert layer.state.pending_tokens == 0
            assert all(getattr(layer.state, name).grad_fn is None for name in ("w0", "w1", "w2"))
    assert not any("pending_" in name for name in engine.controller.memory_state_dict())
    assert all(getattr(module, "rope_deltas", None) is None for module in engine.model.modules())
    assert all(parameter.grad is None for parameter in engine.model.parameters())
    engine.model.generate.assert_not_called()


def test_observe_ask_observe_save_load_and_resume_preserves_memory(tmp_path):
    engine = tiny_video_engine()
    first = engine.ingest_frames(frames("red"), [0, 0.5])
    assert first["frames"] == 2 and first["visual_tokens"] == 4
    after_first = snapshot(engine.controller)
    first_position = engine.position_offset
    assert first_position > 0
    assert engine.answer("Where is the mug?")
    assert_same_state(snapshot(engine.controller), after_first)
    assert engine.position_offset == first_position
    assert_no_observation_history(engine)

    engine.ingest_frames(frames("blue"), [4, 4.5])
    assert different_state(snapshot(engine.controller), after_first)
    assert engine.position_offset > first_position
    saved_state = snapshot(engine.controller)
    saved_position = engine.position_offset
    answer = engine.answer("Where is the mug?")
    directory = tmp_path / "memory"
    engine.save(directory)
    assert {path.name for path in directory.iterdir()} == {
        "memory.json", "fast_weights.safetensors", "spatial.safetensors",
    }
    assert not any("pending_" in key for key in load_file(str(directory / "fast_weights.safetensors")))
    assert "Where is the mug" not in (directory / "memory.json").read_text()

    restored = tiny_video_engine()
    restored.load_memory(directory)
    assert restored.position_offset == saved_position
    assert restored.answer("Where is the mug?") == answer
    assert_same_state(snapshot(restored.controller), saved_state)
    assert_no_observation_history(restored)

    for current in (engine, restored):
        current.ingest_frames(frames("green"), [8, 8.5])
        assert_no_observation_history(current)
    assert_same_state(snapshot(restored.controller), snapshot(engine.controller))
    assert restored.position_offset == engine.position_offset > saved_position
    assert restored.answer("What changed?") == engine.answer("What changed?")
    for key in ("chunks", "frames", "visual_tokens", "input_tokens"):
        assert restored.stats[key] == engine.stats[key]


def test_deleted_source_video_is_unnecessary_for_query_and_reloaded_memory(tmp_path):
    av = pytest.importorskip("av")
    video = tmp_path / "private_household_evidence.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(8):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), (index * 25, 30, 60)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    engine = tiny_video_engine()
    report = engine.ingest_video(video)
    assert report["sessions"] == 1 and report["frames"] == 2
    video.unlink()
    assert engine.answer("What is visible?")
    summary = engine.finish_ingest()
    assert summary["parameter_only"]
    assert summary["retained_video_frames"] == summary["retained_text_records"] == 0
    directory = tmp_path / "memory"
    engine.save(directory)
    assert video.name not in (directory / "memory.json").read_text()
    restored = tiny_video_engine()
    restored.load_memory(directory)
    assert restored.answer("What is visible?") == engine.answer("What is visible?")
    assert_no_observation_history(restored)


@pytest.mark.parametrize("corruption", ["json", "fast_nan", "slow_shape", "position", "stats"])
def test_corrupt_checkpoint_is_rejected_without_destroying_existing_memory(tmp_path, corruption):
    engine = tiny_video_engine()
    engine.ingest_frames(frames("red"))
    directory = tmp_path / "memory"
    engine.save(directory)
    engine.ingest_frames(frames("blue"))
    before = snapshot(engine.controller)
    before_position = engine.position_offset
    before_stats = dict(engine.stats)
    if corruption in {"json", "position", "stats"}:
        metadata_file = directory / "memory.json"
        metadata = json.loads(metadata_file.read_text())
        if corruption == "position":
            metadata["position_offset"] = -1
        if corruption == "stats":
            metadata["stats"] = "not a statistics mapping"
        metadata_file.write_text("{broken" if corruption == "json" else json.dumps(metadata))
    else:
        name = "fast_weights.safetensors" if corruption == "fast_nan" else "spatial.safetensors"
        checkpoint = directory / name
        tensors = load_file(str(checkpoint))
        key = next(key for key, value in tensors.items() if value.ndim > 0)
        if corruption == "fast_nan":
            tensors[key].reshape(-1)[0] = float("nan")
        else:
            tensors[key] = torch.zeros(1)
        save_file(tensors, str(checkpoint))
    with pytest.raises(ValueError):
        engine.load_memory(directory)
    assert_same_state(snapshot(engine.controller), before)
    assert engine.position_offset == before_position
    assert engine.stats == before_stats
    assert engine.answer("Where is the mug?")


def test_failed_ingestion_blocks_queries_and_resume_until_reset(tmp_path, monkeypatch):
    engine = tiny_video_engine()
    engine.ingest_frames(frames("red"))
    directory = tmp_path / "valid"
    engine.save(directory)
    original_encoder = engine._encode_video
    monkeypatch.setattr(engine, "_encode_video", Mock(side_effect=RuntimeError("bad video pixels")))
    with pytest.raises(RuntimeError, match="bad video pixels"):
        engine.ingest_frames(frames("blue"))
    assert engine.state == "failed"
    for operation in (
        lambda: engine.answer("Where is the mug?"),
        lambda: engine.ingest_frames(frames("green")),
        engine.finish_ingest,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    engine.load_memory(directory)
    assert engine.state == "ready" and engine.answer("Where is the mug?")
    monkeypatch.setattr(engine, "_encode_video", original_encoder)
    engine.reset()
    assert engine.position_offset == 0 and not engine.controller.memory_state_dict()
    engine.ingest_frames(frames("green"))
    assert engine.answer("What is visible?")
    assert_no_observation_history(engine)


def test_nonfinite_final_write_is_rejected_even_when_output_logits_are_finite(monkeypatch):
    engine = tiny_video_engine(spatial=settings(
        chunk_size=32, window_size=32, use_conv=True, ttt_scale_init=0.0,
    ))
    memory = engine.controller.layers[2].memory
    original_write = memory._write_chunk

    def corrupt_final_update(state):
        # The final tail update happens after its queries have already read the old weights.
        updated = original_write(state)
        return replace(updated, w0=torch.full_like(updated.w0, float("nan")))

    monkeypatch.setattr(memory, "_write_chunk", corrupt_final_update)
    finite_outputs = []
    hook = engine.model.register_forward_hook(
        lambda module, inputs, output: finite_outputs.append(bool(torch.isfinite(output.logits).all()))
    )
    try:
        with pytest.raises((ValueError, RuntimeError), match="(?i)(nonfinite|finite|nan)"):
            engine.ingest_frames(frames("red"))
    finally:
        hook.remove()
    assert finite_outputs == [True]
    assert engine.state == "failed"
    with pytest.raises(RuntimeError, match="failed"):
        engine.answer("Where is the mug?")
