"""Causality and representation checks without pretrained models or GPU."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ttt_frame.forecast_bench import ForecastCase
from ttt_frame.forecast_features import (
    FeatureConfig,
    QwenImageFeatures,
    cache_case,
    causal_video_frames,
    orthogonal_projection,
    pool_project_image,
    sampling_plan,
)


def _case(tmp_path: Path) -> ForecastCase:
    return ForecastCase(
        id="egolife_001", task="egocentric_action_forecasting_5s",
        context_path=tmp_path / "context.mp4", recent_path=tmp_path / "input.mp4",
        question="unused", options=(),
    )


def _video(path: Path) -> None:
    import av

    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=2)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for value in (0, 50, 100, 150):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), value, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_full_and_smoke_plans_preserve_a_regular_past_only_grid(tmp_path):
    case = _case(tmp_path)
    full = sampling_plan(case, FeatureConfig())
    assert len(full) == 3660 and full[0].prefix_seconds == 0
    assert full[-1].prefix_seconds == 1829.5
    assert full[3599].role == "context" and full[3600].role == "recent"
    assert full[3600].video_seconds == 0
    cropped = sampling_plan(case, FeatureConfig(context_tail_seconds=60))
    assert len(cropped) == 180 and cropped[0].prefix_seconds == 1740
    assert all(b.prefix_seconds - a.prefix_seconds == 0.5
               for a, b in zip(cropped, cropped[1:]))
    limited = sampling_plan(case, FeatureConfig(context_tail_seconds=60, limit_frames=7))
    assert len(limited) == 7 and limited[-1].prefix_seconds == 1743


@pytest.mark.parametrize("options", [
    {"sample_fps": 4}, {"sample_fps": float("nan")}, {"max_side": 65},
    {"projection_dim": 64}, {"context_tail_seconds": -1}, {"limit_frames": -1},
])
def test_invalid_feature_settings_raise(options):
    with pytest.raises(ValueError):
        FeatureConfig(**options)


def test_decoder_returns_preceding_frame_and_actual_pts(tmp_path):
    path = tmp_path / "video.mp4"
    _video(path)
    samples = list(causal_video_frames(path, [0.25, 0.75, 1.25], max_side=64))
    assert [pts for _, pts in samples] == [0.0, 0.5, 1.0]
    assert [round(float(np.asarray(image).mean()) / 50) for image, _ in samples] == [0, 1, 2]
    # Tail seeking must keep the original media time origin.
    tail = list(causal_video_frames(path, [1.25, 1.75], max_side=64))
    assert [pts for _, pts in tail] == [1.0, 1.5]
    with pytest.raises(ValueError, match="short video"):
        list(causal_video_frames(path, [2.1], max_side=64))


def test_projection_is_fixed_orthogonal_and_pooling_is_spatial():
    before = torch.random.get_rng_state()
    projection = orthogonal_projection(32, 16, seed=7)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(projection.T @ projection, torch.eye(16), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(projection, orthogonal_projection(32, 16, seed=7))
    tokens = torch.randn(4 * 8, 32)
    result = pool_project_image(tokens, 4, 8, projection)
    assert result.shape == (8, 16)
    torch.testing.assert_close(result.mean(-1), torch.zeros(8), atol=1e-6, rtol=0)
    assert torch.isfinite(result).all()


def test_cache_metadata_has_only_past_visual_state(tmp_path, monkeypatch):
    import ttt_frame.forecast_features as module

    case = _case(tmp_path)
    for path in (case.context_path, case.recent_path):
        path.write_bytes(b"stub used with decoder mock")

    def fake_frames(path, requested_seconds, *, max_side):
        for t in requested_seconds:
            yield Image.new("RGB", (64, 64)), t

    class FakeEncoder:
        config = FeatureConfig(context_tail_seconds=0, limit_frames=6)
        model_fingerprint = {"sha256": "test"}

        def encode(self, images):
            return torch.ones(len(images), 8, 256)

    monkeypatch.setattr(module, "causal_video_frames", fake_frames)
    result = cache_case(case, FakeEncoder(), input_manifest_sha256="public-input", batch_size=4)
    assert result["features"].shape == (6, 8, 256)
    assert result["requested_prefix_seconds"].tolist() == [1800 + 0.5 * i for i in range(6)]
    assert result["cropped"] is True and result["complete_prefix"] is False
    assert result["audio_used"] is False
    assert "question" not in result and "options" not in result and "answer" not in result
    path = tmp_path / "cache.pt"
    torch.save(result, path)
    reloaded = torch.load(path, weights_only=True)
    assert reloaded["sample_id"] == case.id


def test_real_tiny_visual_tower_keeps_images_independent(tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
        Qwen2VLImageProcessorFast,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLConfig(vision_config={
        "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
        "out_hidden_size": 256, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2, "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    })
    config.vision_config._attn_implementation = "sdpa"
    vision = Qwen3VLVisionModel(config.vision_config)
    config.save_pretrained(tmp_path)
    processor = Qwen2VLImageProcessorFast(
        size={"shortest_edge": 1024, "longest_edge": 128**2}, patch_size=16,
        temporal_patch_size=2, merge_size=2,
        image_mean=[0.5] * 3, image_std=[0.5] * 3,
    )
    processor.save_pretrained(tmp_path)
    save_file({f"model.visual.{k}": v for k, v in vision.state_dict().items()},
              str(tmp_path / "model.safetensors"))
    encoder = QwenImageFeatures(tmp_path, FeatureConfig(max_side=128, dtype="float32"))
    first = Image.new("RGB", (128, 128), color=(20, 120, 200))
    later = Image.new("RGB", (128, 128), color=(210, 10, 30))
    alone, batched = encoder.encode([first]), encoder.encode([first, later])
    assert alone.shape == (1, 8, 256) and batched.shape == (2, 8, 256)
    torch.testing.assert_close(alone[0], batched[0], atol=1e-5, rtol=1e-5)
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert encoder.model_fingerprint["sha256"]
