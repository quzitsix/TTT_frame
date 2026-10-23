"""Verify the actual Qwen patch convolution and complete visual tower."""

import pytest
import torch

from ttt_frame.vision_patch_embed import LinearVisionPatchEmbed


def _model(dtype=torch.float32):
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLConfig(vision_config={
        "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
        "out_hidden_size": 32, "patch_size": 4, "spatial_merge_size": 2,
        "temporal_patch_size": 2, "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    }).vision_config
    config._attn_implementation = "sdpa"
    return Qwen3VLVisionModel(config).to(dtype=dtype).eval()


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_patch_projection_matches_real_convolution(dtype):
    torch.manual_seed(42)
    model = _model(dtype)
    pixels = torch.randn(32, 3 * 2 * 4 * 4, dtype=dtype)
    expected = model.patch_embed(pixels)
    original = model.patch_embed.forward
    with LinearVisionPatchEmbed(model.patch_embed) as patch:
        actual = model.patch_embed(pixels)
        tolerance = 0.008 if dtype == torch.bfloat16 else 2e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        assert patch.optimized_calls == 1 and patch.fallback_calls == 0
        assert patch.module_count == 1
    assert model.patch_embed.forward == original
    assert "forward" not in vars(model.patch_embed)
    patch.remove()


@torch.inference_mode()
@pytest.mark.parametrize("grid", [
    [[1, 4, 4], [1, 4, 4]],
    [[4, 4, 4]],
    [[1, 4, 4], [1, 4, 8]],
])
def test_complete_visual_and_deepstack_outputs(grid):
    torch.manual_seed(19)
    model = _model()
    grid = torch.tensor(grid)
    pixels = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 4 * 4)
    expected = model(pixels, grid)
    with LinearVisionPatchEmbed(model) as patch:
        actual = model(pixels, grid)
        torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=1e-5)
        assert len(actual[1]) == len(expected[1])
        for a, e in zip(actual[1], expected[1]):
            torch.testing.assert_close(a, e, atol=2e-6, rtol=1e-5)
        assert patch.optimized_calls == 1
        with pytest.raises(ValueError, match="already installed"):
            LinearVisionPatchEmbed(model)


def test_training_fallback_and_restoration_of_existing_instance_method():
    model = _model().train()
    original = model.patch_embed.forward
    model.patch_embed.forward = original
    pixels = torch.randn(16, 3 * 2 * 4 * 4, requires_grad=True)
    with LinearVisionPatchEmbed(model) as patch:
        expected = original(pixels)
        actual = model.patch_embed(pixels)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        assert pixels.grad is not None
        assert patch.fallback_calls == 1 and patch.optimized_calls == 0
    assert "forward" in vars(model.patch_embed)
    assert model.patch_embed.forward == original


def test_exception_restores_and_unrelated_model_is_unchanged():
    model, unrelated = _model(), _model()
    original = unrelated.patch_embed.forward
    with pytest.raises(RuntimeError, match="deliberate"):
        with LinearVisionPatchEmbed(model):
            assert unrelated.patch_embed.forward == original
            raise RuntimeError("deliberate")
    assert "forward" not in vars(model.patch_embed)


def test_unsupported_models_and_geometry_fail_before_patching():
    with pytest.raises(ValueError, match="no Qwen3-VL"):
        LinearVisionPatchEmbed(torch.nn.Linear(4, 4))
    model = _model()
    model.patch_embed.proj.padding = (1, 1, 1)
    with pytest.raises(ValueError, match="required geometry"):
        LinearVisionPatchEmbed(model)
    assert "forward" not in vars(model.patch_embed)
