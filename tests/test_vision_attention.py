"""Actual tiny Qwen vision models verify batching without cross-frame mixing."""

import pytest
import torch

from ttt_frame.vision_attention import EqualLengthVisionSDPA


def _model():
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLConfig(vision_config={
        "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
        "out_hidden_size": 32, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2, "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    }).vision_config
    config._attn_implementation = "sdpa"
    return Qwen3VLVisionModel(config).eval()


def _assert_output(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=1e-5)
    for a, e in zip(actual[1], expected[1]):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("grid", [
    [[1, 4, 4], [1, 4, 4]],  # two independent images
    [[4, 4, 4]],             # four independent temporal groups in one video
])
def test_complete_visual_tower_matches_original_and_restores(grid):
    torch.manual_seed(19)
    model = _model()
    grid = torch.tensor(grid)
    pixels = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 16 * 16)
    expected = model(pixels, grid)
    original = model.blocks[0].attn.forward
    with EqualLengthVisionSDPA(model) as patch:
        _assert_output(model(pixels, grid), expected)
        assert patch.optimized_calls == 2 and patch.fallback_calls == 0
        with pytest.raises(ValueError, match="already installed"):
            EqualLengthVisionSDPA(model)
    assert model.blocks[0].attn.forward == original
    assert "forward" not in vars(model.blocks[0].attn)
    _assert_output(model(pixels, grid), expected)
    patch.remove()


@torch.inference_mode()
def test_perturbing_later_group_does_not_change_first_group():
    torch.manual_seed(4)
    model = _model()
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    pixels = torch.randn(32, 3 * 2 * 16 * 16)
    changed = pixels.clone()
    changed[16:] = torch.randn_like(changed[16:]) * 5
    with EqualLengthVisionSDPA(model):
        original = model(pixels, grid)
        perturbed = model(changed, grid)
    torch.testing.assert_close(original[0][:4], perturbed[0][:4], rtol=0, atol=0)
    torch.testing.assert_close(original[1][0][:4], perturbed[1][0][:4], rtol=0, atol=0)
    assert not torch.allclose(original[0][4:], perturbed[0][4:])


@torch.inference_mode()
@pytest.mark.parametrize("training", [False, True])
def test_unequal_groups_or_training_use_original_forward(training):
    model = _model()
    model.train(training)
    grid = torch.tensor([[1, 4, 4], [1, 4, 8]])
    pixels = torch.randn(48, 3 * 2 * 16 * 16)
    expected = model(pixels, grid)
    with EqualLengthVisionSDPA(model) as patch:
        _assert_output(model(pixels, grid), expected)
        assert patch.optimized_calls == 0 and patch.fallback_calls == 2


def test_exception_restores_and_other_models_are_not_patched():
    model, unrelated = _model(), _model()
    before = unrelated.blocks[0].attn.forward
    with pytest.raises(RuntimeError, match="deliberate"):
        with EqualLengthVisionSDPA(model):
            assert unrelated.blocks[0].attn.forward == before
            raise RuntimeError("deliberate")
    assert "forward" not in vars(model.blocks[0].attn)
    with pytest.raises(ValueError, match="no Qwen3-VL"):
        EqualLengthVisionSDPA(torch.nn.Linear(4, 4))
