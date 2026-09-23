"""Differentiable write/read/update checks for the offline Spatial-TTT trainer."""

import torch

from test_spatial_model import settings, tiny_qwen
from ttt_frame.spatial_trainer import SpatialOfflineTrainer, SpatialTrainerConfig
from ttt_frame.spatial_videoqa import SpatialVideoConfig, SpatialVideoMemory


class TextProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return ""


def engine():
    model = tiny_qwen()
    return SpatialVideoMemory(
        SpatialVideoConfig(
            model_path="tiny-offline-qwen",
            device="cpu",
            dtype="float32",
            spatial=settings(use_conv=False, ttt_scale_init=0.3),
        ),
        model=model,
        processor=TextProcessor(),
    )


def batches():
    write = {
        "input_ids": torch.tensor([[5, 6, 7, 8]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    qa = {
        "input_ids": torch.tensor([[9, 10, 11, 12]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 13, 14]]),
    }
    return write, qa


def test_train_episode_updates_slow_parameters_and_resets_fast_state():
    current = engine()
    trainer = SpatialOfflineTrainer(current)
    write, qa = batches()
    before = {
        name: parameter.detach().clone()
        for name, parameter in current.controller.layers[next(iter(current.controller.layers))].named_parameters()
        if not name.startswith("attn_layer.")
    }
    report = trainer.train_episode([write], qa)
    assert report["supervised_tokens"] == 2
    assert report["written_tokens"] == 4
    assert report["loss"] > 0 and torch.isfinite(torch.tensor(report["loss"]))
    assert current.position_offset == 0
    assert all(layer.state is None for layer in current.controller.layers.values())
    after = {
        name: parameter.detach()
        for name, parameter in current.controller.layers[next(iter(current.controller.layers))].named_parameters()
        if not name.startswith("attn_layer.")
    }
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_trainer_rejects_unlabeled_episode():
    current = engine()
    trainer = SpatialOfflineTrainer(current, SpatialTrainerConfig(max_grad_norm=None))
    write, qa = batches()
    qa["labels"] = torch.full_like(qa["labels"], -100)
    try:
        trainer.train_episode([write], qa)
    except ValueError as error:
        assert "no supervised" in str(error)
    else:
        raise AssertionError("expected missing-supervision validation")


def test_video_write_positions_use_qwen_multimodal_rope_axes():
    current = engine()
    trainer = SpatialOfflineTrainer(current)
    input_ids = torch.tensor([[123, 126, 126, 126, 126, 124, 1, 2]])
    attention_mask = torch.ones_like(input_ids)
    grid = torch.tensor([[1, 4, 4]])
    expected, _ = current.model.model.get_rope_index(
        input_ids, video_grid_thw=grid, attention_mask=attention_mask
    )
    actual = trainer._write_position_ids(
        input_ids, video_grid=grid, attention_mask=attention_mask, offset=11
    )
    assert torch.equal(actual, expected + 11)
