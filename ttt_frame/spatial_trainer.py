"""Small offline teacher-forced trainer for the Spatial-TTT rule.

The public video path in :mod:`ttt_frame.spatial_videoqa` is deliberately a
``no_grad`` test-time writer.  This module is a separate, differentiable path
for experiments that train the Spatial-TTT slow parameters from question and
answer labels.  A training episode is streamed as

``video/write batches -> text/read batch -> cross entropy -> optimizer``.

Only parameters added by ``SpatialQwenMemory`` are optimized by default.  The
base Qwen model stays frozen, and the fast state is reset after every episode
so a graph from one episode cannot leak into the next one.  The caller supplies
already processed batches; this keeps the trainer independent of a particular
video sampler or teacher format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class SpatialTrainerConfig:
    """Optimization settings for :class:`SpatialOfflineTrainer`."""

    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    train_readout_only: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or not torch.isfinite(torch.tensor(self.learning_rate)):
            raise ValueError("learning_rate must be finite and positive")
        if self.weight_decay < 0 or not torch.isfinite(torch.tensor(self.weight_decay)):
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.max_grad_norm is not None and (
            self.max_grad_norm <= 0 or not torch.isfinite(torch.tensor(self.max_grad_norm))
        ):
            raise ValueError("max_grad_norm must be None or finite and positive")


def _tensor_batch(batch: Mapping[str, object], *, labels: bool = False) -> dict[str, Tensor]:
    """Select tensor model inputs and reject accidental non-tensor payloads."""

    result = {key: value for key, value in batch.items() if isinstance(value, Tensor)}
    if "input_ids" not in result:
        raise ValueError("batch must contain tensor input_ids")
    if labels and "labels" not in result:
        raise ValueError("qa batch must contain tensor labels")
    return result


class SpatialOfflineTrainer:
    """Train Spatial-TTT slow parameters with a teacher-forced QA loss.

    ``engine`` is a ``SpatialVideoMemory`` instance.  ``write_batches`` are
    processor outputs for one chronological episode and ``qa_batch`` contains
    ``input_ids``, ``attention_mask`` (optional), and same-shaped ``labels``.
    Label positions set to ``-100`` are ignored, as in Transformers.  Video
    batches may include ``video_mask``/``video_grid_thw``; all other tensor
    fields are passed to the model unchanged.

    This intentionally does not call ``ingest_frames``: that method is
    inference-only and decorated with ``torch.no_grad``.  The stateful write
    forward below preserves the autograd graph through fast-weight updates.
    """

    def __init__(self, engine, config: SpatialTrainerConfig | None = None, *, optimizer=None):
        self.engine = engine
        self.config = config or SpatialTrainerConfig()
        self.controller = engine.controller
        self.model = engine.model
        self._parameters = []
        for layer in self.controller.layers.values():
            for name, parameter in layer.named_parameters():
                # The original Qwen attention is frozen; only Spatial-TTT
                # read/write parameters are trainable in this minimal trainer.
                if name.startswith("attn_layer."):
                    parameter.requires_grad_(False)
                    continue
                if self.config.train_readout_only and not (
                    name.startswith("memory.ttt_scale_proj.")
                    or name.startswith("memory.ttt_norm.")
                ):
                    parameter.requires_grad_(False)
                    continue
                parameter.requires_grad_(True)
                self._parameters.append(parameter)
        if not self._parameters:
            raise ValueError("Spatial-TTT has no trainable parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            self._parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _position_ids(self, input_ids: Tensor, offset: int) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        length = input_ids.shape[1]
        positions = torch.arange(
            offset, offset + length, device=input_ids.device, dtype=torch.long
        )
        return positions[None, None, :].expand(3, input_ids.shape[0], length)

    def _write_position_ids(
        self, input_ids: Tensor, *, video_grid: Tensor | None,
        attention_mask: Tensor | None, offset: int,
    ) -> Tensor:
        """Build Qwen multimodal RoPE positions for a write batch.

        Video tokens use temporal/height/width positions from Qwen's own
        ``get_rope_index``.  A plain arange is valid only for text-only mock
        batches and would silently scramble the visual RoPE axes.
        """
        if video_grid is None:
            return self._position_ids(input_ids, offset)
        get_rope_index = getattr(getattr(self.model, "model", None), "get_rope_index", None)
        if get_rope_index is None:
            raise ValueError("model does not provide get_rope_index for video writes")
        positions, _ = get_rope_index(
            input_ids,
            video_grid_thw=video_grid,
            attention_mask=attention_mask,
        )
        return positions + offset

    def _write_one(self, batch: Mapping[str, object], *, flush: bool) -> int:
        tensors = _tensor_batch(batch)
        tensors = self.engine._move(tensors)
        input_ids = tensors["input_ids"]
        video_mask = tensors.get("video_mask")
        video_grid = tensors.get("video_grid_thw")
        if video_mask is None and video_grid is not None:
            token_id = getattr(self.model.config, "video_token_id", None)
            if token_id is not None:
                video_mask = input_ids.eq(token_id)
        # ``video_mask`` and ``video_grid_thw`` are control metadata rather
        # than model kwargs.  They are consumed by SpatialAttention._conv.
        model_batch = {
            key: value
            for key, value in tensors.items()
            if key not in {"labels", "video_mask", "position_ids"}
        }
        positions = tensors.get("position_ids")
        if positions is None:
            positions = self._write_position_ids(
                input_ids,
                video_grid=video_grid,
                attention_mask=tensors.get("attention_mask"),
                offset=self.engine.position_offset,
            )
        else:
            if not isinstance(positions, Tensor) or positions.shape != (3, *input_ids.shape):
                raise ValueError("position_ids must have shape [3, batch, sequence]")
            positions = positions.to(device=input_ids.device, dtype=torch.long)
        with self.controller.context(
            "write", video_mask=video_mask, video_grid_thw=video_grid, flush=flush
        ):
            self.model(
                **model_batch,
                position_ids=positions,
                use_cache=False,
                logits_to_keep=1,
            )
        self.engine.position_offset = int(positions.max().detach().item()) + 1
        return int(input_ids.shape[1])

    def _read_loss(self, qa_batch: Mapping[str, object]) -> tuple[Tensor, int]:
        tensors = _tensor_batch(qa_batch, labels=True)
        tensors = self.engine._move(tensors)
        input_ids, labels = tensors["input_ids"], tensors["labels"]
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        model_batch = {
            key: value
            for key, value in tensors.items()
            if key not in {"labels", "position_ids", "video_mask"}
        }
        positions = tensors.get("position_ids")
        if positions is None:
            positions = self._position_ids(input_ids, self.engine.position_offset)
        elif not isinstance(positions, Tensor) or positions.shape != (3, *input_ids.shape):
            raise ValueError("position_ids must have shape [3, batch, sequence]")
        with self.controller.context("read"):
            outputs = self.model(
                **model_batch,
                position_ids=positions.to(device=input_ids.device, dtype=torch.long),
                use_cache=False,
            )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if logits.shape[:2] != labels.shape:
            raise ValueError("model logits and labels have incompatible shapes")
        # Standard causal teacher forcing: token t predicts label t+1.
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        count = int((labels[:, 1:] != -100).sum().item())
        if count == 0:
            raise ValueError("qa labels contain no supervised completion tokens")
        return loss, count

    def train_episode(
        self,
        write_batches: Iterable[Mapping[str, object]],
        qa_batch: Mapping[str, object],
    ) -> dict[str, float | int]:
        """Run one episode and update the selected Spatial-TTT parameters.

        The episode always starts from an empty fast state.  All write batches
        must be chronological.  The final batch is flushed before the QA read,
        which enforces the same pending-token invariant as ``finish_ingest``.
        The state is reset after ``optimizer.step`` and therefore cannot be
        reused as an inference memory; call the normal online ingestion path
        for that use case.
        """

        batches = list(write_batches)
        if not batches:
            raise ValueError("write_batches must contain at least one batch")
        self.controller.reset()
        self.engine.position_offset = 0
        self.engine._clear_generation_state()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            # The public video path is often called from an inference context;
            # explicitly re-enable autograd for this separate training API.
            with torch.enable_grad():
                written_tokens = 0
                for index, batch in enumerate(batches):
                    written_tokens += self._write_one(batch, flush=index == len(batches) - 1)
                if any(layer.state is None for layer in self.controller.layers.values()):
                    raise RuntimeError("Spatial-TTT write did not initialize every selected layer")
                loss, supervised_tokens = self._read_loss(qa_batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite Spatial-TTT QA loss")
                loss.backward()
                gradients = [parameter.grad for parameter in self._parameters
                             if parameter.grad is not None]
                if not gradients:
                    raise RuntimeError("teacher-forced loss produced no Spatial-TTT gradients")
                if self.config.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._parameters, self.config.max_grad_norm
                    )
                else:
                    grad_norm = torch.linalg.vector_norm(
                        torch.stack([gradient.detach().norm() for gradient in gradients])
                    )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("nonfinite Spatial-TTT gradient norm")
                self.optimizer.step()
                return {
                    "loss": float(loss.detach().cpu()),
                    "supervised_tokens": supervised_tokens,
                    "written_tokens": written_tokens,
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
        finally:
            # Never leave a graph-bearing episode state behind, including on a
            # malformed batch or a failed backward pass.
            self.optimizer.zero_grad(set_to_none=True)
            self.controller.reset()
            self.engine.position_offset = 0
            self.engine._clear_generation_state()


__all__ = ["SpatialOfflineTrainer", "SpatialTrainerConfig"]
