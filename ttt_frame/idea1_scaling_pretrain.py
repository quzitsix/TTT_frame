"""Train-only delayed prediction initialization for Idea 01 scaling studies.

This is ordinary offline supervised regression on *observed training prefixes*.
It is not MAML, an outer gradient through online updates, or additional MCQ
annotation. The returned weights initialize independent per-episode memories.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_frame.forecast_residual import RidgeRecentPredictor


TARGETS = ("plain", "residual")


@dataclass(frozen=True)
class OfflinePretrainConfig:
    hidden_dim: int = 128
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "batch_size", "seed"):
            minimum = 0 if name == "seed" else 1
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("learning_rate", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.weight_decay, bool) or not math.isfinite(self.weight_decay)
                or self.weight_decay < 0):
            raise ValueError("weight_decay must be finite and nonnegative")


class FrozenPairDataset:
    """Within-case causal visual cue/outcome pairs, with a frozen recent model.

    The caller supplies only a declared role's past-prefix tensors. No files,
    choices or labels are accepted. Origins are chosen uniformly by case and
    uniformly within that case; all horizons share the same origin. Sequences
    can have different lengths and targets never cross an episode boundary.
    """

    def __init__(self, sequences: list[Tensor], baseline: RidgeRecentPredictor,
                 *, device: str | torch.device = "cpu") -> None:
        if not sequences:
            raise ValueError("need at least one observed sequence")
        self.device = torch.device(device)
        self.horizons = tuple(baseline.horizons)
        self.max_horizon = max(self.horizons)
        self.window_steps = baseline.window_steps
        self.tokens, self.dim = baseline.tokens, baseline.dim
        self.mean, self.std, self.coefficients = (
            value.detach().to(device=self.device, dtype=torch.float32).clone()
            for value in (baseline.mean, baseline.std, baseline.coefficients)
        )
        self.streams = []
        self.origins_per_case = []
        for sequence in sequences:
            if (not isinstance(sequence, Tensor) or sequence.ndim != 3
                    or not sequence.is_floating_point()
                    or sequence.shape[0] <= self.max_horizon
                    or tuple(sequence.shape[1:]) != (self.tokens, self.dim)
                    or not bool(torch.isfinite(sequence).all())):
                raise ValueError("sequences must be finite [steps>max_horizon,tokens,dim]")
            z = (sequence.detach().to(self.device, torch.float32) - self.mean) / self.std
            if not bool(torch.isfinite(z).all()):
                raise ValueError("nonfinite standardized training features")
            self.streams.append(z)
            self.origins_per_case.append(sequence.shape[0] - self.max_horizon)
        self.equal_lengths = len(set(stream.shape[0] for stream in self.streams)) == 1
        self.stacked = torch.stack(self.streams) if self.equal_lengths else None
        if self.stacked is not None:
            # Views retain only one physical allocation after this replacement.
            self.streams = list(self.stacked.unbind(dim=0))
        self.horizon_tensor = torch.tensor(self.horizons, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def gather(self, case_indices: Tensor, origins: Tensor) -> dict[str, Tensor]:
        cases, steps = case_indices.detach().cpu().long(), origins.detach().cpu().long()
        if cases.ndim != 1 or steps.shape != cases.shape or cases.numel() < 1:
            raise ValueError("case indices/origins must be equal nonempty vectors")
        if not bool(((cases >= 0) & (cases < len(self.streams))).all()):
            raise ValueError("case index out of bounds")
        limits = torch.tensor(self.origins_per_case)[cases]
        if not bool(((steps >= 0) & (steps < limits)).all()):
            raise ValueError("origin would cross the sequence cutoff")
        indices, offsets = cases.to(self.device), steps.to(self.device)
        if self.stacked is not None:
            keys = self.stacked[indices, offsets]
            outcomes = self.stacked[indices[:, None], offsets[:, None] + self.horizon_tensor]
            recent = torch.zeros_like(keys)
            for back in range(self.window_steps):
                valid = offsets >= back
                recent.add_(self.stacked[indices, (offsets - back).clamp_min(0)] * valid[:, None, None])
            recent.div_((offsets + 1).clamp_max(self.window_steps)[:, None, None])
        else:
            keys = torch.stack([self.streams[int(case)][int(step)] for case, step in zip(cases, steps)])
            outcomes = torch.stack([self.streams[int(case)][int(step) + self.horizon_tensor]
                                    for case, step in zip(cases, steps)])
            recent = torch.stack([self.streams[int(case)][max(0, int(step) - self.window_steps + 1):int(step) + 1].mean(0)
                                  for case, step in zip(cases, steps)])
        design = torch.cat((keys, recent, torch.ones_like(keys[..., :1])), dim=-1)
        recent_predictions = torch.einsum("bpf,hfd->hbpd", design, self.coefficients)
        plain = outcomes.permute(1, 0, 2, 3)
        targets = torch.stack((plain, plain - recent_predictions), dim=0)
        return {"keys": keys, "targets": targets, "case_indices": cases, "origins": steps}

    def sample(self, batch_size: int, generator: torch.Generator) -> dict[str, Tensor]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        cases = torch.randint(len(self.streams), (batch_size,), generator=generator)
        limits = torch.tensor(self.origins_per_case)[cases]
        steps = (torch.rand(batch_size, generator=generator) * limits).long()
        return self.gather(cases, steps)


class OfflineForecastMLPs(nn.Module):
    """Independent plain/residual MLPs for each horizon, trained in one batch."""

    def __init__(self, horizon_count: int, dim: int, config: OfflinePretrainConfig,
                 *, device: str | torch.device = "cpu") -> None:
        super().__init__()
        if horizon_count < 1 or dim < 1:
            raise ValueError("positive horizon_count and feature dimension required")
        initial = torch.randn(horizon_count, config.hidden_dim, dim,
                              generator=torch.Generator().manual_seed(config.seed)).div_(math.sqrt(dim))
        self.w_in = nn.Parameter(initial.unsqueeze(0).repeat(2, 1, 1, 1).to(device))
        self.w_out = nn.Parameter(torch.zeros(2, horizon_count, dim, config.hidden_dim, device=device))
        self.gain = math.sqrt(128 / config.hidden_dim)

    def forward(self, keys: Tensor) -> Tensor:
        if keys.ndim != 3 or keys.shape[-1] != self.w_in.shape[-1]:
            raise ValueError("keys must be [batch,tokens,dim]")
        batch, tokens, dim = keys.shape
        preactivation = keys.reshape(1, 1, batch * tokens, dim) @ self.w_in.transpose(-1, -2)
        prediction = (F.silu(preactivation) * self.gain) @ self.w_out.transpose(-1, -2)
        return prediction.reshape(2, self.w_in.shape[1], batch, tokens, dim)

    def initial_weights(self) -> dict[str, dict[str, Tensor]]:
        return {name: {"w_in": self.w_in[index].detach().cpu().clone(),
                       "w_out": self.w_out[index].detach().cpu().clone()}
                for index, name in enumerate(TARGETS)}


def train_step(model: OfflineForecastMLPs, optimizer: torch.optim.Optimizer,
               batch: dict[str, Tensor], config: OfflinePretrainConfig) -> dict[str, object]:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch["keys"])
    targets = batch["targets"].detach()
    if targets.shape != prediction.shape:
        raise ValueError("target shape does not match independent arm/horizon predictions")
    losses = (prediction - targets).square().mean(dim=(2, 3, 4))
    # Sum independent model losses, avoiding an arm-count/horizon-count change
    # in the gradient convention as the model family is batched.
    loss = losses.sum()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite offline training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {"mse_by_arm_horizon": losses.detach().cpu().tolist(),
            "gradient_norm_before_clip": float(norm.detach())}


@torch.no_grad()
def validation_losses(model: OfflineForecastMLPs, batch: dict[str, Tensor]) -> dict[str, object]:
    losses = (model(batch["keys"]) - batch["targets"]).square().mean(dim=(2, 3, 4))
    if not bool(torch.isfinite(losses).all()):
        raise RuntimeError("nonfinite offline validation loss")
    return {name: {"mse": float(losses[index].mean()), "by_horizon": losses[index].cpu().tolist()}
            for index, name in enumerate(TARGETS)}
