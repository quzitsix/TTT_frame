"""Final, past-only Idea 01 forecasts for a shared MCQ readout.

This is a batched inference implementation of ``forecast_residual``. It reuses
the train-fitted, frozen ``RidgeRecentPredictor`` and exactly the same manual
MSE update. Independent arms and horizons are batched; stream steps remain
sequential. No labels, answer choices, or future frames enter this interface.

The input is one observed episode prefix. All state is recreated on every call.
The final H x P x D forecasts are returned in the original feature space, on
CPU. There is no outer training of the memory initialization or key encoder.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_frame.forecast_residual import RidgeRecentPredictor


ARMS = (
    "persistence", "recent_only", "plain_delayed", "residual",
    "reset_residual", "nearest_bank", "recent_tail_residual",
)
FAST_ARMS = frozenset((
    "plain_delayed", "residual", "reset_residual", "recent_tail_residual",
))


@dataclass(frozen=True)
class FinalForecastConfig:
    hidden_dim: int = 128
    learning_rate: float = 0.05
    bank_capacity: int = 16
    seed: int = 0
    # At 2 fps, the usual 3660-step prefix has a 30-second tail from step 3600.
    # The tail arm drops earlier weights, pending cues, and recent-window data.
    tail_start_step: int = 3600

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "bank_capacity", "seed", "tail_start_step"):
            value = getattr(self, name)
            minimum = 0 if name in ("seed", "tail_start_step") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")


def _validate_sequence(sequence: Tensor, baseline: RidgeRecentPredictor) -> None:
    if (
        not isinstance(sequence, Tensor) or sequence.ndim != 3
        or not sequence.is_floating_point() or sequence.shape[0] < 1
        or not bool(torch.isfinite(sequence).all())
    ):
        raise ValueError("sequence must be finite floating point [steps, tokens, dim]")
    if tuple(sequence.shape[1:]) != (baseline.tokens, baseline.dim):
        raise ValueError("sequence tokens/dimension must match the frozen baseline")


def _batched_update(
    w_in: Tensor, w_out: Tensor, keys: Tensor, targets: Tensor,
    active: Tensor, learning_rate: float,
) -> None:
    """Update [arm, horizon] independent MLPs in place, using old W_out."""
    preactivation = keys @ w_in.transpose(-1, -2)
    hidden = F.silu(preactivation)
    prediction = hidden @ w_out.transpose(-1, -2)
    derivative = (2.0 * (prediction - targets) / targets.shape[-2:].numel()) * active
    grad_out = derivative.transpose(-1, -2) @ hidden
    sigma = torch.sigmoid(preactivation)
    silu_derivative = sigma * (1 + preactivation * (1 - sigma))
    grad_in = ((derivative @ w_out) * silu_derivative).transpose(-1, -2) @ keys
    # Separate operations preserve the reference update's rounding semantics.
    w_in.sub_(learning_rate * grad_in)
    w_out.sub_(learning_rate * grad_out)


@torch.no_grad()
def run_final_forecasts(
    sequence: Tensor,
    baseline: RidgeRecentPredictor,
    config: FinalForecastConfig | None = None,
    *,
    arms: tuple[str, ...] = ARMS,
    device: str | torch.device = "cpu",
    keep_trajectories: bool = False,
) -> dict[str, Any]:
    """Read an observed prefix and predict its still-unobserved future features.

    ``nearest_bank`` is the reference's ``explicit_bank``: nearest whole cue,
    independently per horizon, with oldest-first tie breaking. It is an
    explicit visual-memory control, not language-model ICL.

    ``reset_residual`` resets weights before each target arrival but retains
    the delayed FIFO. ``recent_tail_residual`` starts a new episode at the
    configured step; its cues and baseline windows cannot cross that boundary.
    If the boundary exceeds this prefix, that arm stays inactive and returns
    the recent-only prediction. ``keep_trajectories`` is intended for small
    parity/causality checks, since it stores every prediction.
    """
    config = config or FinalForecastConfig()
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError(f"arms must be distinct nonempty members of {ARMS}")
    _validate_sequence(sequence, baseline)
    device = torch.device(device)
    mean, std, coefficients = (
        value.detach().to(device=device, dtype=torch.float32)
        for value in (baseline.mean, baseline.std, baseline.coefficients)
    )
    z = (sequence.detach().to(device=device, dtype=torch.float32) - mean) / std
    if not bool(torch.isfinite(z).all()):
        raise ValueError("standardized sequence must remain finite")
    length, tokens, dim = z.shape
    horizons = baseline.horizons
    count, max_horizon = len(horizons), max(horizons)
    horizon_indices = torch.arange(count, device=device)
    horizon_values = torch.tensor(horizons, device=device)
    fast_arms = tuple(arm for arm in arms if arm in FAST_ARMS)
    fast_index = {arm: index for index, arm in enumerate(fast_arms)}
    initial_in = torch.randn(
        count, config.hidden_dim, dim,
        generator=torch.Generator().manual_seed(config.seed),
    ).div_(math.sqrt(dim)).to(device)
    w_in = initial_in.unsqueeze(0).repeat(len(fast_arms), 1, 1, 1)
    w_out = torch.zeros(len(fast_arms), count, dim, config.hidden_dim, device=device)
    # Rings are fixed size. Slots are read before today's schedule overwrites
    # them, including the longest horizon whose due slot is today's write slot.
    key_ring = torch.zeros(max_horizon, tokens, dim, device=device)
    base_ring = torch.zeros(max_horizon, count, tokens, dim, device=device)
    tail_ring = torch.zeros_like(base_ring) if "recent_tail_residual" in arms else None
    bank = {h: deque(maxlen=config.bank_capacity) for h in horizons}
    trajectory = {arm: [] for arm in arms} if keep_trajectories else None
    last: dict[str, Tensor] = {}

    def recent_forecast(step: int, start: int = 0) -> Tensor:
        recent = z[max(start, step - baseline.window_steps + 1):step + 1]
        design = torch.cat((recent[-1], recent.mean(dim=0), torch.ones_like(z[step, :, :1])), -1)
        return torch.einsum("pf,hfd->hpd", design, coefficients)

    for step in range(length):
        target = z[step]
        slots = (step - horizon_values) % max_horizon
        keys = key_ring[slots]
        previous_baseline = base_ring[slots, horizon_indices]
        if fast_arms:
            if "reset_residual" in fast_index:
                index = fast_index["reset_residual"]
                w_in[index].copy_(initial_in)
                w_out[index].zero_()
            targets = (target.unsqueeze(0) - previous_baseline).unsqueeze(0).repeat(
                len(fast_arms), 1, 1, 1
            )
            active = (horizon_values <= step).view(1, count, 1, 1).expand(
                len(fast_arms), -1, -1, -1
            ).clone()
            if "plain_delayed" in fast_index:
                targets[fast_index["plain_delayed"]] = target
            if "recent_tail_residual" in fast_index:
                index = fast_index["recent_tail_residual"]
                targets[index] = target - tail_ring[slots, horizon_indices]
                active[index] = (horizon_values <= step - config.tail_start_step).view(count, 1, 1)
            _batched_update(w_in, w_out, keys.unsqueeze(0), targets, active, config.learning_rate)
        if "nearest_bank" in arms:
            for index, horizon in enumerate(horizons):
                if step >= horizon:
                    bank[horizon].append((
                        keys[index].clone(), (target - previous_baseline[index]).clone(),
                    ))
        predicted_baseline = recent_forecast(step)
        predicted_tail = (
            recent_forecast(step, config.tail_start_step)
            if tail_ring is not None and step >= config.tail_start_step
            else predicted_baseline
        )
        slot = step % max_horizon
        key_ring[slot].copy_(target)
        base_ring[slot].copy_(predicted_baseline)
        if tail_ring is not None:
            tail_ring[slot].copy_(predicted_tail)
        # The write uses the observed target and stored frozen baseline, not
        # a previously emitted forecast. Intermediate reads may be omitted.
        if keep_trajectories or step == length - 1:
            fast_reads = F.silu(target @ w_in.transpose(-1, -2)) @ w_out.transpose(-1, -2)
            for arm in arms:
                if arm == "persistence":
                    prediction = target.unsqueeze(0).expand(count, -1, -1)
                elif arm == "recent_only":
                    prediction = predicted_baseline
                elif arm in fast_index:
                    prediction = fast_reads[fast_index[arm]]
                    if arm != "plain_delayed":
                        prediction = prediction + (predicted_tail if arm == "recent_tail_residual" else predicted_baseline)
                else:
                    per_horizon = []
                    for index, horizon in enumerate(horizons):
                        entries = bank[horizon]
                        if entries:
                            distances = torch.stack([(key - target).square().mean() for key, _ in entries])
                            residual = entries[int(distances.argmin())][1]
                            per_horizon.append(predicted_baseline[index] + residual)
                        else:
                            per_horizon.append(predicted_baseline[index])
                    prediction = torch.stack(per_horizon)
                last[arm] = (prediction * std + mean).detach().cpu()
                if trajectory is not None:
                    trajectory[arm].append(last[arm])
    if any(not bool(torch.isfinite(value).all()) for value in (*last.values(), w_in, w_out)):
        raise RuntimeError("nonfinite final forecast or fast weights; episode is invalid")
    pair_bytes = 2 * tokens * dim * 4
    fast_bytes = count * 2 * dim * config.hidden_dim * 4
    outputs = {}
    for arm in arms:
        seen_steps = max(0, length - config.tail_start_step) if arm == "recent_tail_residual" else length
        updates = sum(max(0, seen_steps - h) for h in horizons) if arm in FAST_ARMS or arm == "nearest_bank" else 0
        state_bytes = capacity_bytes = pending_bytes = 0
        if arm in FAST_ARMS:
            state_bytes = capacity_bytes = fast_bytes
            pending_bytes = pair_bytes * sum(min(h, seen_steps) for h in horizons)
        elif arm == "nearest_bank":
            state_bytes = pair_bytes * sum(len(entries) for entries in bank.values())
            capacity_bytes = pair_bytes * config.bank_capacity * count
            pending_bytes = pair_bytes * sum(min(h, length) for h in horizons)
        outputs[arm] = {
            "last_forecasts": last[arm], "updates": updates,
            "state_bytes": state_bytes, "state_capacity_bytes": capacity_bytes,
            "pending_bytes": pending_bytes,
        }
        if arm in fast_index:
            index = fast_index[arm]
            outputs[arm]["fast_weight_delta_l2"] = float(torch.sqrt(
                (w_in[index] - initial_in).square().sum() + w_out[index].square().sum()
            ))
        if trajectory is not None:
            outputs[arm]["predictions"] = torch.stack(trajectory[arm])
    return {
        "schema": "idea01_final_forecasts_v1", "sequence_steps": length,
        "feature_shape": [tokens, dim], "horizons": list(horizons),
        "tail_start_step": config.tail_start_step,
        "outer_training_performed": False, "explicit_bank_is_vlm_icl": False,
        "resident_input_feature_cache_bytes": z.numel() * z.element_size(),
        "shared_fifo_allocation_bytes": sum(
            value.numel() * value.element_size() for value in (key_ring, base_ring)
        ) + (tail_ring.numel() * tail_ring.element_size() if tail_ring is not None else 0),
        "frozen_recent_predictor_bytes": sum(
            value.numel() * value.element_size() for value in (mean, std, coefficients)
        ),
        "state_byte_scope": "per-arm logical state; shared physical FIFO allocation reported separately; input cache excluded",
        "reset_policy": "reset weights before each maturity; retain delayed cue FIFO",
        "tail_policy": "discard all cues, fast weights and recent-window data before tail_start_step",
        "final_forecasts_scored": False, "arms": outputs,
    }


__all__ = ["ARMS", "FinalForecastConfig", "RidgeRecentPredictor", "run_final_forecasts"]
