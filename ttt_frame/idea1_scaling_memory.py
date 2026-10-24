"""Width/training-budget controls for causal Idea 01 visual memory.

The frozen encoder and recent predictor live outside this module. This module
accepts an observed prefix only; all cues are written after their target has
arrived. Optional offline weights are copied into independent episode states.
No labels, choices, test-future tensors or optimizer moments are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_frame.forecast_residual import RidgeRecentPredictor


ARMS = (
    "persistence", "recent_only", "plain_delayed", "residual", "reset_residual",
    "recent_tail_residual", "residual_no_update", "plain_no_update",
)
FAST_ARMS = frozenset((
    "plain_delayed", "residual", "reset_residual", "recent_tail_residual",
))
MLP_ARMS = FAST_ARMS | {"residual_no_update", "plain_no_update"}


@dataclass(frozen=True)
class ScalingMemoryConfig:
    hidden_dim: int = 128
    learning_rate: float = 0.05
    inner_steps: int = 1
    seed: int = 0
    tail_start_step: int = 3600
    gain_mode: str = "normalized"
    # Score predictions issued in the final N observed steps after their
    # actual targets arrive. Zero disables this diagnostic and its read cost.
    score_last_steps: int = 60

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "inner_steps", "seed", "tail_start_step", "score_last_steps"):
            value = getattr(self, name)
            minimum = 0 if name in ("seed", "tail_start_step", "score_last_steps") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate)
                or self.learning_rate < 0):
            raise ValueError("learning_rate must be finite and nonnegative")
        if self.gain_mode not in ("normalized", "reference"):
            raise ValueError("gain_mode must be normalized or reference")

    @property
    def output_gain(self) -> float:
        return math.sqrt(128 / self.hidden_dim) if self.gain_mode == "normalized" else 1.0


def initial_fast_weights(horizons_count: int, dim: int, hidden_dim: int, seed: int = 0) -> dict[str, Tensor]:
    """The historical random-A / zero-B initialization, allocated on CPU."""
    for value in (horizons_count, dim, hidden_dim):
        if type(value) is not int or value < 1:
            raise ValueError("positive integer dimensions required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {
        "w_in": torch.randn(horizons_count, hidden_dim, dim,
                            generator=torch.Generator().manual_seed(seed)).div_(math.sqrt(dim)),
        "w_out": torch.zeros(horizons_count, dim, hidden_dim),
    }


def _validate_weights(value: dict, *, count: int, dim: int, hidden: int) -> dict[str, Tensor]:
    expected = {"w_in": (count, hidden, dim), "w_out": (count, dim, hidden)}
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("each initialization must contain only w_in and w_out")
    result = {}
    for name, shape in expected.items():
        tensor = value[name]
        if (not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape
                or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"invalid finite initialization shape for {name}: expected {shape}")
        result[name] = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
    return result


def _batched_update(
    w_in: Tensor, w_out: Tensor, keys: Tensor, targets: Tensor,
    active: Tensor, learning_rate: float, output_gain: float,
) -> None:
    """One exact MSE gradient step per active arm/horizon, using old B."""
    preactivation = keys @ w_in.transpose(-1, -2)
    hidden = F.silu(preactivation) * output_gain
    prediction = hidden @ w_out.transpose(-1, -2)
    derivative = (2.0 * (prediction - targets) / targets.shape[-2:].numel()) * active
    grad_out = derivative.transpose(-1, -2) @ hidden
    sigma = torch.sigmoid(preactivation)
    silu_derivative = sigma * (1 + preactivation * (1 - sigma))
    grad_in = (((derivative @ w_out) * output_gain) * silu_derivative).transpose(-1, -2) @ keys
    w_in.sub_(learning_rate * grad_in)
    w_out.sub_(learning_rate * grad_out)


@torch.no_grad()
def run_scaling_forecasts_batch(
    sequences: Tensor,
    baseline: RidgeRecentPredictor,
    config: ScalingMemoryConfig | None = None,
    *,
    arms: tuple[str, ...] = ARMS,
    initial_weights: dict[str, dict[str, Tensor]] | None = None,
    device: str | torch.device = "cpu",
    keep_trajectories: bool = False,
    return_states: bool = False,
) -> list[dict[str, Any]]:
    """Run an independent, causal episode and return final visual forecasts.

    Initializations are ``{'plain_delayed': {'w_in': [H,M,D], 'w_out': [H,D,M]},
    'residual': ...}``; ``plain`` aliases ``plain_delayed``. Missing kinds use
    random-A/zero-B. Frozen controls read exactly the corresponding initial
    weights. Reset and tail controls start from the same residual init.

    Width-normalized mode multiplies hidden activations by sqrt(128/M); thus
    merely adding neurons does not multiply the initial B-gradient response
    at a fixed learning rate. Reference mode omits that gain. At M=128, both
    modes reproduce the previous single-step update. Multiple inner steps use
    the same already-matured pairs, with gradients recomputed every time.

    ``mature_mse`` compares forecasts issued before targets arrive, and is
    distinct from post-update reconstruction error. Final future forecasts are
    returned but never scored, since this function has no future input.
    """
    config = config or ScalingMemoryConfig()
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError(f"arms must be distinct nonempty members of {ARMS}")
    if (not isinstance(sequences, Tensor) or sequences.ndim != 4
            or not sequences.is_floating_point() or min(sequences.shape[:2]) < 1
            or tuple(sequences.shape[2:]) != (baseline.tokens, baseline.dim)
            or not bool(torch.isfinite(sequences).all())):
        raise ValueError("sequences must be finite floating point [batch,steps,tokens,dim] matching baseline")
    device = torch.device(device)
    mean, std, coefficients = [
        tensor.detach().to(device=device, dtype=torch.float32)
        for tensor in (baseline.mean, baseline.std, baseline.coefficients)
    ]
    if (not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all())
            or not bool((std > 0).all()) or not bool(torch.isfinite(coefficients).all())):
        raise ValueError("frozen predictor must have finite coefficients and positive scales")
    z = (sequences.detach().to(device=device, dtype=torch.float32) - mean) / std
    if not bool(torch.isfinite(z).all()):
        raise ValueError("standardized sequence must remain finite")
    batch_count, length, tokens, dim = z.shape
    horizons = tuple(baseline.horizons)
    count, max_horizon = len(horizons), max(horizons)
    h_values = torch.tensor(horizons, device=device)
    h_indices = torch.arange(count, device=device)
    random_init = initial_fast_weights(count, dim, config.hidden_dim, config.seed)
    initial_weights = {} if initial_weights is None else initial_weights
    if not isinstance(initial_weights, dict) or not set(initial_weights) <= {"plain", "plain_delayed", "residual"}:
        raise ValueError("initial_weights accepts only plain_delayed/plain and residual")
    if "plain" in initial_weights and "plain_delayed" in initial_weights:
        raise ValueError("supply only one of plain and plain_delayed")
    initial = {}
    for kind in ("plain_delayed", "residual"):
        value = initial_weights.get(kind, initial_weights.get("plain") if kind == "plain_delayed" else None)
        initial[kind] = _validate_weights(random_init if value is None else value,
                                          count=count, dim=dim, hidden=config.hidden_dim)
    # Frozen controls never join the update batch. Their shared initial weights
    # are read only at scored origins and the final cutoff, saving two MLP
    # backward passes per arrival and avoiding per-episode frozen copies.
    mlp_arms = tuple(arm for arm in arms if arm in FAST_ARMS)
    mlp_index = {arm: index for index, arm in enumerate(mlp_arms)}
    arm_kind = {arm: "plain_delayed" if arm in ("plain_delayed", "plain_no_update") else "residual"
                for arm in arms if arm in MLP_ARMS}
    frozen_initial = {kind: {name: value.to(device) for name, value in state.items()}
                      for kind, state in initial.items()}
    if mlp_arms:
        initial_in = torch.stack([initial[arm_kind[arm]]["w_in"] for arm in mlp_arms]).to(device)
        initial_out = torch.stack([initial[arm_kind[arm]]["w_out"] for arm in mlp_arms]).to(device)
    else:
        initial_in = torch.empty(0, count, config.hidden_dim, dim, device=device)
        initial_out = torch.empty(0, count, dim, config.hidden_dim, device=device)
    w_in = initial_in.unsqueeze(0).repeat(batch_count, 1, 1, 1, 1)
    w_out = initial_out.unsqueeze(0).repeat(batch_count, 1, 1, 1, 1)
    key_ring = torch.zeros(max_horizon, batch_count, tokens, dim, device=device)
    base_ring = torch.zeros(max_horizon, batch_count, count, tokens, dim, device=device)
    tail_ring = torch.zeros_like(base_ring) if "recent_tail_residual" in arms else None
    score_start = max(0, length - config.score_last_steps)
    score_enabled = config.score_last_steps > 0
    prediction_ring = torch.zeros(max_horizon, batch_count, len(arms), count, tokens, dim, device=device) if score_enabled else None
    score_sums = torch.zeros(batch_count, len(arms), count, device=device, dtype=torch.float64)
    score_counts = torch.zeros(count, dtype=torch.int64)
    trajectories = {arm: [] for arm in arms} if keep_trajectories else None
    last = {}
    gain = config.output_gain

    def recent_forecast(step: int, start: int = 0) -> Tensor:
        recent = z[:, max(start, step - baseline.window_steps + 1):step + 1]
        design = torch.cat((recent[:, -1], recent.mean(dim=1), torch.ones_like(z[:, step, :, :1])), -1)
        return torch.einsum("npf,hfd->nhpd", design, coefficients)

    for step in range(length):
        target = z[:, step]
        slots = (step - h_values) % max_horizon
        keys = key_ring[slots].transpose(0, 1)
        previous_baseline = base_ring[slots, :, h_indices].transpose(0, 1)
        # Score the original prediction before today changes fast weights.
        if score_enabled and step > score_start:
            for index, horizon in enumerate(horizons):
                origin = step - horizon
                if origin >= score_start:
                    saved = prediction_ring[origin % max_horizon, :, :, index]
                    score_sums[:, :, index].add_((saved.double() - target[:, None]).square().mean(dim=(-2, -1)))
                    score_counts[index] += 1
        if mlp_arms:
            if "reset_residual" in mlp_index:
                index = mlp_index["reset_residual"]
                w_in[:, index].copy_(initial_in[index])
                w_out[:, index].copy_(initial_out[index])
            targets = (target[:, None] - previous_baseline)[:, None].repeat(1, len(mlp_arms), 1, 1, 1)
            active = (h_values <= step).view(1, 1, count, 1, 1).expand(batch_count, len(mlp_arms), -1, -1, -1).clone()
            for arm in mlp_arms:
                index = mlp_index[arm]
                if arm == "plain_delayed":
                    targets[:, index] = target[:, None]
                elif arm == "recent_tail_residual":
                    targets[:, index] = target[:, None] - tail_ring[slots, :, h_indices].transpose(0, 1)
                    active[:, index] = (h_values <= step - config.tail_start_step).view(count, 1, 1)
            for _ in range(config.inner_steps):
                _batched_update(w_in, w_out, keys.unsqueeze(1), targets, active,
                                config.learning_rate, gain)
        predicted_baseline = recent_forecast(step)
        predicted_tail = (recent_forecast(step, config.tail_start_step)
                          if tail_ring is not None and step >= config.tail_start_step else predicted_baseline)
        slot = step % max_horizon
        key_ring[slot].copy_(target)
        base_ring[slot].copy_(predicted_baseline)
        if tail_ring is not None:
            tail_ring[slot].copy_(predicted_tail)
        needs_read = keep_trajectories or step == length - 1 or (score_enabled and step >= score_start)
        if needs_read:
            fast_reads = (F.silu(target[:, None, None] @ w_in.transpose(-1, -2)) * gain) @ w_out.transpose(-1, -2)
            current = []
            for arm in arms:
                if arm == "persistence":
                    prediction = target.unsqueeze(1).expand(-1, count, -1, -1)
                elif arm == "recent_only":
                    prediction = predicted_baseline
                elif arm in mlp_index:
                    prediction = fast_reads[:, mlp_index[arm]]
                    if arm_kind[arm] == "residual":
                        prediction = prediction + (predicted_tail if arm == "recent_tail_residual" else predicted_baseline)
                else:
                    init = frozen_initial[arm_kind[arm]]
                    hidden = F.silu(target[:, None] @ init["w_in"].transpose(-1, -2)) * gain
                    prediction = hidden @ init["w_out"].transpose(-1, -2)
                    if arm_kind[arm] == "residual":
                        prediction = prediction + predicted_baseline
                current.append(prediction)
                if step == length - 1:
                    last[arm] = (prediction * std + mean).detach().cpu()
                if trajectories is not None:
                    trajectories[arm].append((prediction * std + mean).detach().cpu())
            if score_enabled and step >= score_start:
                prediction_ring[slot].copy_(torch.stack(current, dim=1))
    if any(not bool(torch.isfinite(value).all()) for value in (*last.values(), w_in, w_out, score_sums)):
        raise RuntimeError("nonfinite fast state or forecast; episode is invalid")
    outputs = [dict() for _ in range(batch_count)]
    pair_bytes = 2 * tokens * dim * 4
    fast_bytes = count * 2 * dim * config.hidden_dim * 4
    scores_cpu = score_sums.cpu()
    for arm_index, arm in enumerate(arms):
        seen_steps = max(0, length - config.tail_start_step) if arm == "recent_tail_residual" else length
        matured = sum(max(0, seen_steps - horizon) for horizon in horizons) if arm in FAST_ARMS else 0
        score_n = int(score_counts.sum())
        for batch_index in range(batch_count):
            item = {
            "last_forecasts": last[arm][batch_index], "updates": matured,
            "gradient_steps": matured * config.inner_steps,
            "state_bytes": fast_bytes if arm in MLP_ARMS else 0,
            "state_capacity_bytes": fast_bytes if arm in MLP_ARMS else 0,
            "pending_bytes": pair_bytes * sum(min(h, seen_steps) for h in horizons) if arm in FAST_ARMS else 0,
            "mature_mse": float(scores_cpu[batch_index, arm_index].sum() / score_n) if score_n else None,
            "mature_count": score_n,
            "mature_mse_by_horizon": {
                str(horizon): (float(scores_cpu[batch_index, arm_index, index] / score_counts[index])
                               if score_counts[index] else None)
                for index, horizon in enumerate(horizons)
            },
            }
            if arm in mlp_index:
                index = mlp_index[arm]
                item["fast_weight_delta_l2"] = float(torch.sqrt(
                    (w_in[batch_index, index] - initial_in[index]).square().sum()
                    + (w_out[batch_index, index] - initial_out[index]).square().sum()))
                if return_states:
                    item["state"] = {"w_in": w_in[batch_index, index].detach().cpu().clone(),
                                     "w_out": w_out[batch_index, index].detach().cpu().clone()}
            elif arm in MLP_ARMS:
                item["fast_weight_delta_l2"] = 0.0
                if return_states:
                    item["state"] = {name: value.clone() for name, value in initial[arm_kind[arm]].items()}
            if trajectories is not None:
                item["predictions"] = torch.stack(trajectories[arm])[:, batch_index]
            outputs[batch_index][arm] = item
    metadata = {
        "schema": "idea01_scaling_forecasts_v1", "sequence_steps": length,
        "feature_shape": [tokens, dim], "horizons": list(horizons),
        "hidden_dim": config.hidden_dim, "inner_steps": config.inner_steps,
        "learning_rate": config.learning_rate, "width_gain_mode": config.gain_mode,
        "width_output_gain": gain, "tail_start_step": config.tail_start_step,
        "score_last_steps": config.score_last_steps,
        "offline_initialization_supplied": bool(initial_weights),
        "outer_training_performed": False, "final_forecasts_scored": False,
        "resident_input_feature_cache_bytes": z[0].numel() * z.element_size(),
        "shared_fifo_allocation_bytes": (sum(value.numel() * value.element_size()
                                             for value in (key_ring, base_ring))
                                            + (tail_ring.numel() * tail_ring.element_size() if tail_ring is not None else 0)) // batch_count,
        "scoring_prediction_fifo_bytes": (prediction_ring.numel() * prediction_ring.element_size()
                                          if prediction_ring is not None else 0) // batch_count,
        "frozen_recent_predictor_bytes": sum(value.numel() * value.element_size()
                                               for value in (mean, std, coefficients)),
        "state_byte_scope": "per-arm logical weights/FIFO; physical shared rings and offline input cache reported separately",
        "reset_policy": "reset weights to supplied initialization before each maturity; retain cue FIFO",
        "tail_policy": "start a new episode from supplied initialization at tail_start_step",
        "mse_scope": "standardized observed targets, prediction issued before arrival; final unobserved targets never scored",
    }
    return [{**metadata, "arms": output} for output in outputs]


def run_scaling_forecasts(
    sequence: Tensor, baseline: RidgeRecentPredictor, config: ScalingMemoryConfig | None = None,
    *, arms: tuple[str, ...] = ARMS,
    initial_weights: dict[str, dict[str, Tensor]] | None = None,
    device: str | torch.device = "cpu", keep_trajectories: bool = False, return_states: bool = False,
) -> dict[str, Any]:
    """Single-episode wrapper around the independent batched executor."""
    if not isinstance(sequence, Tensor) or sequence.ndim != 3:
        raise ValueError("sequence must be [steps,tokens,dim]")
    return run_scaling_forecasts_batch(
        sequence.unsqueeze(0), baseline, config, arms=arms, initial_weights=initial_weights,
        device=device, keep_trajectories=keep_trajectories, return_states=return_states,
    )[0]


__all__ = ["ARMS", "FAST_ARMS", "ScalingMemoryConfig", "initial_fast_weights", "run_scaling_forecasts", "run_scaling_forecasts_batch"]
