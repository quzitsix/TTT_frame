"""Past-only latent forecasting preflight for Idea 01.

The frozen recent predictor is fitted on explicitly supplied training streams.
At evaluation, every prediction is stored before its target arrives. When the
target becomes an observed prefix feature, the old prediction is scored first,
then online memories mature, then new predictions are made. No benchmark
future, answer text, or task label is accepted by this module.

This CPU reference is deliberately smaller than the complete proposal: the key
is the standardized current feature, the initialization is random/zero, and the
learning rate is fixed. There is no outer training or language readout. A bounded
nearest-cue bank is an explicit visual-memory control, not VLM ICL.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from ttt_frame.predictive_memory import DelayedResidualConfig, DelayedResidualMemory


ARMS = (
    "persistence", "recent_only", "plain_delayed", "residual", "reset_residual",
    "explicit_bank",
)
FAST_ARMS = frozenset(("plain_delayed", "residual", "reset_residual"))


def _positive_int(value: int, name: str, *, zero_allowed: bool = False) -> None:
    if type(value) is not int or value < (0 if zero_allowed else 1):
        qualifier = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _horizons(values: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(values)
    if not values or any(type(h) is not int or h < 1 for h in values):
        raise ValueError("horizons must be positive integer stream steps")
    if len(set(values)) != len(values):
        raise ValueError("horizons must be unique")
    return values


def _sequence(value: Tensor, shape: tuple[int, int] | None = None) -> Tensor:
    if (
        not isinstance(value, Tensor) or value.ndim != 3
        or not value.is_floating_point() or min(value.shape) < 1
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("sequence must be finite floating point [steps, tokens, dim]")
    if shape is not None and tuple(value.shape[1:]) != shape:
        raise ValueError(f"sequence tokens/dimension must match {shape}")
    return value.detach().to(device="cpu", dtype=torch.float32)


def _design(last: Tensor, recent_mean: Tensor) -> Tensor:
    return torch.cat((last, recent_mean, torch.ones_like(last[..., :1])), dim=-1)


@dataclass(frozen=True)
class RidgeRecentPredictor:
    """Frozen ridge predictor of future features from last + recent mean.

    Mean/std are per feature channel across all training tokens. One regression
    is shared across spatial tokens for each horizon. ``training_rows`` counts
    the selected token rows per horizon; selection is deterministic and never
    pairs the end of one stream with the beginning of another.
    """

    horizons: tuple[int, ...]
    window_steps: int
    mean: Tensor
    std: Tensor
    coefficients: Tensor
    tokens: int
    training_rows: tuple[int, ...]
    standardize: bool
    ridge: float

    @property
    def dim(self) -> int:
        return self.mean.numel()

    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        train_sequences: list[Tensor],
        *,
        horizons: tuple[int, ...] = (1, 2, 4, 10),
        window_steps: int = 4,
        ridge: float = 1e-2,
        standardize: bool = True,
        max_fit_rows: int = 20000,
    ) -> RidgeRecentPredictor:
        horizons = _horizons(horizons)
        _positive_int(window_steps, "window_steps")
        _positive_int(max_fit_rows, "max_fit_rows")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if not train_sequences:
            raise ValueError("at least one training sequence is required")
        streams = [_sequence(train_sequences[0])]
        shape = tuple(streams[0].shape[1:])
        streams.extend(_sequence(s, shape) for s in train_sequences[1:])
        tokens, dim = shape
        if max_fit_rows < tokens:
            raise ValueError("max_fit_rows must allow all tokens of at least one cue")
        total = sum(s.shape[0] * tokens for s in streams)
        mean = sum(s.double().sum(dim=(0, 1)) for s in streams) / total
        variance = sum((s.double() - mean).square().sum(dim=(0, 1)) for s in streams) / total
        if standardize:
            mean, std = mean.float(), variance.sqrt().clamp_min(1e-4).float()
        else:
            mean, std = torch.zeros(dim), torch.ones(dim)
        coefficients, training_rows = [], []
        width = 2 * dim + 1
        for horizon in horizons:
            counts = [max(0, s.shape[0] - horizon) for s in streams]
            possible = sum(counts)
            if possible < 1:
                raise ValueError(f"no within-stream training pair for horizon {horizon}")
            count = min(possible, max_fit_rows // tokens)
            # Pick origins globally, retaining complete spatial cue groups.
            selected = torch.linspace(0, possible - 1, count).round().long()
            gram, cross = torch.zeros(width, width, dtype=torch.float64), torch.zeros(
                width, dim, dtype=torch.float64
            )
            offset, rows = 0, 0
            for stream, available in zip(streams, counts, strict=True):
                local = selected[(selected >= offset) & (selected < offset + available)] - offset
                offset += available
                if local.numel() == 0:
                    continue
                z = (stream - mean) / std
                cumulative = torch.cat((torch.zeros_like(z[:1]), z.cumsum(dim=0)))
                starts = (local - window_steps + 1).clamp_min(0)
                recent = (cumulative[local + 1] - cumulative[starts]) / (
                    local + 1 - starts
                ).view(-1, 1, 1)
                x = _design(z[local], recent).reshape(-1, width).double()
                y = z[local + horizon].reshape(-1, dim).double()
                gram += x.T @ x
                cross += x.T @ y
                rows += x.shape[0]
            regularizer = torch.eye(width, dtype=torch.float64) * ridge
            regularizer[-1, -1] = 0.0  # The intercept is not penalized.
            coefficient = torch.linalg.solve(gram / rows + regularizer, cross / rows).float()
            if not bool(torch.isfinite(coefficient).all()):
                raise RuntimeError("recent predictor fit produced nonfinite coefficients")
            coefficients.append(coefficient)
            training_rows.append(rows)
        return cls(
            horizons, window_steps, mean, std, torch.stack(coefficients), tokens,
            tuple(training_rows), standardize, float(ridge),
        )

    def normalize(self, features: Tensor) -> Tensor:
        return (features - self.mean) / self.std

    def denormalize(self, features: Tensor) -> Tensor:
        return features * self.std + self.mean

    def predict_standardized(self, recent: Tensor) -> Tensor:
        """Predict all horizons from already standardized past/current tokens."""
        if recent.ndim != 3 or tuple(recent.shape[1:]) != (self.tokens, self.dim):
            raise ValueError("recent must have shape [steps, tokens, dim]")
        if not 1 <= recent.shape[0] <= self.window_steps:
            raise ValueError("recent window length exceeds the fitted window")
        x = _design(recent[-1], recent.mean(dim=0))
        return torch.einsum("pf,hfd->hpd", x, self.coefficients)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "horizons": list(self.horizons), "window_steps": self.window_steps,
            "mean": self.mean.detach().clone(), "std": self.std.detach().clone(),
            "coefficients": self.coefficients.detach().clone(), "tokens": self.tokens,
            "training_rows": list(self.training_rows), "standardize": self.standardize,
            "ridge": self.ridge,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RidgeRecentPredictor:
        if payload.get("format_version") != 1:
            raise ValueError("unsupported frozen recent predictor version")
        horizons = _horizons(tuple(payload["horizons"]))
        _positive_int(payload["window_steps"], "window_steps")
        _positive_int(payload["tokens"], "tokens")
        mean, std, coefficients = (
            payload[key].detach().to(device="cpu", dtype=torch.float32).clone()
            for key in ("mean", "std", "coefficients")
        )
        if (
            mean.ndim != 1 or mean.numel() < 1 or std.shape != mean.shape
            or coefficients.shape != (len(horizons), 2 * mean.numel() + 1, mean.numel())
            or any(not bool(torch.isfinite(x).all()) for x in (mean, std, coefficients))
            or bool((std <= 0).any())
        ):
            raise ValueError("invalid frozen recent predictor tensors")
        rows = tuple(payload["training_rows"])
        if len(rows) != len(horizons) or any(type(n) is not int or n < 1 for n in rows):
            raise ValueError("invalid training row counts")
        ridge = payload["ridge"]
        if not math.isfinite(ridge) or ridge <= 0 or type(payload["standardize"]) is not bool:
            raise ValueError("invalid frozen recent predictor configuration")
        return cls(horizons, payload["window_steps"], mean, std, coefficients,
                   payload["tokens"], rows, payload["standardize"], ridge)


@dataclass(frozen=True)
class PredictiveRunConfig:
    hidden_dim: int = 128
    learning_rate: float = 0.05
    bank_capacity: int = 64
    seed: int = 0
    score_start_step: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.hidden_dim, "hidden_dim")
        _positive_int(self.bank_capacity, "bank_capacity")
        _positive_int(self.score_start_step, "score_start_step", zero_allowed=True)
        _positive_int(self.seed, "seed", zero_allowed=True)
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")


@torch.no_grad()
def run_prequential(
    sequence: Tensor,
    baseline: RidgeRecentPredictor,
    config: PredictiveRunConfig | None = None,
    *,
    arms: tuple[str, ...] = ARMS,
    keep_predictions: bool = False,
) -> dict[str, Any]:
    """Score causal forecasts inside an observed prefix, then return final forecasts.

    Scores use the frozen train-standardized space. Each counted pair is one
    (origin, horizon) whose target is in the supplied prefix; its MSE averages
    tokens and channels. ``score_start_step`` filters prediction origins, never
    target arrival times. The final forecasts remain unscored: their outcomes
    have not arrived. Optional full trajectories and final forecasts are in the
    original feature space.

    ``reset_residual`` resets fast weights before each maturity call but keeps
    delayed cues/baselines; it can use updates maturing now but cannot retain
    accumulated fast weights from earlier arrivals. The bank stores one whole
    cue/residual matrix per slot and retrieves the nearest cue by mean squared
    key distance, independently for each horizon.
    """
    config = config or PredictiveRunConfig()
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError(f"arms must be distinct nonempty members of {ARMS}")
    z = baseline.normalize(_sequence(sequence, (baseline.tokens, baseline.dim)))
    length, tokens, dim = z.shape
    horizons = baseline.horizons
    memories, states = {}, {}
    for arm in arms:
        if arm in FAST_ARMS:
            memory = DelayedResidualMemory(DelayedResidualConfig(
                dim=dim, hidden_dim=config.hidden_dim, horizons=horizons,
                learning_rate=config.learning_rate, max_pending=sum(horizons), seed=config.seed,
            ))
            memory.requires_grad_(False)
            memories[arm], states[arm] = memory, memory.initial_state()
    bank = {h: deque(maxlen=config.bank_capacity) for h in horizons}
    bank_pending: dict[int, list[tuple[int, Tensor, Tensor]]] = defaultdict(list)
    bank_updates = 0
    # This queue exists only for evaluation. It never supplies a model read.
    score_pending: dict[int, list[tuple[int, int, dict[str, Tensor]]]] = defaultdict(list)
    sums = {arm: {h: 0.0 for h in horizons} for arm in arms}
    counts = {h: 0 for h in horizons}
    trajectories = {arm: [] for arm in arms} if keep_predictions else None
    last: dict[str, Tensor] = {}
    for step in range(length):
        target = z[step]
        # Score the *old* prediction before any update with today's feature.
        for origin, horizon, predictions in score_pending.pop(step, []):
            if origin >= config.score_start_step:
                counts[horizon] += 1
                for arm, prediction in predictions.items():
                    error = float((prediction - target).square().mean())
                    if not math.isfinite(error):
                        raise RuntimeError(f"nonfinite prequential error in {arm} at {step}")
                    sums[arm][horizon] += error
        for arm, memory in memories.items():
            if arm == "reset_residual":
                fresh = memory.initial_state()
                states[arm] = replace(states[arm], w_in=fresh.w_in, w_out=fresh.w_out)
            states[arm], _ = memory.mature(step=step, target=target, state=states[arm])
        if "explicit_bank" in arms:
            for horizon, key, previous_baseline in bank_pending.pop(step, []):
                bank[horizon].append((key, (target - previous_baseline).clone()))
                bank_updates += 1
        recent = z[max(0, step - baseline.window_steps + 1):step + 1]
        predicted_baseline = baseline.predict_standardized(recent)
        per_arm: dict[str, list[Tensor]] = {arm: [] for arm in arms}
        for index, horizon in enumerate(horizons):
            b = predicted_baseline[index]
            for arm in arms:
                if arm == "persistence":
                    prediction = target.clone()
                elif arm == "recent_only":
                    prediction = b.clone()
                elif arm in FAST_ARMS:
                    base = torch.zeros_like(b) if arm == "plain_delayed" else b
                    prediction, states[arm] = memories[arm].schedule(
                        step=step, horizon=horizon, key=target.clone(), baseline=base.clone(),
                        state=states[arm],
                    )
                else:
                    if bank[horizon]:
                        distances = torch.stack([
                            (key - target).square().mean() for key, _ in bank[horizon]
                        ])
                        residual = bank[horizon][int(distances.argmin())][1]
                        prediction = b + residual
                    else:
                        prediction = b.clone()
                    bank_pending[step + horizon].append((horizon, target.clone(), b.clone()))
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError(f"nonfinite {arm} forecast at step {step}")
                per_arm[arm].append(prediction.detach())
            # Keep only a bounded maturity ring, even when scoring a late tail.
            if step + horizon < length:
                score_pending[step + horizon].append((
                    step, horizon, {arm: per_arm[arm][-1].clone() for arm in arms}
                ))
        last = {arm: torch.stack(values) for arm, values in per_arm.items()}
        if trajectories is not None:
            for arm in arms:
                trajectories[arm].append(baseline.denormalize(last[arm]))
    outputs = {}
    for arm in arms:
        state_bytes = capacity_bytes = pending_bytes = updates = 0
        if arm in FAST_ARMS:
            state_bytes = capacity_bytes = memories[arm].memory_bytes(states[arm])
            pending_bytes = memories[arm].pending_bytes(states[arm])
            updates = states[arm].updates
        elif arm == "explicit_bank":
            cue_bytes = 2 * tokens * dim * 4
            state_bytes = cue_bytes * sum(len(entries) for entries in bank.values())
            capacity_bytes = cue_bytes * config.bank_capacity * len(horizons)
            pending_bytes = cue_bytes * sum(len(entries) for entries in bank_pending.values())
            updates = bank_updates
        count = sum(counts.values())
        outputs[arm] = {
            "mse": sum(sums[arm].values()) / count if count else None,
            "count": count,
            "by_horizon": {str(h): {"mse": sums[arm][h] / counts[h] if counts[h] else None,
                                   "count": counts[h]} for h in horizons},
            "last_forecasts": baseline.denormalize(last[arm]),
            "state_bytes": state_bytes, "state_capacity_bytes": capacity_bytes,
            "pending_bytes": pending_bytes, "updates": updates,
        }
        if trajectories is not None:
            outputs[arm]["predictions"] = torch.stack(trajectories[arm])
    return {
        "schema": "idea01_prequential_latent_v1", "sequence_steps": length,
        "feature_shape": [tokens, dim], "score_start_step": config.score_start_step,
        "horizons": list(horizons), "score_space": "frozen_train_standardized_features",
        "count_unit": "matured_origin_horizon_pair; each MSE averages tokens and channels",
        "shared_recent_buffer_bytes": min(length, baseline.window_steps) * tokens * dim * 4,
        "resident_input_feature_cache_bytes": z.numel() * z.element_size(),
        "frozen_recent_predictor_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in (baseline.mean, baseline.std, baseline.coefficients)
        ),
        "state_byte_scope": "episode model state only; input cache and scoring queue excluded",
        "reset_policy": "reset weights before each maturity; retain delayed cue FIFO",
        "bank_capacity_unit": "whole cue/residual pairs per horizon",
        "outer_training_performed": False, "explicit_bank_is_vlm_icl": False,
        "arms": outputs,
    }


__all__ = ["ARMS", "PredictiveRunConfig", "RidgeRecentPredictor", "run_prequential"]
