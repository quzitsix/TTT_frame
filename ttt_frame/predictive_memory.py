"""Small delayed-residual fast memory for predictive-TTT preflight experiments.

This module implements the minimal mechanism proposed by Idea 01 in
``ideax_output``.  It deliberately lives outside the 21-layer Spatial-TTT
wrapper: the first experiment should decide whether the write target is useful
on frozen visual features before changing the full VLM.

At step ``t`` a caller predicts a future representation using a frozen recent
predictor (``baseline``) plus a residual read from fast weights, then schedules
the cue.  The fast weights are updated only when the representation at
``t + horizon`` has actually arrived.  Future targets are therefore never
available to an online read before their due step.

The implementation is functional.  Runtime weights are stored in an explicit
state object, while their trainable initialization is kept in the module.  The
manual MSE update remains differentiable, so an outer loss can train the slow
initialization and an upstream key projector without retaining video frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DelayedResidualConfig:
    """Configuration for :class:`DelayedResidualMemory`.

    Horizons are integer stream steps.  A video experiment decides what one
    step means (for example 0.5 seconds); keeping that conversion outside the
    memory prevents timestamp rounding from silently changing causal order.
    """

    dim: int = 256
    hidden_dim: int = 128
    horizons: tuple[int, ...] = (1, 2, 4, 10)
    learning_rate: float = 1e-2
    # With the default four horizons, scheduling one cue per horizon and
    # stream step needs at most 17 live entries.  Forty leaves room for minor
    # batching jitter while retaining the small-state budget in the proposal.
    max_pending: int = 40
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 1:
            raise ValueError("dim must be a positive integer")
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or self.hidden_dim < 1
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if not self.horizons or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.horizons
        ):
            raise ValueError("horizons must contain positive integer stream steps")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if (
            isinstance(self.max_pending, bool)
            or not isinstance(self.max_pending, int)
            or self.max_pending < 1
        ):
            raise ValueError("max_pending must be a positive integer")


@dataclass(frozen=True)
class PendingResidual:
    """One cue waiting for its causally observed outcome."""

    due_step: int
    horizon_index: int
    key: Tensor
    baseline: Tensor


@dataclass(frozen=True)
class DelayedResidualState:
    """Episode state: fixed fast weights plus a bounded maturity FIFO."""

    w_in: Tensor
    w_out: Tensor
    pending: tuple[PendingResidual, ...] = ()
    updates: int = 0
    last_step: int = -1

    def detached(self) -> DelayedResidualState:
        return replace(
            self,
            w_in=self.w_in.detach(),
            w_out=self.w_out.detach(),
            pending=tuple(
                replace(item, key=item.key.detach(), baseline=item.baseline.detach())
                for item in self.pending
            ),
        )


def _check_step(step: int) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")


def _check_feature(name: str, value: Tensor, dim: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != dim:
        raise ValueError(f"{name} must have shape [tokens, {dim}]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and floating point")


def residual_swiglu_read(w_in: Tensor, w_out: Tensor, key: Tensor) -> Tensor:
    """Evaluate ``B silu(A k)`` using row-major token inputs."""

    return F.silu(key @ w_in.transpose(-1, -2)) @ w_out.transpose(-1, -2)


def residual_mse_update(
    w_in: Tensor,
    w_out: Tensor,
    key: Tensor,
    target: Tensor,
    learning_rate: float,
) -> tuple[Tensor, Tensor]:
    """One explicit gradient step on mean-squared residual prediction error.

    The derivatives are written with ordinary tensor operations rather than
    ``autograd.grad``.  This makes the online update usable inside
    ``torch.no_grad`` while still allowing an outer graph when gradients are
    enabled by the caller.
    """

    preactivation = key @ w_in.transpose(-1, -2)
    hidden = F.silu(preactivation)
    prediction = hidden @ w_out.transpose(-1, -2)
    derivative = 2.0 * (prediction - target) / target.numel()
    grad_out = derivative.transpose(-1, -2) @ hidden
    sigma = torch.sigmoid(preactivation)
    silu_derivative = sigma * (1 + preactivation * (1 - sigma))
    grad_in = ((derivative @ w_out) * silu_derivative).transpose(-1, -2) @ key
    return w_in - learning_rate * grad_in, w_out - learning_rate * grad_out


class DelayedResidualMemory(nn.Module):
    """A horizon-indexed fast MLP updated when scheduled outcomes mature."""

    def __init__(self, config: DelayedResidualConfig | None = None) -> None:
        super().__init__()
        self.config = config or DelayedResidualConfig()
        count = len(self.config.horizons)
        generator = torch.Generator().manual_seed(self.config.seed)
        self.w_in = nn.Parameter(
            torch.randn(
                count,
                self.config.hidden_dim,
                self.config.dim,
                generator=generator,
            )
            / math.sqrt(self.config.dim)
        )
        # A zero residual at initialization makes the recent predictor the
        # exact cold-start baseline.  Outer training may learn a nonzero prior.
        self.w_out = nn.Parameter(
            torch.zeros(count, self.config.dim, self.config.hidden_dim)
        )
        self._horizon_to_index = {
            horizon: index for index, horizon in enumerate(self.config.horizons)
        }

    def initial_state(self) -> DelayedResidualState:
        return DelayedResidualState(self.w_in.float().clone(), self.w_out.float().clone())

    def _state(self, state: DelayedResidualState | None) -> DelayedResidualState:
        current = self.initial_state() if state is None else state
        expected_in = (
            len(self.config.horizons),
            self.config.hidden_dim,
            self.config.dim,
        )
        expected_out = (
            len(self.config.horizons),
            self.config.dim,
            self.config.hidden_dim,
        )
        if current.w_in.shape != expected_in or current.w_out.shape != expected_out:
            raise ValueError("state weight shapes do not match memory configuration")
        if current.w_in.device != self.w_in.device or current.w_out.device != self.w_out.device:
            raise ValueError("state and memory module must be on the same device")
        if not bool(torch.isfinite(current.w_in).all()) or not bool(
            torch.isfinite(current.w_out).all()
        ):
            raise ValueError("state contains nonfinite fast weights")
        return current

    def read(
        self,
        key: Tensor,
        horizon: int,
        state: DelayedResidualState | None = None,
    ) -> Tensor:
        """Read a residual without scheduling or changing episode state."""

        _check_feature("key", key, self.config.dim)
        if horizon not in self._horizon_to_index:
            raise ValueError(f"unsupported horizon {horizon}; expected {self.config.horizons}")
        current = self._state(state)
        index = self._horizon_to_index[horizon]
        key = key.to(device=current.w_in.device, dtype=torch.float32)
        return residual_swiglu_read(current.w_in[index], current.w_out[index], key)

    def schedule(
        self,
        *,
        step: int,
        horizon: int,
        key: Tensor,
        baseline: Tensor,
        state: DelayedResidualState | None = None,
    ) -> tuple[Tensor, DelayedResidualState]:
        """Predict now and enqueue the cue for a later, causally valid update."""

        _check_step(step)
        _check_feature("key", key, self.config.dim)
        _check_feature("baseline", baseline, self.config.dim)
        if key.shape != baseline.shape:
            raise ValueError("key and baseline must have identical token shapes")
        if horizon not in self._horizon_to_index:
            raise ValueError(f"unsupported horizon {horizon}; expected {self.config.horizons}")
        current = self._state(state)
        if step < current.last_step:
            raise ValueError("stream steps must be nondecreasing")
        due = [item.due_step for item in current.pending if item.due_step <= step]
        if due:
            raise RuntimeError(
                f"mature pending cue due at step {min(due)} before scheduling step {step}"
            )
        if len(current.pending) >= self.config.max_pending:
            raise RuntimeError("delayed-residual FIFO is full")
        key = key.to(device=current.w_in.device, dtype=torch.float32)
        baseline = baseline.to(device=current.w_in.device, dtype=torch.float32)
        prediction = baseline + self.read(key, horizon, current)
        item = PendingResidual(
            due_step=step + horizon,
            horizon_index=self._horizon_to_index[horizon],
            key=key,
            # The recent predictor is trained separately and frozen for the
            # memory experiment.  Do not backpropagate an inner update into it.
            baseline=baseline.detach(),
        )
        return prediction, replace(
            current,
            pending=(*current.pending, item),
            last_step=step,
        )

    def mature(
        self,
        *,
        step: int,
        target: Tensor,
        state: DelayedResidualState,
    ) -> tuple[DelayedResidualState, dict[str, float | int]]:
        """Apply every update due at ``step`` using the newly observed target.

        A skipped due step is rejected.  Updating an overdue cue with a later
        frame would silently change its forecasting horizon and violate the
        causal protocol.
        """

        _check_step(step)
        _check_feature("target", target, self.config.dim)
        current = self._state(state)
        if step < current.last_step:
            raise ValueError("stream steps must be nondecreasing")
        overdue = [item.due_step for item in current.pending if item.due_step < step]
        if overdue:
            raise RuntimeError(f"missed maturity step {min(overdue)} before step {step}")
        due = [item for item in current.pending if item.due_step == step]
        if any(item.key.shape != target.shape for item in due):
            raise ValueError("target token shape does not match a due prediction")
        target = target.to(device=current.w_in.device, dtype=torch.float32).detach()
        w_in, w_out = current.w_in, current.w_out
        before_error = 0.0
        after_error = 0.0
        for item in due:
            index = item.horizon_index
            residual_target = target - item.baseline
            before = residual_swiglu_read(w_in[index], w_out[index], item.key)
            before_error += float(F.mse_loss(before.detach(), residual_target).cpu())
            next_in, next_out = residual_mse_update(
                w_in[index],
                w_out[index],
                item.key,
                residual_target,
                self.config.learning_rate,
            )
            w_in = torch.cat((w_in[:index], next_in.unsqueeze(0), w_in[index + 1 :]))
            w_out = torch.cat((w_out[:index], next_out.unsqueeze(0), w_out[index + 1 :]))
            after = residual_swiglu_read(w_in[index], w_out[index], item.key)
            after_error += float(F.mse_loss(after.detach(), residual_target).cpu())
        remaining = tuple(item for item in current.pending if item.due_step != step)
        count = len(due)
        metrics: dict[str, float | int] = {
            "matured": count,
            "pending": len(remaining),
            "updates": current.updates + count,
            "mse_before": before_error / count if count else 0.0,
            "mse_after": after_error / count if count else 0.0,
        }
        return replace(
            current,
            w_in=w_in,
            w_out=w_out,
            pending=remaining,
            updates=current.updates + count,
            last_step=step,
        ), metrics

    def memory_bytes(self, state: DelayedResidualState | None = None) -> int:
        """Bytes in fixed fast weights, excluding the explicitly reported FIFO."""

        current = self._state(state)
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (current.w_in, current.w_out)
        )

    def pending_bytes(self, state: DelayedResidualState) -> int:
        current = self._state(state)
        return sum(
            item.key.numel() * item.key.element_size()
            + item.baseline.numel() * item.baseline.element_size()
            for item in current.pending
        )


__all__ = [
    "DelayedResidualConfig",
    "DelayedResidualMemory",
    "DelayedResidualState",
    "PendingResidual",
    "residual_mse_update",
    "residual_swiglu_read",
]
