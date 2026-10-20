"""Causal and mathematical checks for the delayed-residual MVP."""

from __future__ import annotations

import pytest
import torch

from ttt_frame.predictive_memory import DelayedResidualConfig, DelayedResidualMemory


def make_memory(**kwargs) -> DelayedResidualMemory:
    options = dict(dim=8, hidden_dim=6, horizons=(1, 2), learning_rate=0.5, seed=7)
    options.update(kwargs)
    return DelayedResidualMemory(DelayedResidualConfig(**options))


def features(seed: int, tokens: int = 3, dim: int = 8) -> torch.Tensor:
    return torch.randn(tokens, dim, generator=torch.Generator().manual_seed(seed))


def test_cold_start_is_exact_recent_baseline_and_future_does_not_leak():
    memory = make_memory()
    key, baseline = features(1), features(2)
    prediction, state = memory.schedule(
        step=0, horizon=2, key=key, baseline=baseline
    )
    torch.testing.assert_close(prediction, baseline)
    pristine = state.w_out.clone()
    state, metrics = memory.mature(step=1, target=features(3), state=state)
    assert metrics["matured"] == 0 and len(state.pending) == 1
    assert torch.equal(state.w_out, pristine)


def test_matured_residual_update_reduces_error_at_written_key():
    memory = make_memory()
    key, baseline = features(4), features(5)
    target = baseline + 0.2 * features(6)
    _, state = memory.schedule(step=0, horizon=1, key=key, baseline=baseline)
    state, metrics = memory.mature(step=1, target=target, state=state)
    assert metrics["matured"] == 1 and metrics["pending"] == 0
    assert metrics["mse_after"] < metrics["mse_before"]
    assert state.updates == 1
    assert memory.read(key, 1, state).abs().sum() > 0


def test_each_horizon_has_independent_weights_and_due_step():
    memory = make_memory()
    key, baseline, target = features(7), features(8), features(9)
    _, state = memory.schedule(step=0, horizon=1, key=key, baseline=baseline)
    _, state = memory.schedule(step=0, horizon=2, key=key, baseline=baseline, state=state)
    before_h2 = state.w_out[1].clone()
    state, _ = memory.mature(step=1, target=target, state=state)
    assert not torch.equal(state.w_out[0], torch.zeros_like(state.w_out[0]))
    assert torch.equal(state.w_out[1], before_h2)
    assert len(state.pending) == 1 and state.pending[0].due_step == 2


def test_skipping_a_due_step_is_rejected_instead_of_changing_the_horizon():
    memory = make_memory()
    _, state = memory.schedule(
        step=0, horizon=1, key=features(10), baseline=features(11)
    )
    with pytest.raises(RuntimeError, match="missed maturity"):
        memory.mature(step=2, target=features(12), state=state)


def test_due_updates_must_mature_before_scheduling_at_the_same_step():
    memory = make_memory()
    _, state = memory.schedule(
        step=0, horizon=1, key=features(50), baseline=features(51)
    )
    with pytest.raises(RuntimeError, match="mature pending cue"):
        memory.schedule(
            step=1,
            horizon=2,
            key=features(52),
            baseline=features(53),
            state=state,
        )
    state, metrics = memory.mature(step=1, target=features(54), state=state)
    assert metrics["matured"] == 1
    _, state = memory.schedule(
        step=1,
        horizon=2,
        key=features(52),
        baseline=features(53),
        state=state,
    )
    assert state.pending[0].due_step == 3


def test_outer_gradient_flows_through_manual_update_to_initialization_and_key():
    memory = make_memory()
    key = features(13).requires_grad_()
    baseline, target = features(14), features(15)
    _, state = memory.schedule(step=0, horizon=1, key=key, baseline=baseline)
    state, _ = memory.mature(step=1, target=target, state=state)
    loss = memory.read(features(16), 1, state).square().sum()
    loss.backward()
    assert memory.w_in.grad is not None and torch.isfinite(memory.w_in.grad).all()
    assert memory.w_out.grad is not None and torch.isfinite(memory.w_out.grad).all()
    assert key.grad is not None and key.grad.abs().sum() > 0


def test_memory_size_is_fixed_and_fifo_is_bounded():
    memory = make_memory(max_pending=2)
    state = memory.initial_state()
    fixed = memory.memory_bytes(state)
    for horizon in (1, 2):
        _, state = memory.schedule(
            step=0,
            horizon=horizon,
            key=features(17 + horizon),
            baseline=features(20 + horizon),
            state=state,
        )
    assert memory.memory_bytes(state) == fixed
    assert memory.pending_bytes(state) > 0
    with pytest.raises(RuntimeError, match="FIFO is full"):
        memory.schedule(
            step=0,
            horizon=1,
            key=features(30),
            baseline=features(31),
            state=state,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"dim": 0},
        {"hidden_dim": 0},
        {"horizons": ()},
        {"horizons": (1, 1)},
        {"learning_rate": -1},
        {"max_pending": 0},
    ],
)
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError):
        DelayedResidualConfig(**options)


def test_wrong_feature_shape_and_nonmonotonic_steps_are_rejected():
    memory = make_memory()
    with pytest.raises(ValueError, match="shape"):
        memory.schedule(
            step=0,
            horizon=1,
            key=torch.zeros(2, 7),
            baseline=torch.zeros(2, 8),
        )
    _, state = memory.schedule(
        step=2, horizon=1, key=features(40), baseline=features(41)
    )
    with pytest.raises(ValueError, match="nondecreasing"):
        memory.schedule(
            step=1,
            horizon=1,
            key=features(42),
            baseline=features(43),
            state=state,
        )
