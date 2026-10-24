"""Causal, independent and width-controlled scaled fast memories."""
from dataclasses import replace

import pytest
import torch

from ttt_frame.forecast_residual import RidgeRecentPredictor
from ttt_frame.idea1_mcq_memory import FinalForecastConfig, run_final_forecasts
from ttt_frame.idea1_scaling_memory import (
    ARMS, ScalingMemoryConfig, _batched_update, initial_fast_weights,
    run_scaling_forecasts, run_scaling_forecasts_batch,
)


def stream(seed=1, steps=24):
    return torch.randn(steps, 2, 4, generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def baseline():
    return RidgeRecentPredictor.fit([stream(1), stream(2)], horizons=(1, 3), window_steps=3)


@pytest.mark.parametrize("width,gain_mode", [(128, "normalized"), (5, "reference")])
def test_reference_single_step_predictions_and_statistics(baseline, width, gain_mode):
    cfg = ScalingMemoryConfig(hidden_dim=width, gain_mode=gain_mode, tail_start_step=8)
    actual = run_scaling_forecasts(stream(), baseline, cfg, keep_trajectories=True)
    expected = run_final_forecasts(stream(), baseline,
                                  FinalForecastConfig(hidden_dim=width, tail_start_step=8),
                                  keep_trajectories=True)
    for arm in ("persistence", "recent_only", "plain_delayed", "residual", "reset_residual", "recent_tail_residual"):
        torch.testing.assert_close(actual["arms"][arm]["predictions"],
                                   expected["arms"][arm]["predictions"], rtol=1e-5, atol=1e-6)
        assert actual["arms"][arm]["updates"] == expected["arms"][arm]["updates"]


def test_manual_normalized_gradient_matches_autograd():
    generator = torch.Generator().manual_seed(8)
    a = torch.randn(2, 3, 2, 5, 4, generator=generator).requires_grad_()
    b = torch.randn(2, 3, 2, 4, 5, generator=generator).requires_grad_()
    keys = torch.randn(2, 1, 2, 2, 4, generator=generator)
    targets = torch.randn(2, 3, 2, 2, 4, generator=generator)
    active = torch.ones(2, 3, 2, 1, 1)
    active[1, 1, 0] = 0
    gain, lr = 0.25, 0.03
    prediction = (torch.nn.functional.silu(keys @ a.transpose(-1, -2)) * gain) @ b.transpose(-1, -2)
    loss = ((prediction - targets).square().mean(dim=(-2, -1)) * active.squeeze(-1).squeeze(-1)).sum()
    grad_a, grad_b = torch.autograd.grad(loss, (a, b))
    manual_a, manual_b = a.detach().clone(), b.detach().clone()
    _batched_update(manual_a, manual_b, keys, targets, active, lr, gain)
    torch.testing.assert_close(manual_a, a.detach() - lr * grad_a, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(manual_b, b.detach() - lr * grad_b, atol=1e-6, rtol=1e-5)


def test_width_gain_removes_first_update_response_from_duplicate_neurons():
    """Repeating the exact same random features must not quadruple the step."""
    generator = torch.Generator().manual_seed(4)
    narrow_a = torch.randn(1, 1, 128, 4, generator=generator)
    narrow_b = torch.zeros(1, 1, 4, 128)
    wide_a, wide_b = narrow_a.repeat_interleave(4, dim=-2), narrow_b.repeat_interleave(4, dim=-1)
    keys = torch.randn(1, 1, 2, 4, generator=generator)
    targets = torch.randn(1, 1, 2, 4, generator=generator)
    active = torch.ones(1, 1, 1, 1)
    _batched_update(narrow_a, narrow_b, keys, targets, active, 0.03, 1)
    _batched_update(wide_a, wide_b, keys, targets, active, 0.03, 0.5)
    narrow_prediction = torch.nn.functional.silu(keys @ narrow_a.transpose(-1, -2)) @ narrow_b.transpose(-1, -2)
    wide_prediction = (torch.nn.functional.silu(keys @ wide_a.transpose(-1, -2)) * 0.5) @ wide_b.transpose(-1, -2)
    torch.testing.assert_close(narrow_prediction, wide_prediction, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("steps", [1, 4])
def test_batch_matches_independent_serial_episodes(baseline, steps):
    cfg = ScalingMemoryConfig(hidden_dim=7, inner_steps=steps, tail_start_step=8, gain_mode="reference")
    sequences = torch.stack([stream(3), stream(4)])
    batched = run_scaling_forecasts_batch(sequences, baseline, cfg, keep_trajectories=True, return_states=True)
    for index, sequence in enumerate(sequences):
        serial = run_scaling_forecasts(sequence, baseline, cfg, keep_trajectories=True, return_states=True)
        for arm in ARMS:
            for name in ("last_forecasts", "predictions"):
                torch.testing.assert_close(batched[index]["arms"][arm][name], serial["arms"][arm][name],
                                           rtol=1e-5, atol=1e-6)
            assert batched[index]["arms"][arm]["mature_count"] == serial["arms"][arm]["mature_count"]
            if "state" in serial["arms"][arm]:
                torch.testing.assert_close(batched[index]["arms"][arm]["state"]["w_out"],
                                           serial["arms"][arm]["state"]["w_out"], rtol=1e-5, atol=1e-6)


def test_future_suffix_and_other_episode_do_not_change_earlier_predictions(baseline):
    sequences = torch.stack([stream(3), stream(4)])
    changed = sequences.clone()
    changed[0, 12:] += 2
    changed[1] -= 5
    cfg = ScalingMemoryConfig(hidden_dim=7, inner_steps=4, tail_start_step=8, gain_mode="reference")
    left = run_scaling_forecasts_batch(sequences, baseline, cfg, keep_trajectories=True)
    right = run_scaling_forecasts_batch(changed, baseline, cfg, keep_trajectories=True)
    for arm in ARMS:
        assert torch.equal(left[0]["arms"][arm]["predictions"][:12], right[0]["arms"][arm]["predictions"][:12])


def test_learned_initialization_copied_and_controls_are_frozen(baseline):
    init = initial_fast_weights(2, 4, 7, 5)
    init["w_out"].normal_(0, 0.03)
    before = {name: value.clone() for name, value in init.items()}
    cfg = ScalingMemoryConfig(hidden_dim=7, tail_start_step=8, gain_mode="reference")
    weights = {"plain_delayed": init, "residual": init}
    result = run_scaling_forecasts(stream(), baseline, cfg, initial_weights=weights, return_states=True)
    for arm in ("plain_no_update", "residual_no_update"):
        assert result["arms"][arm]["updates"] == 0
        assert result["arms"][arm]["gradient_steps"] == 0
        assert result["arms"][arm]["fast_weight_delta_l2"] == 0
        for name, value in init.items():
            assert torch.equal(result["arms"][arm]["state"][name], value)
    assert not torch.equal(result["arms"]["residual_no_update"]["last_forecasts"],
                           result["arms"]["recent_only"]["last_forecasts"])
    for name, value in init.items():
        assert torch.equal(value, before[name])
    assert result["offline_initialization_supplied"]


def test_tail_resets_all_earlier_cues_and_learned_state(baseline):
    init = initial_fast_weights(2, 4, 7)
    init["w_out"].normal_(0, 0.03)
    cfg = ScalingMemoryConfig(hidden_dim=7, inner_steps=4, tail_start_step=8, gain_mode="reference")
    full = run_scaling_forecasts(stream(), baseline, cfg, initial_weights={"residual": init}, keep_trajectories=True)
    tail = run_scaling_forecasts(stream()[8:], baseline, cfg, arms=("residual",),
                                initial_weights={"residual": init}, keep_trajectories=True)
    torch.testing.assert_close(full["arms"]["recent_tail_residual"]["predictions"][8:],
                               tail["arms"]["residual"]["predictions"], rtol=1e-5, atol=1e-6)


def test_multiple_steps_counted_without_adding_new_observations(baseline):
    cfg = ScalingMemoryConfig(hidden_dim=7, inner_steps=4, tail_start_step=8, gain_mode="reference")
    result = run_scaling_forecasts(stream(), baseline, cfg)
    assert result["arms"]["residual"]["updates"] == 23 + 21
    assert result["arms"]["residual"]["gradient_steps"] == 4 * (23 + 21)
    assert result["arms"]["recent_tail_residual"]["gradient_steps"] == 4 * (15 + 13)


def test_mature_mse_scores_past_predictions_before_update(baseline):
    cfg = ScalingMemoryConfig(hidden_dim=7, score_last_steps=12)
    sequence = stream()
    result = run_scaling_forecasts(sequence, baseline, cfg, keep_trajectories=True)
    for arm in ARMS:
        saved = (result["arms"][arm]["predictions"] - baseline.mean) / baseline.std
        z = baseline.normalize(sequence)
        losses = []
        for origin in range(12, 24):
            for index, horizon in enumerate(baseline.horizons):
                if origin + horizon < 24:
                    losses.append((saved[origin, index] - z[origin + horizon]).double().square().mean())
        assert result["arms"][arm]["mature_count"] == 11 + 9
        assert result["arms"][arm]["mature_mse"] == pytest.approx(float(torch.stack(losses).mean()), rel=1e-6)
    assert not result["final_forecasts_scored"]


def test_zero_lr_preserves_init_and_scoring_does_not_change_state(baseline):
    cfg = ScalingMemoryConfig(hidden_dim=7, learning_rate=0)
    result = run_scaling_forecasts(stream(), baseline, cfg, return_states=True)
    assert torch.equal(result["arms"]["residual"]["last_forecasts"], result["arms"]["recent_only"]["last_forecasts"])
    cfg = replace(cfg, learning_rate=0.05)
    scored = run_scaling_forecasts(stream(), baseline, cfg)
    lean = run_scaling_forecasts(stream(), baseline, replace(cfg, score_last_steps=0))
    for arm in ARMS:
        assert torch.equal(scored["arms"][arm]["last_forecasts"], lean["arms"][arm]["last_forecasts"])


@pytest.mark.parametrize("kwargs", [{"hidden_dim": 0}, {"inner_steps": 0}, {"gain_mode": "bad"},
                                    {"learning_rate": float("nan")}, {"score_last_steps": -1}])
def test_bad_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ScalingMemoryConfig(**kwargs)


def test_invalid_initialization_and_duplicate_arms_rejected(baseline):
    cfg = ScalingMemoryConfig(hidden_dim=7)
    with pytest.raises(ValueError):
        run_scaling_forecasts(stream(), baseline, cfg, initial_weights={"residual": {"w_in": torch.zeros(1)}})
    with pytest.raises(ValueError):
        run_scaling_forecasts(stream(), baseline, cfg, arms=("residual", "residual"))
    with pytest.raises(ValueError):
        run_scaling_forecasts(stream(), baseline, cfg, initial_weights={"residual": {}})


@pytest.mark.parametrize("arm", ["persistence", "recent_only", "residual_no_update", "residual"])
def test_arm_subsets_match_joint_execution(baseline, arm):
    cfg = ScalingMemoryConfig(hidden_dim=7, gain_mode="reference", tail_start_step=8)
    joint = run_scaling_forecasts(stream(), baseline, cfg)
    single = run_scaling_forecasts(stream(), baseline, cfg, arms=(arm,))
    torch.testing.assert_close(single["arms"][arm]["last_forecasts"], joint["arms"][arm]["last_forecasts"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_batch_agree(baseline):
    cfg = ScalingMemoryConfig(hidden_dim=7, inner_steps=4, tail_start_step=8, gain_mode="reference")
    sequences = torch.stack([stream(3), stream(4)])
    cpu = run_scaling_forecasts_batch(sequences, baseline, cfg)
    gpu = run_scaling_forecasts_batch(sequences, baseline, cfg, device="cuda")
    for index in range(2):
        for arm in ARMS:
            torch.testing.assert_close(cpu[index]["arms"][arm]["last_forecasts"],
                                       gpu[index]["arms"][arm]["last_forecasts"], rtol=1e-4, atol=1e-5)
