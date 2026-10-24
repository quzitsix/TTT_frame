"""Checks against the frozen reference and independent causal controls."""

from dataclasses import replace

import pytest
import torch

from ttt_frame.forecast_residual import PredictiveRunConfig, RidgeRecentPredictor, run_prequential
from ttt_frame.idea1_mcq_memory import ARMS, FinalForecastConfig, run_final_forecasts


def stream(seed=3, steps=24):
    return torch.randn(steps, 2, 4, generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def baseline():
    return RidgeRecentPredictor.fit([stream(1), stream(2)], horizons=(1, 3), window_steps=3)


@pytest.mark.parametrize("steps", [1, 2, 24])
def test_all_original_arms_match_reference_predictions_and_accounting(baseline, steps):
    config = FinalForecastConfig(hidden_dim=5, learning_rate=0.07, bank_capacity=3, seed=9)
    actual = run_final_forecasts(stream(steps=steps), baseline, config, keep_trajectories=True)
    expected = run_prequential(
        stream(steps=steps), baseline,
        PredictiveRunConfig(hidden_dim=5, learning_rate=0.07, bank_capacity=3, seed=9),
        keep_predictions=True,
    )
    for arm in ARMS[:-1]:
        old_arm = "explicit_bank" if arm == "nearest_bank" else arm
        left, right = actual["arms"][arm], expected["arms"][old_arm]
        torch.testing.assert_close(left["last_forecasts"], right["last_forecasts"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(left["predictions"], right["predictions"], rtol=1e-5, atol=1e-6)
        for key in ("updates", "state_bytes", "state_capacity_bytes", "pending_bytes"):
            assert left[key] == right[key]


def test_forecast_only_and_trajectory_paths_agree(baseline):
    config = FinalForecastConfig(hidden_dim=5, tail_start_step=8)
    lean = run_final_forecasts(stream(), baseline, config)
    complete = run_final_forecasts(stream(), baseline, config, keep_trajectories=True)
    for arm in ARMS:
        assert "predictions" not in lean["arms"][arm]
        assert torch.equal(lean["arms"][arm]["last_forecasts"], complete["arms"][arm]["predictions"][-1])


def test_future_suffix_cannot_change_earlier_reads(baseline):
    original, changed = stream(), stream()
    changed[12:] = -2 * changed[12:] + 1
    config = FinalForecastConfig(hidden_dim=6, tail_start_step=8)
    left = run_final_forecasts(original, baseline, config, keep_trajectories=True)
    right = run_final_forecasts(changed, baseline, config, keep_trajectories=True)
    for arm in ARMS:
        assert torch.equal(left["arms"][arm]["predictions"][:12], right["arms"][arm]["predictions"][:12])


def test_tail_is_exactly_a_new_episode_on_the_suffix(baseline):
    sequence = stream()
    config = FinalForecastConfig(hidden_dim=6, bank_capacity=3, tail_start_step=8)
    full = run_final_forecasts(sequence, baseline, config, keep_trajectories=True)
    suffix = run_final_forecasts(sequence[8:], baseline, config, arms=("residual",), keep_trajectories=True)
    tail = full["arms"]["recent_tail_residual"]
    torch.testing.assert_close(tail["predictions"][8:], suffix["arms"]["residual"]["predictions"], rtol=1e-5, atol=1e-6)
    for key in ("updates", "pending_bytes"):
        assert tail[key] == suffix["arms"]["residual"][key]
    changed = sequence.clone()
    changed[:8] += 5
    other = run_final_forecasts(changed, baseline, config, arms=("recent_tail_residual",))
    assert torch.equal(tail["last_forecasts"], other["arms"]["recent_tail_residual"]["last_forecasts"])


def test_reset_forgets_old_weights_and_zero_lr_removes_residual(baseline):
    config = FinalForecastConfig(hidden_dim=5, learning_rate=0, tail_start_step=0)
    result = run_final_forecasts(stream(), baseline, config, keep_trajectories=True)["arms"]
    for arm in ("residual", "reset_residual", "recent_tail_residual"):
        assert torch.equal(result[arm]["predictions"], result["recent_only"]["predictions"])
    config = replace(config, learning_rate=0.05)
    altered = stream()
    altered[:8] += 2
    left = run_final_forecasts(stream(), baseline, config)["arms"]
    right = run_final_forecasts(altered, baseline, config)["arms"]
    assert torch.equal(left["reset_residual"]["last_forecasts"], right["reset_residual"]["last_forecasts"])
    assert not torch.equal(left["residual"]["last_forecasts"], right["residual"]["last_forecasts"])


def test_states_and_training_statistics_do_not_leak_between_episodes(baseline):
    before = baseline.to_dict()
    config = FinalForecastConfig(hidden_dim=5, tail_start_step=8)
    sequence = stream().requires_grad_()
    left = run_final_forecasts(sequence, baseline, config)
    run_final_forecasts(stream(19), baseline, config)
    right = run_final_forecasts(sequence, baseline, config)
    for arm in ARMS:
        forecast = left["arms"][arm]["last_forecasts"]
        assert not forecast.requires_grad and forecast.grad_fn is None
        assert torch.equal(forecast, right["arms"][arm]["last_forecasts"])
    for key in ("mean", "std", "coefficients"):
        assert torch.equal(baseline.to_dict()[key], before[key])
    assert sequence.grad is None
    assert left["final_forecasts_scored"] is False
    assert left["outer_training_performed"] is False


def test_fifo_and_memory_capacity_are_bounded_independently_of_episode_length(baseline):
    config = FinalForecastConfig(hidden_dim=5, bank_capacity=2, tail_start_step=8)
    short = run_final_forecasts(stream(steps=24), baseline, config)
    long = run_final_forecasts(stream(steps=48), baseline, config)
    assert short["shared_fifo_allocation_bytes"] == long["shared_fifo_allocation_bytes"]
    for arm in ARMS:
        for key in ("state_bytes", "state_capacity_bytes", "pending_bytes"):
            assert short["arms"][arm][key] == long["arms"][arm][key]
    assert short["arms"]["residual"]["updates"] == 23 + 21
    assert short["arms"]["recent_tail_residual"]["updates"] == 15 + 13


@pytest.mark.parametrize("arms", [("recent_only",), ("nearest_bank",), ("residual",), ("recent_tail_residual",)])
def test_arm_subsets_match_joint_run(baseline, arms):
    config = FinalForecastConfig(hidden_dim=5, tail_start_step=8)
    joint = run_final_forecasts(stream(), baseline, config)
    single = run_final_forecasts(stream(), baseline, config, arms=arms)
    torch.testing.assert_close(single["arms"][arms[0]]["last_forecasts"], joint["arms"][arms[0]]["last_forecasts"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"hidden_dim": 0}, {"bank_capacity": 0}, {"learning_rate": float("nan")},
    {"seed": -1}, {"tail_start_step": -1},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FinalForecastConfig(**kwargs)


def test_nonfinite_inputs_and_duplicate_arms_are_rejected(baseline):
    bad = stream()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run_final_forecasts(bad, baseline)
    with pytest.raises(ValueError, match="distinct"):
        run_final_forecasts(stream(), baseline, arms=("residual", "residual"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu(baseline):
    config = FinalForecastConfig(hidden_dim=5, bank_capacity=3, tail_start_step=8)
    cpu = run_final_forecasts(stream(), baseline, config)
    gpu = run_final_forecasts(stream(), baseline, config, device="cuda")
    for arm in ARMS:
        torch.testing.assert_close(cpu["arms"][arm]["last_forecasts"], gpu["arms"][arm]["last_forecasts"], rtol=1e-4, atol=1e-5)
