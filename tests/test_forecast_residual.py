"""Causal and bounded-state checks for the frozen-feature Idea 01 preflight."""

import pytest
import torch

from ttt_frame.forecast_residual import (
    ARMS,
    PredictiveRunConfig,
    RidgeRecentPredictor,
    run_prequential,
)


def stream(seed=3, steps=24, tokens=2, dim=4):
    return torch.randn(steps, tokens, dim, generator=torch.Generator().manual_seed(seed))


def fitted():
    return RidgeRecentPredictor.fit([stream(1), stream(2)], horizons=(1, 3), window_steps=3)


def test_baseline_stats_are_train_only_and_serialization_is_tensor_safe(tmp_path):
    train = [stream(1), stream(2)]
    baseline = RidgeRecentPredictor.fit(train, horizons=(1, 3))
    expected = torch.cat(train).mean((0, 1))
    torch.testing.assert_close(baseline.mean, expected)
    before = baseline.to_dict()
    run_prequential(stream(4) + 10, baseline, PredictiveRunConfig(hidden_dim=5),
                    arms=("recent_only",))
    for name in ("mean", "std", "coefficients"):
        assert torch.equal(baseline.to_dict()[name], before[name])
    path = tmp_path / "ridge.pt"
    torch.save(before, path)
    restored = RidgeRecentPredictor.from_dict(torch.load(path, weights_only=True))
    recent = baseline.normalize(train[0][:3])
    torch.testing.assert_close(restored.predict_standardized(recent),
                               baseline.predict_standardized(recent))


def test_fit_does_not_pair_different_streams_and_bounds_training_rows():
    tiny = [torch.ones(1, 2, 3), torch.zeros(1, 2, 3)]
    with pytest.raises(ValueError, match="within-stream"):
        RidgeRecentPredictor.fit(tiny, horizons=(1,))
    baseline = RidgeRecentPredictor.fit([stream(), stream(5)], horizons=(1, 3), max_fit_rows=12)
    assert baseline.training_rows == (12, 12)
    again = RidgeRecentPredictor.fit([stream(), stream(5)], horizons=(1, 3), max_fit_rows=12)
    assert torch.equal(baseline.coefficients, again.coefficients)


def test_suffix_changes_never_change_earlier_forecasts_for_any_arm():
    baseline = fitted()
    original, altered = stream(), stream()
    altered[12:] = altered[12:] * -2 + 1
    config = PredictiveRunConfig(hidden_dim=5, learning_rate=0.1, bank_capacity=3)
    left = run_prequential(original, baseline, config, keep_predictions=True)
    right = run_prequential(altered, baseline, config, keep_predictions=True)
    for arm in ARMS:
        torch.testing.assert_close(left["arms"][arm]["predictions"][:12],
                                   right["arms"][arm]["predictions"][:12], rtol=0, atol=0)


def test_prequential_scores_the_original_prediction_before_maturity_update():
    baseline = fitted()
    sequence = stream(7)
    config = PredictiveRunConfig(hidden_dim=5, score_start_step=17)
    result = run_prequential(sequence, baseline, config, keep_predictions=True)
    for arm in ARMS:
        rows = result["arms"][arm]
        errors = []
        for index, horizon in enumerate(baseline.horizons):
            for origin in range(config.score_start_step, len(sequence) - horizon):
                predicted = rows["predictions"][origin, index]
                errors.append(((predicted - sequence[origin + horizon]) / baseline.std)
                              .square().mean())
        assert rows["count"] == (24 - 17 - 1) + (24 - 17 - 3)
        assert rows["mse"] == pytest.approx(float(torch.stack(errors).mean()), rel=1e-5)
        assert rows["by_horizon"]["3"]["count"] == 4


def test_zero_learning_rate_is_recent_only_for_residual_and_reset():
    result = run_prequential(stream(), fitted(), PredictiveRunConfig(
        hidden_dim=5, learning_rate=0), keep_predictions=True)
    for arm in ("residual", "reset_residual"):
        assert torch.equal(result["arms"][arm]["predictions"],
                           result["arms"]["recent_only"]["predictions"])


def test_run_is_deterministic_episode_local_and_retains_no_training_graph():
    baseline = fitted()
    sequence = stream().requires_grad_()
    config = PredictiveRunConfig(hidden_dim=5, bank_capacity=2, seed=8)
    left = run_prequential(sequence, baseline, config)
    run_prequential(stream(19), baseline, config)
    right = run_prequential(sequence, baseline, config)
    for arm in ARMS:
        assert "predictions" not in left["arms"][arm]
        assert left["arms"][arm]["mse"] == right["arms"][arm]["mse"]
        forecast = left["arms"][arm]["last_forecasts"]
        assert not forecast.requires_grad and forecast.grad_fn is None
        assert torch.equal(forecast, right["arms"][arm]["last_forecasts"])
    assert sequence.grad is None
    bank = left["arms"]["explicit_bank"]
    assert bank["state_bytes"] == bank["state_capacity_bytes"] == 2 * 2 * 4 * 4 * 2 * 2
    for arm in ("plain_delayed", "residual", "reset_residual", "explicit_bank"):
        assert left["arms"][arm]["pending_bytes"] == 2 * 2 * 4 * 4 * (1 + 3)


def test_reset_does_not_retain_fast_weights_from_distant_past():
    baseline = fitted()
    first, second = stream(), stream()
    second[:8] += 2
    config = PredictiveRunConfig(hidden_dim=6, learning_rate=0.02)
    left = run_prequential(first, baseline, config, arms=("reset_residual", "residual"),
                           keep_predictions=True)
    right = run_prequential(second, baseline, config, arms=("reset_residual", "residual"),
                            keep_predictions=True)
    # Beyond longest horizon + baseline recent window, the reset arm can only
    # use identical recently matured cues, whereas accumulated memory differs.
    assert torch.equal(left["arms"]["reset_residual"]["predictions"][15:],
                       right["arms"]["reset_residual"]["predictions"][15:])
    assert not torch.equal(left["arms"]["residual"]["last_forecasts"],
                           right["arms"]["residual"]["last_forecasts"])


def test_final_forecast_targets_are_unseen_and_never_scored():
    result = run_prequential(stream(3, steps=2), fitted(), arms=("recent_only",))
    rows = result["arms"]["recent_only"]
    assert rows["count"] == 1
    assert rows["by_horizon"]["3"] == {"mse": None, "count": 0}
    assert rows["last_forecasts"].shape == (2, 2, 4)


def test_matured_residual_writes_improve_a_repeated_new_episode_pattern():
    # This tests mechanism only: recent training saw zeros, while the observed
    # episode repeatedly produces ones. Persistence is an intentionally strong
    # control here; no semantic or benchmark conclusion follows from this test.
    baseline = RidgeRecentPredictor.fit(
        [torch.zeros(20, 1, 4)], horizons=(1,), standardize=False,
    )
    result = run_prequential(
        torch.ones(80, 1, 4), baseline,
        PredictiveRunConfig(hidden_dim=8, learning_rate=0.05, score_start_step=50),
        arms=("recent_only", "residual", "reset_residual", "persistence"),
    )["arms"]
    assert result["residual"]["mse"] < result["reset_residual"]["mse"]
    assert result["reset_residual"]["mse"] < result["recent_only"]["mse"]
    assert result["persistence"]["mse"] == 0


@pytest.mark.parametrize("kwargs", [
    {"hidden_dim": 0}, {"bank_capacity": 0}, {"learning_rate": float("nan")},
    {"score_start_step": -1},
])
def test_invalid_runner_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PredictiveRunConfig(**kwargs)


def test_bad_features_and_duplicate_arms_are_rejected():
    baseline = fitted()
    bad = stream()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        run_prequential(bad, baseline)
    with pytest.raises(ValueError, match="distinct"):
        run_prequential(stream(), baseline, arms=("residual", "residual"))
