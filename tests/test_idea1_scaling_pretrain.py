"""Train-only pair construction, independent targets and initialization learning."""
import pytest
import torch

from ttt_frame.forecast_residual import RidgeRecentPredictor
from ttt_frame.idea1_scaling_pretrain import (
    FrozenPairDataset, OfflineForecastMLPs, OfflinePretrainConfig, train_step, validation_losses,
)


@pytest.fixture
def baseline():
    return RidgeRecentPredictor(
        horizons=(1, 3), window_steps=3, mean=torch.tensor([0.2, -0.1, 0.5]),
        std=torch.tensor([0.5, 2.0, 1.5]), coefficients=torch.randn(
            2, 7, 3, generator=torch.Generator().manual_seed(1)) / 7,
        tokens=2, training_rows=(10, 10), standardize=True, ridge=0.01,
    )


def streams(length=12):
    return [torch.randn(length, 2, 3, generator=torch.Generator().manual_seed(seed))
            for seed in (2, 3)]


def test_pair_targets_match_frozen_recent_reference_and_never_cross_episode(baseline):
    sequences = streams()
    dataset = FrozenPairDataset(sequences, baseline)
    cases, origins = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 2, 5, 8])
    actual = dataset.gather(cases, origins)
    for index, (case, origin) in enumerate(zip(cases.tolist(), origins.tolist())):
        z = baseline.normalize(sequences[case])
        torch.testing.assert_close(actual["keys"][index], z[origin])
        frozen = baseline.predict_standardized(z[max(0, origin-2):origin+1])
        target = torch.stack([z[origin+h] for h in baseline.horizons])
        torch.testing.assert_close(actual["targets"][0, :, index], target)
        torch.testing.assert_close(actual["targets"][1, :, index], target-frozen)
    with pytest.raises(ValueError, match="cross"):
        dataset.gather(torch.tensor([0]), torch.tensor([9]))


def test_target_observation_cannot_change_saved_cue_or_recent_prediction(baseline):
    original = streams()
    changed = [value.clone() for value in original]
    changed[0][8:] += 10
    left = FrozenPairDataset(original, baseline).gather(torch.tensor([0]), torch.tensor([7]))
    right = FrozenPairDataset(changed, baseline).gather(torch.tensor([0]), torch.tensor([7]))
    assert torch.equal(left["keys"], right["keys"])
    torch.testing.assert_close(right["targets"][0]-left["targets"][0],
                               right["targets"][1]-left["targets"][1])


def test_unequal_sequence_lengths_follow_same_within_case_rule(baseline):
    sequences = [streams()[0], streams(15)[1]]
    data = FrozenPairDataset(sequences, baseline)
    assert data.stacked is None
    batch = data.gather(torch.tensor([1]), torch.tensor([11]))
    z = baseline.normalize(sequences[1])
    torch.testing.assert_close(batch["targets"][0, :, 0], torch.stack([z[12], z[14]]))


def test_sampling_is_deterministic_and_reports_only_valid_origins(baseline):
    data = FrozenPairDataset(streams(), baseline)
    left = data.sample(128, torch.Generator().manual_seed(4))
    right = data.sample(128, torch.Generator().manual_seed(4))
    for key in left:
        assert torch.equal(left[key], right[key])
    assert int(left["origins"].max()) < 9
    assert set(left["case_indices"].tolist()) == {0, 1}


def test_initial_weights_match_expected_shapes_zero_output_and_gain(baseline):
    config = OfflinePretrainConfig(hidden_dim=512)
    model = OfflineForecastMLPs(2, 3, config)
    assert model.gain == 0.5
    weights = model.initial_weights()
    assert set(weights) == {"plain", "residual"}
    assert weights["plain"]["w_in"].shape == (2, 512, 3)
    assert weights["plain"]["w_out"].shape == (2, 3, 512)
    assert torch.equal(weights["plain"]["w_in"], weights["residual"]["w_in"])
    assert torch.equal(model(torch.randn(4, 2, 3)), torch.zeros(2, 2, 4, 2, 3))
    weights["plain"]["w_in"].zero_()
    assert not torch.equal(weights["plain"]["w_in"], model.w_in[0])


def test_training_reduces_each_target_loss_without_changing_data_or_baseline(baseline):
    before = baseline.to_dict()
    sequences = streams()
    frozen_inputs = [value.clone() for value in sequences]
    data = FrozenPairDataset(sequences, baseline)
    batch = data.gather(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 3, 5]))
    config = OfflinePretrainConfig(hidden_dim=16, learning_rate=0.02, batch_size=4)
    model = OfflineForecastMLPs(2, 3, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    initial = validation_losses(model, batch)
    for _ in range(80):
        result = train_step(model, optimizer, batch, config)
    final = validation_losses(model, batch)
    assert all(final[name]["mse"] < initial[name]["mse"] * 0.5 for name in ("plain", "residual"))
    assert result["gradient_norm_before_clip"] >= 0
    assert not torch.equal(model.w_in[0], model.w_in[1])
    for key in ("mean", "std", "coefficients"):
        assert torch.equal(before[key], baseline.to_dict()[key])
    for left, right in zip(sequences, frozen_inputs):
        assert torch.equal(left, right)


@pytest.mark.parametrize("kwargs", [
    {"hidden_dim": 0}, {"batch_size": 0}, {"learning_rate": 0},
    {"weight_decay": -1}, {"grad_clip": float("nan")}, {"seed": -1},
])
def test_invalid_pretraining_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        OfflinePretrainConfig(**kwargs)


def test_nonfinite_or_short_data_rejected(baseline):
    with pytest.raises(ValueError):
        FrozenPairDataset([torch.zeros(3, 2, 3)], baseline)
    bad = streams()[0]
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        FrozenPairDataset([bad], baseline)
