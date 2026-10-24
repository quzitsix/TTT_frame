import pytest
import torch

from ttt_frame.idea1_mcq_readout import (
    SharedRidgeReadout, candidate_prompt, pool_forecasts,
)


def data(seed=8, cases=9, arms=7, horizons=4, tokens=8, feature_dim=6, text_dim=11):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(cases, arms, horizons, tokens, feature_dim, generator=generator)
    options = torch.randn(cases, 6, text_dim, generator=generator)
    labels = torch.arange(cases) % 6
    return pool_forecasts(x), options, labels


def test_pool_spatial_tokens_and_concatenate_horizons():
    values = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).reshape(2, 4, 3, 5)
    pooled = pool_forecasts(values)
    expected = values.mean(dim=-2).flatten(start_dim=-2)
    assert pooled.shape == (2, 20)
    torch.testing.assert_close(pooled, expected.double())


def test_shared_readout_scores_each_arm_and_options_permutation_equivariantly():
    x, options, labels = data()
    model = SharedRidgeReadout.fit(
        x, options, labels, ridge_lambda=0.1,
        train_ids=[f"case-{index}" for index in range(len(x))],
        arm_names=[f"arm-{index}" for index in range(x.shape[1])],
    )
    scores = model.predict_scores(x, options)
    assert scores.shape == (len(x), 7, 6)
    assert model.independent_train_cases == len(x)
    assert model.train_rows == len(x) * 7
    order = torch.tensor([3, 1, 5, 0, 4, 2])
    permuted = model.predict_scores(x, options[:, order])
    torch.testing.assert_close(permuted, scores[:, :, order])
    blind = model.blind_scores(options)
    assert blind.shape == (len(x), 6)
    torch.testing.assert_close(model.blind_scores(options[:, order]), blind[:, order])


def test_fit_rejects_invalid_labels_and_duplicate_case_ids():
    x, options, labels = data(cases=4)
    with pytest.raises(ValueError, match="outside available options"):
        SharedRidgeReadout.fit(x, options, torch.tensor([0, 1, 2, 6]), ridge_lambda=1)
    with pytest.raises(ValueError, match="uniquely identify"):
        SharedRidgeReadout.fit(x, options, labels, ridge_lambda=1, train_ids=["x"] * len(x))


def test_serialization_preserves_predictions_and_training_provenance():
    x, options, labels = data(cases=5)
    model = SharedRidgeReadout.fit(x, options, labels, ridge_lambda=10, train_ids=list("abcde"))
    restored = SharedRidgeReadout.from_dict(model.to_dict())
    torch.testing.assert_close(restored.predict_scores(x, options), model.predict_scores(x, options))
    assert restored.to_dict()["independent_train_cases"] == 5


def test_candidate_prompt_does_not_include_option_letter_or_position():
    prompt = candidate_prompt("接下来会做什么？", "我拿起杯子")
    assert prompt == "问题：接下来会做什么？\n候选动作：我拿起杯子"
    assert all(letter not in prompt for letter in "ABCDEF")
    with pytest.raises(ValueError):
        candidate_prompt("question", "  ")


def test_repeating_arms_does_not_change_fit_or_strengthen_supervision():
    x, options, labels = data(cases=8, arms=2)
    original = SharedRidgeReadout.fit(x, options, labels, ridge_lambda=0.3)
    repeated = SharedRidgeReadout.fit(x.repeat(1, 3, 1), options, labels, ridge_lambda=0.3)
    assert original.independent_train_cases == repeated.independent_train_cases == 8
    assert repeated.train_rows == 3 * original.train_rows
    torch.testing.assert_close(original.coefficients, repeated.coefficients, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(original.predict_scores(x, options), repeated.predict_scores(x, options),
                               rtol=1e-10, atol=1e-10)


def test_validation_predictions_cannot_change_training_statistics():
    x, options, labels = data()
    model = SharedRidgeReadout.fit(x, options, labels, ridge_lambda=1)
    before = model.to_dict()
    model.predict_scores(x * 1000 + 3000, options * 2000 - 4000)
    model.blind_scores(options * 10000)
    after = model.to_dict()
    for key in ("feature_mean", "feature_scale", "text_mean", "text_scale",
                "coefficients", "target_mean"):
        assert torch.equal(before[key], after[key])
    torch.testing.assert_close(model.feature_mean, x.flatten(0, 1).mean(0))
    torch.testing.assert_close(model.text_mean, options.double().flatten(0, 1).mean(0))


def test_dual_solver_matches_primal_regularized_least_squares():
    x, options, labels = data(cases=12, arms=2, horizons=1, feature_dim=3, text_dim=5)
    model = SharedRidgeReadout.fit(x, options, labels, ridge_lambda=0.7)
    design = (x.flatten(0, 1) - model.feature_mean) / model.feature_scale
    targets = model.transform_options(options)[torch.arange(len(x)), labels]
    targets = (targets - targets.mean(0)).repeat_interleave(2, dim=0)
    expected = torch.linalg.solve(
        design.T @ design + len(design) * 0.7 * torch.eye(3, dtype=torch.float64),
        design.T @ targets,
    )
    torch.testing.assert_close(model.coefficients, expected, rtol=1e-10, atol=1e-10)


def test_learns_semantic_mapping_without_option_position_features():
    # Independent random order per case prevents learning any answer letter.
    generator = torch.Generator().manual_seed(41)
    candidates = torch.eye(6, dtype=torch.float64)
    train_target = torch.arange(120) % 6
    train_orders = torch.stack([torch.randperm(6, generator=generator) for _ in range(120)])
    train_options = candidates[train_orders]
    train_labels = (train_orders == train_target[:, None]).long().argmax(-1)
    x = candidates[train_target][:, None, :].repeat(1, 7, 1)
    model = SharedRidgeReadout.fit(x, train_options, train_labels, ridge_lambda=0.01)
    test_target = torch.arange(30) % 6
    test_orders = torch.stack([torch.randperm(6, generator=generator) for _ in range(30)])
    scores = model.predict_scores(candidates[test_target], candidates[test_orders])
    predicted_semantics = test_orders.gather(1, scores.argmax(-1)[:, None]).squeeze(1)
    assert torch.equal(predicted_semantics, test_target)
