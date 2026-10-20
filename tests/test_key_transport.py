"""CPU preflight tests for visual-key coordinate transport."""

from __future__ import annotations

import pytest
import torch

from ttt_frame.key_transport import (
    alignment_metrics,
    apply_key_transport,
    fit_heldout_metrics,
    fit_key_transport,
    regularized_orthogonal_procrustes,
    transport_swiglu_input_weights,
)
from ttt_frame.spatial_memory import swiglu_read


def random_orthogonal(dimension: int, *, seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(dimension, dimension, generator=generator, dtype=torch.double)
    orthogonal, _ = torch.linalg.qr(matrix)
    return orthogonal


def test_regularized_procrustes_is_orthogonal_and_improves_heldout_matches():
    generator = torch.Generator().manual_seed(19)
    expected = random_orthogonal(6)
    old = torch.randn(80, 6, generator=generator, dtype=torch.double)
    new = apply_key_transport(old, expected)
    fit_old, heldout_old = old[:48], old[48:]
    fit_new, heldout_new = new[:48], new[48:]

    actual = regularized_orthogonal_procrustes(
        fit_old, fit_new, regularization=1e-6
    )
    torch.testing.assert_close(actual.T @ actual, torch.eye(6, dtype=torch.double))
    report = fit_heldout_metrics(
        actual, fit_old, fit_new, heldout_old, heldout_new
    )
    assert report.fit.transported_rmse < 1e-6
    assert report.heldout is not None
    assert report.heldout.relative_rmse_improvement > 0.999999
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_regularization_biases_an_underdetermined_fit_toward_identity():
    old = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.double)
    new = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.double)
    unregularized = regularized_orthogonal_procrustes(old, new)
    regularized = regularized_orthogonal_procrustes(old, new, regularization=10.0)
    identity = torch.eye(3, dtype=torch.double)
    assert torch.linalg.matrix_norm(regularized - identity) < torch.linalg.matrix_norm(
        unregularized - identity
    )


def test_swiglu_input_transport_preserves_the_function_in_new_coordinates():
    generator = torch.Generator().manual_seed(29)
    batch_heads, tokens, dimension, hidden = 3, 7, 5, 9
    transport = random_orthogonal(dimension)
    w0 = torch.randn(
        batch_heads, hidden, dimension, generator=generator, dtype=torch.double
    )
    w1 = torch.randn(
        batch_heads, dimension, hidden, generator=generator, dtype=torch.double
    )
    w2 = torch.randn(
        batch_heads, hidden, dimension, generator=generator, dtype=torch.double
    )
    old_keys = torch.randn(
        batch_heads, tokens, dimension, generator=generator, dtype=torch.double
    )
    original_w0, original_w2 = w0.clone(), w2.clone()

    new_w0, new_w2 = transport_swiglu_input_weights(w0, w2, transport)
    new_keys = apply_key_transport(old_keys, transport)
    before = swiglu_read(w0, w1, w2, old_keys)
    after = swiglu_read(new_w0, w1, new_w2, new_keys)

    torch.testing.assert_close(after, before, rtol=2e-12, atol=2e-12)
    assert torch.equal(w0, original_w0) and torch.equal(w2, original_w2)


def test_swiglu_transport_supports_one_orthogonal_map_per_head():
    generator = torch.Generator().manual_seed(31)
    w0 = torch.randn(2, 4, 3, generator=generator, dtype=torch.double)
    w2 = torch.randn(2, 4, 3, generator=generator, dtype=torch.double)
    transports = torch.stack([random_orthogonal(3, seed=7), random_orthogonal(3, seed=8)])
    actual0, actual2 = transport_swiglu_input_weights(w0, w2, transports)
    for head in range(2):
        expected0, expected2 = transport_swiglu_input_weights(
            w0[head], w2[head], transports[head]
        )
        torch.testing.assert_close(actual0[head], expected0)
        torch.testing.assert_close(actual2[head], expected2)


@pytest.mark.parametrize(
    ("scene_cut", "fit_count", "reason"),
    [(True, 8, "scene_cut"), (False, 2, "insufficient_fit_matches")],
)
def test_scene_cut_and_low_match_decisions_return_identity(scene_cut, fit_count, reason):
    generator = torch.Generator().manual_seed(37)
    old = torch.randn(fit_count, 4, generator=generator)
    new = torch.randn(fit_count, 4, generator=generator)
    decision = fit_key_transport(
        old, new, scene_cut=scene_cut, min_fit_matches=3
    )
    assert not decision.accepted and decision.reason == reason
    assert decision.candidate_transport is None and decision.metrics is None
    torch.testing.assert_close(decision.transport, torch.eye(4))


def test_rejected_heldout_fit_keeps_candidate_only_for_diagnostics():
    generator = torch.Generator().manual_seed(41)
    transport = random_orthogonal(4).float()
    fit_old = torch.randn(16, 4, generator=generator)
    heldout_old = torch.randn(8, 4, generator=generator)
    decision = fit_key_transport(
        fit_old,
        apply_key_transport(fit_old, transport),
        heldout_old=heldout_old,
        heldout_new=heldout_old,
        min_relative_improvement=0.1,
    )
    assert not decision.accepted
    assert decision.reason == "improvement_below_threshold"
    assert decision.candidate_transport is not None and decision.metrics is not None
    torch.testing.assert_close(decision.transport, torch.eye(4))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: regularized_orthogonal_procrustes(torch.ones(3, 4), torch.ones(3, 5)),
        lambda: regularized_orthogonal_procrustes(
            torch.tensor([[float("nan"), 0.0]]), torch.ones(1, 2)
        ),
        lambda: alignment_metrics(torch.ones(3, 2), torch.ones(3, 2), torch.eye(3)),
        lambda: transport_swiglu_input_weights(
            torch.ones(2, 3, 4), torch.ones(2, 3, 5), torch.eye(4)
        ),
        lambda: transport_swiglu_input_weights(
            torch.ones(2, 3, 4), torch.ones(2, 3, 4), torch.full((4, 4), float("inf"))
        ),
        lambda: transport_swiglu_input_weights(
            torch.ones(2, 3, 4), torch.ones(2, 3, 4), 2 * torch.eye(4)
        ),
    ],
)
def test_invalid_dimensions_nonfinite_values_and_nonorthogonal_maps_raise(operation):
    with pytest.raises(ValueError):
        operation()
