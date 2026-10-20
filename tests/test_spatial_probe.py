"""CPU preflight checks for inverse reading of frozen Spatial fast weights."""

from dataclasses import fields

import pytest
import torch

from ttt_frame.spatial_memory import FastWeightConfig, FastWeightState, SwiGLUFastWeightMemory
from ttt_frame.spatial_probe import (
    association_delta,
    compare_inverse_address_candidates,
    projected_inverse_address_search,
)


def _memory_and_write():
    torch.manual_seed(17)
    memory = SwiGLUFastWeightMemory(FastWeightConfig(
        dim=16,
        num_heads=2,
        chunk_size=8,
        base_lr=0.05,
        use_muon=False,
        use_momentum=False,
        ttt_scale_init=0.1,
        seed=11,
    ))
    q = torch.randn(1, 6, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    hidden = torch.randn_like(q)
    reset = memory.initial_state()
    _, current = memory(q, k, v, hidden, reset, flush=True)
    keys = memory._heads(k, memory.fw_head_dim)
    values = memory._heads(v, memory.fw_head_dim)
    return memory, reset, current, keys, values


def _snapshot(state: FastWeightState):
    return {
        field.name: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for field in fields(state)
        for value in (getattr(state, field.name),)
    }


def _assert_snapshot(state: FastWeightState, snapshot) -> None:
    for name, expected in snapshot.items():
        actual = getattr(state, name)
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected), name
        else:
            assert actual == expected, name


def _scalar_states() -> tuple[FastWeightState, FastWeightState]:
    one = torch.ones(1, 1, 1)
    zero = torch.zeros_like(one)
    current = FastWeightState(
        w0=one.clone(), w1=one.clone(), w2=one.clone(),
        w0_norm=one.clone(), w1_norm=one.clone(), w2_norm=one.clone(),
    )
    reset = FastWeightState(
        w0=zero, w1=one.clone(), w2=one.clone(),
        w0_norm=zero.clone(), w1_norm=one.clone(), w2_norm=one.clone(),
    )
    return current, reset


def test_identical_current_and_reset_have_zero_association_delta():
    _, reset, _, keys, values = _memory_and_write()
    score = association_delta(reset, reset, keys, values)
    per_token = association_delta(reset, reset, keys, values, reduction="none")
    assert score.item() == 0.0
    assert torch.count_nonzero(per_token).item() == 0


def test_associative_write_increases_dot_product_at_written_keys():
    _, reset, current, keys, values = _memory_and_write()
    assert association_delta(current, reset, keys, values).item() > 0
    # A single-token-per-head input is accepted without a token dimension.
    assert association_delta(current, reset, keys[:, 0], values[:, 0]).ndim == 0


@pytest.mark.parametrize(("name", "value"), (("query", float("nan")),
                                               ("query", float("inf")),
                                               ("target", float("-inf"))))
def test_nonfinite_query_or_target_is_rejected(name, value):
    _, reset, current, keys, values = _memory_and_write()
    query, target = keys[:, :1].clone(), values[:, :1].clone()
    (query if name == "query" else target)[0, 0, 0] = value
    with pytest.raises(ValueError, match=f"{name} must contain only finite values"):
        association_delta(current, reset, query, target)
    with pytest.raises(ValueError, match=f"{name} must contain only finite values"):
        projected_inverse_address_search(
            current, reset, query, target, radius=0.1, steps=1, step_size=0.1
        )


def test_inverse_search_respects_each_address_ball_and_does_not_mutate_states():
    _, reset, current, keys, values = _memory_and_write()
    current_before, reset_before = _snapshot(current), _snapshot(reset)
    origin = keys[:, :2].clone()
    radius = 0.075
    result = projected_inverse_address_search(
        current,
        reset,
        origin,
        values[:, :2],
        radius=radius,
        steps=4,
        step_size=0.1,
        regularization=0.01,
    )
    assert result.steps == 4
    assert result.direct_score.ndim == result.search_score.ndim == 0
    assert torch.all(result.displacement_norm <= radius + 1e-6)
    assert torch.allclose((result.query - origin).norm(dim=-1), result.displacement_norm)
    assert result.search_objective >= result.direct_objective - 1e-5
    _assert_snapshot(current, current_before)
    _assert_snapshot(reset, reset_before)


def test_candidate_reordering_only_reorders_scores_and_addresses():
    _, reset, current, keys, values = _memory_and_write()
    query = keys[:, 0]
    targets = torch.stack((values[:, 0], -values[:, 0], torch.roll(values[:, 0], 1, 0)))
    options = dict(radius=0.1, steps=3, step_size=0.05, regularization=0.02)
    original = compare_inverse_address_candidates(current, reset, query, targets, **options)
    permutation = torch.tensor([2, 0, 1])
    permuted = compare_inverse_address_candidates(
        current, reset, query, targets[permutation], **options
    )
    for name in (
        "direct_scores",
        "search_scores",
        "direct_objectives",
        "search_objectives",
        "queries",
        "displacement_norms",
    ):
        torch.testing.assert_close(getattr(permuted, name), getattr(original, name)[permutation])


def test_zero_step_search_reports_the_direct_query_and_score():
    _, reset, current, keys, values = _memory_and_write()
    result = projected_inverse_address_search(
        current, reset, keys[:, 0], values[:, 0],
        radius=0.0, steps=0, step_size=0.0,
    )
    assert torch.equal(result.query, keys[:, 0])
    assert torch.equal(result.search_score, result.direct_score)
    assert torch.count_nonzero(result.displacement_norm).item() == 0


def test_search_keeps_direct_query_when_an_oversized_step_is_worse():
    current, reset = _scalar_states()
    query = torch.tensor([[-1.0]])
    target = torch.ones_like(query)
    assert association_delta(current, reset, query, target) > 0
    result = projected_inverse_address_search(
        current,
        reset,
        query,
        target,
        radius=100.0,
        steps=1,
        step_size=100.0,
    )
    assert torch.equal(result.query, query)
    assert torch.equal(result.search_score, result.direct_score)
    assert torch.equal(result.search_objective, result.direct_objective)


def test_search_locally_enables_query_gradients_inside_no_grad():
    _, reset, current, keys, values = _memory_and_write()
    with torch.no_grad():
        result = projected_inverse_address_search(
            current,
            reset,
            keys[:, :1],
            values[:, :1],
            radius=0.1,
            steps=2,
            step_size=0.05,
        )
    assert torch.isfinite(result.search_score)
    assert result.search_objective >= result.direct_objective
