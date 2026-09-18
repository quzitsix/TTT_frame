"""Mathematical and streaming invariants for Spatial-TTT fast weights."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from ttt_frame.spatial_memory import (
    FastWeightConfig,
    FastWeightState,
    SwiGLUFastWeightMemory,
    swiglu_fast_weight_gradients,
    swiglu_read,
)


def make_memory(**kwargs):
    options = dict(dim=8, num_heads=2, chunk_size=4, ttt_scale_init=0.2, seed=31)
    options.update(kwargs)
    return SwiGLUFastWeightMemory(FastWeightConfig(**options))


def inputs(length=11, batch=1, dim=8):
    generator = torch.Generator().manual_seed(41)
    return tuple(torch.randn(batch, length, dim, generator=generator) for _ in range(4))


def assert_same_state(actual, expected, exact=False):
    left, right = actual.to_tensor_dict(), expected.to_tensor_dict()
    assert left.keys() == right.keys()
    for key in left:
        if exact or left[key].dtype == torch.int64:
            assert torch.equal(left[key], right[key]), key
        else:
            torch.testing.assert_close(left[key], right[key], rtol=2e-5, atol=2e-6, msg=key)


def test_manual_associative_gradient_matches_autograd_for_each_rate():
    """Each matrix gets its own per-token rate, rather than an MSE residual."""
    generator = torch.Generator().manual_seed(17)
    w0 = torch.randn(2, 5, 3, generator=generator, dtype=torch.double, requires_grad=True)
    w1 = torch.randn(2, 3, 5, generator=generator, dtype=torch.double, requires_grad=True)
    w2 = torch.randn(2, 5, 3, generator=generator, dtype=torch.double, requires_grad=True)
    k, v = [torch.randn(2, 7, 3, generator=generator, dtype=torch.double) for _ in range(2)]
    rates = [torch.rand(2, 7, 1, generator=generator, dtype=torch.double) for _ in range(3)]
    actual = swiglu_fast_weight_gradients(w0, w1, w2, k, v, *rates)
    # Independent autograd objective, intentionally expressed without the helper read.
    prediction = (F.silu(k @ w0.transpose(1, 2)) * (k @ w2.transpose(1, 2))) @ w1.transpose(1, 2)
    for weight, rate, gradient in zip((w0, w1, w2), rates, actual):
        expected, = torch.autograd.grad((prediction * v * rate).sum(), weight, retain_graph=True)
        torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_muon", [False, True])
def test_read_then_write_has_no_future_chunk_leakage(use_muon):
    memory = make_memory(use_muon=use_muon)
    q, k, v, hidden = inputs(length=12)
    original, original_state = memory(q, k, v, hidden)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, 4:8] *= -2
    changed_v[:, 4:8] += 3
    changed, changed_state = memory(q, changed_k, changed_v, hidden)
    # Chunk two reads the same weights even though its own write targets differ.
    torch.testing.assert_close(changed[:, :8], original[:, :8], rtol=0, atol=0)
    assert not torch.allclose(changed[:, 8:], original[:, 8:])
    assert not torch.allclose(changed_state.w0, original_state.w0)


@pytest.mark.parametrize("use_muon", [False, True])
@pytest.mark.parametrize("splits", [(1, 3, 2, 5), (4, 4, 3), (7, 4)])
def test_arbitrary_call_splits_equal_single_stream(use_muon, splits):
    memory = make_memory(use_muon=use_muon)
    values = inputs()
    expected_output, expected_state = memory(*values)
    state, pieces, start = None, [], 0
    for length in splits:
        output, state = memory(*(x[:, start:start + length] for x in values), state)
        pieces.append(output)
        start += length
        assert state.pending_tokens < memory.config.chunk_size
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected_output, rtol=2e-5, atol=2e-6)
    assert_same_state(state, expected_state)
    assert state.tokens == 11 and state.updates == 2 and state.pending_tokens == 3


def test_read_only_never_mutates_or_commits_pending_observations():
    memory = make_memory(use_muon=False)
    _, state = memory(*inputs(length=3))
    snapshot = state.clone()
    read_output, returned = memory(*inputs(length=10), state, write=False, flush=True)
    assert returned is state
    assert read_output.shape == (1, 10, 8)
    assert_same_state(state, snapshot, exact=True)


def test_writes_do_not_mutate_previous_state_or_initial_parameters():
    memory = make_memory(use_muon=False)
    state = memory.initial_state()
    snapshot = state.clone()
    initial_w0 = memory.w0.detach().clone()
    _, updated = memory(*inputs(length=9), state)
    assert_same_state(state, snapshot, exact=True)
    assert torch.equal(memory.w0, initial_w0)
    assert not torch.allclose(updated.w0, state.w0)


def test_flush_commits_tail_once_and_empty_input_can_flush():
    memory = make_memory(use_muon=False)
    data = inputs(length=3)
    output, pending = memory(*data)
    assert pending.updates == 0 and pending.pending_tokens == 3
    assert torch.equal(pending.w0, memory.initial_state().w0)
    flushed = memory.flush(pending)
    assert flushed.updates == 1 and flushed.pending_tokens == 0 and flushed.tokens == 3
    assert not torch.allclose(flushed.w0, pending.w0)
    assert memory.flush(flushed) is flushed
    direct_output, direct_state = memory(*data, flush=True)
    torch.testing.assert_close(output, direct_output)
    assert_same_state(flushed, direct_state, exact=True)
    empty_output, empty_state = memory(*(x[:, :0] for x in data), pending, flush=True)
    assert empty_output.shape == (1, 0, 8)
    assert_same_state(flushed, empty_state, exact=True)


@pytest.mark.parametrize("use_muon", [False, True])
def test_zero_lr_is_exact_noop_including_nonzero_momentum(use_muon):
    memory = make_memory(base_lr=0, use_muon=use_muon)
    state = memory.initial_state()
    state = replace(state, **{
        f"dw{i}_momentum": torch.ones_like(getattr(state, f"w{i}")) for i in range(3)
    })
    expected, _ = memory(*inputs(), state, write=False)
    actual, updated = memory(*inputs(), state, flush=True)
    # Different GEMM chunk shapes may round the same mathematical read slightly.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    for i in range(3):
        assert torch.equal(getattr(updated, f"w{i}"), getattr(state, f"w{i}"))
        assert torch.equal(getattr(updated, f"dw{i}_momentum"), getattr(state, f"dw{i}_momentum"))


@pytest.mark.parametrize("use_muon", [False, True])
def test_batch_members_have_independent_memory_and_match_separate_calls(use_muon):
    memory = make_memory(use_muon=use_muon, inter_multi=1.5)
    data = inputs(batch=3)
    batched_output, batched_state = memory(*data, flush=True)
    assert batched_state.w0.shape == (6, 6, 4)
    assert batched_state.w1.shape == (6, 4, 6)
    for batch in range(3):
        output, state = memory(*(x[batch:batch + 1] for x in data), flush=True)
        torch.testing.assert_close(batched_output[batch:batch + 1], output, rtol=2e-5, atol=2e-6)
        for i in range(3):
            torch.testing.assert_close(
                getattr(batched_state, f"w{i}")[batch * 2:(batch + 1) * 2],
                getattr(state, f"w{i}"), rtol=2e-5, atol=2e-6,
            )
    batched_state.validate(memory.config, batch_size=3)


@pytest.mark.parametrize("use_muon", [False, True])
def test_offline_loss_backpropagates_through_writes_and_controls(use_muon):
    memory = make_memory(use_muon=use_muon)
    q, k, v, hidden = [x.requires_grad_() for x in inputs(length=12)]
    output, state = memory(q, k, v, hidden)
    target = torch.randn_like(output[:, 8:])
    (output[:, 8:] * target).sum().backward()
    for name in (
        "w0", "w1", "w2", "lr_proj.weight", "momentum_proj.0.weight",
        "ttt_scale_proj.weight", "ttt_norm.weight",
    ):
        gradient = dict(memory.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
    assert k.grad[:, :8].abs().sum() > 0
    assert v.grad[:, :8].abs().sum() > 0
    # No implicit detach in training; explicit inference detach does break the graph.
    assert state.w0.grad_fn is not None
    assert state.detached().w0.grad_fn is None


def test_default_zero_gate_preserves_output_but_still_writes_and_can_train_gate():
    memory = make_memory(ttt_scale_init=0, use_muon=False)
    output, state = memory(*inputs())
    assert torch.count_nonzero(output) == 0
    assert not torch.allclose(state.w0, memory.initial_state().w0)
    output.sum().backward()
    assert memory.ttt_scale_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("use_muon", [False, True])
def test_runtime_is_fp32_and_row_norms_stay_bounded_over_stream(use_muon):
    memory = make_memory(use_muon=use_muon).to(dtype=torch.bfloat16)
    state = None
    for _ in range(8):
        with torch.no_grad():
            output, state = memory(*(x.bfloat16() for x in inputs(length=5)), state, flush=True)
    assert output.dtype == torch.bfloat16
    for i in range(3):
        weight = getattr(state, f"w{i}")
        assert weight.dtype == torch.float32 and torch.isfinite(weight).all()
        torch.testing.assert_close(
            weight.norm(dim=2, keepdim=True), getattr(state, f"w{i}_norm"),
            rtol=5e-5, atol=5e-5,
        )


def test_checkpoint_roundtrip_preserves_pending_and_future_writes():
    memory = make_memory(use_muon=False)
    with torch.no_grad():
        _, state = memory(*inputs(length=7))
        payload = state.to_tensor_dict()
        restored = FastWeightState.from_tensor_dict(payload)
        restored.validate(memory.config, batch_size=1)
        assert_same_state(restored, state, exact=True)
        out1, state1 = memory(*inputs(length=6), state)
        out2, state2 = memory(*inputs(length=6), restored)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    assert_same_state(state1, state2, exact=True)
    # Serialization must not share writable storage with live memory.
    payload["w0"].zero_()
    assert torch.count_nonzero(state.w0) > 0


@pytest.mark.parametrize("corruption", ["nan", "shape", "missing_rate", "count", "dtype", "unknown"])
def test_checkpoint_rejects_malformed_state(corruption):
    memory = make_memory(use_muon=False)
    _, state = memory(*inputs(length=3))
    payload = state.to_tensor_dict()
    if corruption == "nan":
        payload["w1"][0, 0, 0] = float("nan")
    elif corruption == "shape":
        payload["w1"] = payload["w1"][:, :, :1]
    elif corruption == "missing_rate":
        del payload["pending_lr0"]
    elif corruption == "count":
        payload["tokens"] = torch.tensor(-1, dtype=torch.int64)
    elif corruption == "dtype":
        payload["w0"] = payload["w0"].bfloat16()
    else:
        payload["mystery"] = torch.tensor(1)
    with pytest.raises(ValueError):
        FastWeightState.from_tensor_dict(payload)


def test_associative_write_improves_alignment_at_written_keys():
    """A learning update should encode its supplied values, not merely move."""
    memory = make_memory(use_muon=False, use_momentum=False, base_lr=0.02)
    q, k, v, hidden = inputs(length=3)
    before = memory.initial_state()
    keys, values = (memory._heads(x, memory.fw_head_dim) for x in (k, v))
    old_score = (swiglu_read(before.w0, before.w1, before.w2, keys) * values).sum()
    _, after = memory(q, k, v, hidden, before, flush=True)
    new_score = (swiglu_read(after.w0, after.w1, after.w2, keys) * values).sum()
    assert new_score > old_score
