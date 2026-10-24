"""Native association-reader isolation, causal readout and target-order checks."""

from types import SimpleNamespace

import pytest
import torch

from ttt_frame.idea3_spatial_reader import (
    NativeQKVCapture, TextAddresses, batched_inverse_address_candidates,
    extract_text_addresses, score_addresses,
    snapshot_states, visual_key_diagnostic,
)
from ttt_frame.spatial_model import SpatialModelConfig, SpatialQwenMemory
from ttt_frame.spatial_memory import FastWeightConfig, FastWeightState, SwiGLUFastWeightMemory
from ttt_frame.spatial_probe import compare_inverse_address_candidates


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class CharacterTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        self.calls.append(text)
        return {"input_ids": torch.tensor([[1 + ord(char) % 100 for char in text]])}


def make_engine():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(4)
    config = transformers.Qwen3VLConfig(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=512,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                         pad_token_id=0),
        vision_config=dict(hidden_size=16, intermediate_size=32, depth=1, num_heads=2,
                           patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, deepstack_visual_indexes=[]),
        image_token_id=125, video_token_id=126, vision_start_token_id=123,
        vision_end_token_id=124,
    )
    model = transformers.Qwen3VLForConditionalGeneration(config).eval()
    controller = SpatialQwenMemory(model, SpatialModelConfig(
        num_heads=4, chunk_size=4, window_size=128, base_lr=0.05,
        use_muon=False, use_momentum=True, use_conv=False, ttt_scale_init=0.2,
    ))
    engine = SimpleNamespace(model=model, controller=controller, device=torch.device("cpu"),
                             processor=SimpleNamespace(tokenizer=CharacterTokenizer()),
                             position_offset=23, _clear_generation_state=lambda: None)
    with torch.no_grad(), controller.context("write"):
        model(input_ids=torch.tensor([[4, 5, 6, 7, 8]]), use_cache=False)
    return engine


def test_snapshot_owns_weights_omits_momentum_and_matches_learned_reset():
    engine = make_engine()
    current, reset = snapshot_states(engine), snapshot_states(engine, reset=True)
    for index, layer in engine.controller.layers.items():
        assert current[index].updates == layer.state.updates
        assert current[index].dw0_momentum is None
        assert current[index].pending_k is None
        assert current[index].w0.data_ptr() != layer.state.w0.data_ptr()
        torch.testing.assert_close(current[index].w0, layer.state.w0)
        torch.testing.assert_close(reset[index].w0, layer.memory.w0)
        current[index].w0.zero_()
        assert torch.count_nonzero(layer.state.w0)


def test_capture_restores_methods_after_failure_and_rejects_overlap():
    engine = make_engine()
    original = {index: layer._fast_qkv for index, layer in engine.controller.layers.items()}
    with pytest.raises(RuntimeError, match="deliberate"):
        with NativeQKVCapture(engine):
            with pytest.raises(RuntimeError, match="overlap"):
                with NativeQKVCapture(engine):
                    pass
            raise RuntimeError("deliberate")
    for index, layer in engine.controller.layers.items():
        assert layer._fast_qkv == original[index]
        assert "_fast_qkv" not in vars(layer)
        assert "_idea3_capture_active" not in vars(layer)


def test_text_targets_priors_and_queries_are_candidate_order_equivariant():
    engine = make_engine()
    original_states = {index: layer.state for index, layer in engine.controller.layers.items()}
    before = engine.controller.memory_state_dict()
    options = ["open the drawer", "lift a cup", "walk away"]
    first = extract_text_addresses(engine, "What happens next?", options)
    permutation = [2, 0, 1]
    second = extract_text_addresses(engine, "What happens next?", [options[i] for i in permutation])
    for index, layer in engine.controller.layers.items():
        assert layer.state is original_states[index]
        assert first.queries[index].shape == (4, 1, 8)
        assert first.targets[index].shape == (3, 4, 1, 8)
        torch.testing.assert_close(first.queries[index], second.queries[index], rtol=0, atol=0)
        torch.testing.assert_close(first.targets[index][permutation], second.targets[index], rtol=0, atol=0)
        torch.testing.assert_close(first.targets[index].norm(dim=-1), torch.ones(3, 4, 1))
    assert [first.prior_scores[i] for i in permutation] == list(second.prior_scores)
    assert first.position_offset == engine.position_offset == 23
    assert engine.processor.tokenizer.calls[:4] == [
        "Question: What happens next?\nCandidate action: ", *options]
    for name, value in before.items():
        assert torch.equal(value, engine.controller.memory_state_dict()[name]), name


def test_prior_matches_teacher_forced_likelihood_at_exact_candidate_boundary():
    engine = make_engine()
    addresses = extract_text_addresses(engine, "Next?", ["run"])
    tokenizer = engine.processor.tokenizer
    prefix = tokenizer("Question: Next?\nCandidate action: ", return_tensors="pt",
                       add_special_tokens=False)["input_ids"]
    option = tokenizer("run", return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = torch.cat((prefix, option), 1)
    for layer in engine.controller.layers.values():
        layer.state = None
    positions = torch.arange(ids.shape[1]) + engine.position_offset
    with torch.no_grad(), engine.controller.context("read"):
        output = engine.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                              position_ids=positions[None, None].expand(3, 1, -1),
                              use_cache=False, logits_to_keep=0)
    selected = output.logits[0, prefix.shape[1] - 1:-1].float().log_softmax(-1)
    expected = selected.gather(-1, option[0, :, None]).mean().item()
    assert addresses.prior_scores[0] == pytest.approx(expected, abs=1e-6)


def test_failed_text_forward_restores_episode_states_and_capture(monkeypatch):
    engine = make_engine()
    old_states = {index: layer.state for index, layer in engine.controller.layers.items()}

    def fail(*_, **__):
        raise RuntimeError("failed forward")

    monkeypatch.setattr(engine.model, "forward", fail)
    with pytest.raises(RuntimeError, match="failed forward"):
        extract_text_addresses(engine, "Next?", ["run"])
    for index, layer in engine.controller.layers.items():
        assert layer.state is old_states[index]
        assert "_fast_qkv" not in vars(layer)
    assert not engine.controller._active


def test_search_zero_reset_immutability_and_candidate_equivariance():
    engine = make_engine()
    addresses = extract_text_addresses(engine, "Next?", ["run", "sit", "eat"])
    current, reset = snapshot_states(engine), snapshot_states(engine, reset=True)
    before = {index: state.to_tensor_dict() for index, state in current.items()}
    options = dict(radius=0.2, steps=3, step_size=1.0, regularization=0.01)
    scored = score_addresses(current, reset, addresses, **options)
    empty = score_addresses(reset, reset, addresses, **options)
    assert empty["max_displacement"] == 0
    for name in ("direct_scores", "search_scores", "direct_objectives", "search_objectives"):
        assert empty[name] == [0.0, 0.0, 0.0]
    assert scored["max_displacement"] <= 0.2 + 1e-6
    assert all(a >= b - 1e-7 for a, b in zip(scored["search_objectives"], scored["direct_objectives"]))
    reordered = TextAddresses(addresses.queries,
                              {index: targets[[2, 0, 1]] for index, targets in addresses.targets.items()},
                              addresses.query_tokens, tuple(addresses.candidate_tokens[i] for i in [2, 0, 1]),
                              addresses.position_offset)
    reordered_scores = score_addresses(current, reset, reordered, **options)
    assert reordered_scores["search_scores"] == pytest.approx(
        [scored["search_scores"][i] for i in [2, 0, 1]], rel=2e-5, abs=1e-7)
    for index, state in current.items():
        for name, value in state.to_tensor_dict().items():
            assert torch.equal(value, before[index][name]), name


def test_last_visual_capture_ignores_later_nonvisual_tokens():
    engine = make_engine()
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad(), NativeQKVCapture(engine, selection="last_visual") as capture:
        with engine.controller.context("write", video_mask=mask):
            engine.model(input_ids=torch.tensor([[4, 5, 6, 7]]), use_cache=False)
    assert set(capture.records) == set(engine.controller.layers)
    for record in capture.records.values():
        assert record["k"].shape == (4, 1, 8)
    addresses = extract_text_addresses(engine, "Next?", ["run"])
    current, reset = snapshot_states(engine), snapshot_states(engine, reset=True)
    result = visual_key_diagnostic(current, reset, addresses, capture.records)
    assert set(result) == {"per_layer", "visual_key", "text_key", "inverse_text_key",
                           "inverse_text_objective", "max_displacement"}
    assert result["inverse_text_objective"] >= result["text_key"] - 1e-7
    empty = visual_key_diagnostic(reset, reset, addresses, capture.records)
    assert empty["visual_key"] == empty["text_key"] == empty["inverse_text_key"] == 0


@pytest.mark.parametrize("tokens", [None, 1, 3])
@pytest.mark.parametrize("step_size", [0.05, 1.0, 1000.0])
def test_batched_candidate_search_matches_independent_reference(tokens, step_size):
    torch.manual_seed(29)
    memory = SwiGLUFastWeightMemory(FastWeightConfig(dim=16, num_heads=2,
                                                    use_muon=False, use_momentum=False))
    reset, current = memory.initial_state(), memory.initial_state()
    with torch.no_grad():
        current.w0.add_(torch.randn_like(current.w0) * 0.1)
        current.w1.add_(torch.randn_like(current.w1) * 0.1)
    shape = (2, 8) if tokens is None else (2, tokens, 8)
    query, targets = torch.randn(shape), torch.randn(6, *shape)
    before = [state.to_tensor_dict() for state in (current, reset)]
    options = dict(radius=0.5, steps=3, step_size=step_size, regularization=0.03)
    actual = batched_inverse_address_candidates(current, reset, query, targets, **options)
    expected = compare_inverse_address_candidates(current, reset, query, targets, **options)
    for name in ("direct_scores", "search_scores", "direct_objectives", "search_objectives",
                 "queries", "displacement_norms"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=2e-5, atol=3e-6)
    assert actual.displacement_norms.max() <= 0.5 + 1e-6
    for state, snapshot in zip((current, reset), before):
        for name, value in state.to_tensor_dict().items():
            assert torch.equal(value, snapshot[name]), name


def test_batched_search_zero_reset_and_order_equivariance():
    torch.manual_seed(17)
    memory = SwiGLUFastWeightMemory(FastWeightConfig(dim=16, num_heads=2))
    reset = memory.initial_state()
    current = memory.initial_state()
    with torch.no_grad():
        current.w1.add_(0.02)
    query, targets = torch.randn(2, 1, 8), torch.randn(6, 2, 1, 8)
    options = dict(radius=1.0, steps=3, step_size=1.0, regularization=0.01)
    empty = batched_inverse_address_candidates(reset, reset, query, targets, **options)
    assert torch.count_nonzero(empty.direct_scores) == 0
    assert torch.count_nonzero(empty.search_objectives) == 0
    assert torch.count_nonzero(empty.displacement_norms) == 0
    normal = batched_inverse_address_candidates(current, reset, query, targets, **options)
    order = [5, 2, 1, 4, 0, 3]
    shuffled = batched_inverse_address_candidates(current, reset, query, targets[order], **options)
    torch.testing.assert_close(shuffled.search_scores, normal.search_scores[order], rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(shuffled.queries, normal.queries[order], rtol=2e-5, atol=3e-6)


def test_batched_search_keeps_initial_query_for_each_candidate_independently():
    one, zero = torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
    current = FastWeightState(w0=one, w1=one, w2=one,
                              w0_norm=one, w1_norm=one, w2_norm=one)
    reset = FastWeightState(w0=zero, w1=one, w2=one,
                            w0_norm=zero, w1_norm=one, w2_norm=one)
    query = torch.tensor([[-1.0]])
    targets = torch.tensor([[[1.0]], [[-0.01]], [[0.0]]])
    options = dict(radius=100.0, steps=1, step_size=100.0)
    result = batched_inverse_address_candidates(current, reset, query, targets, **options)
    reference = compare_inverse_address_candidates(current, reset, query, targets, **options)
    torch.testing.assert_close(result.queries, reference.queries)
    torch.testing.assert_close(result.search_objectives, reference.search_objectives)
    assert torch.equal(result.queries[0], query)
    assert torch.equal(result.queries[2], query)
    assert not torch.equal(result.queries[1], query)


def test_batched_search_rejects_nonfinite_later_candidates():
    memory = SwiGLUFastWeightMemory(FastWeightConfig(dim=16, num_heads=2))
    state = memory.initial_state()
    query, targets = torch.randn(2, 1, 8), torch.randn(6, 2, 1, 8)
    targets[-1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        batched_inverse_address_candidates(state, state, query, targets, radius=0.5)
