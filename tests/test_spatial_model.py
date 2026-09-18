"""Offline integration checks for parameter memory; random weights do not test QA quality."""

from copy import deepcopy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from ttt_frame.spatial_model import SpatialModelConfig, SpatialQwenMemory


@pytest.mark.parametrize("corrupt", [False, True])
def test_official_full_checkpoint_loads_tuned_base_and_validates_before_mutation(tmp_path, corrupt):
    from safetensors.torch import save_file
    from ttt_frame.spatial_model import load_official_checkpoint

    source, source_controller = engine()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.01)
    payload = {name.replace(".self_attn.memory.", ".self_attn."): value.clone()
               for name, value in source.state_dict().items()}
    if corrupt:
        # A late tensor must not leave early base weights partially overwritten.
        payload["model.language_model.layers.3.mlp.down_proj.weight"].view(-1)[0] = float("nan")
    path = tmp_path / "model.safetensors"
    save_file(payload, str(path))
    target, controller = engine()
    forward(target, controller, [5, 6, 7, 8], "write")
    before = {name: value.clone() for name, value in target.state_dict().items()}
    state = snapshot(controller)
    if corrupt:
        with pytest.raises(ValueError, match="nonfinite"):
            load_official_checkpoint(controller, path)
        assert_same_state(before, target.state_dict())
        assert_same_state(state, snapshot(controller))
    else:
        load_official_checkpoint(controller, path)
        assert_same_state(source.state_dict(), target.state_dict())
        assert not controller.memory_state_dict()


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_qwen():
    """Exercise the real Qwen attention/vision interfaces without downloaded weights."""
    torch.manual_seed(71)
    config = transformers.Qwen3VLConfig(
        text_config=dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=512,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
            pad_token_id=0,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            depth=2,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=64,
            deepstack_visual_indexes=[0],
        ),
        image_token_id=125,
        video_token_id=126,
        vision_start_token_id=123,
        vision_end_token_id=124,
    )
    return transformers.Qwen3VLForConditionalGeneration(config).eval()


def settings(**overrides):
    options = dict(
        num_heads=4,
        chunk_size=4,
        window_size=32,
        base_lr=0.1,
        use_muon=False,
        use_momentum=True,
        use_conv=False,
        ttt_scale_init=0.2,
        seed=3,
    )
    options.update(overrides)
    return SpatialModelConfig(**options)


def engine(**overrides):
    model = tiny_qwen()
    return model, SpatialQwenMemory(model, settings(**overrides))


def forward(model, controller, tokens, mode, **context):
    with torch.no_grad(), controller.context(mode=mode, **context):
        return model(input_ids=torch.tensor([tokens]), use_cache=False).logits.detach().clone()


def snapshot(controller):
    return {name: value.detach().clone() for name, value in controller.memory_state_dict().items()}


def assert_same_state(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def different_state(left, right):
    return left.keys() != right.keys() or any(
        not torch.equal(left[name], right[name]) for name in left
    )


def test_default_three_to_one_layout_reuses_base_projections():
    model = tiny_qwen()
    original = [layer.self_attn for layer in model.model.language_model.layers]
    controller = SpatialQwenMemory(model, settings())
    assert set(controller.layers) == {0, 1, 2}
    assert model.model.language_model.layers[3].self_attn is original[3]
    for index, layer in controller.layers.items():
        assert model.model.language_model.layers[index].self_attn is layer
        for projection in ("q_proj", "k_proj", "v_proj"):
            assert getattr(layer, projection) is getattr(original[index], projection)


def test_zero_scale_full_window_matches_unmodified_qwen():
    model = tiny_qwen()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    with torch.no_grad():
        original = model(input_ids=torch.tensor([tokens]), use_cache=False).logits
    controller = SpatialQwenMemory(model, settings(ttt_scale_init=0.0))
    base = forward(model, controller, tokens, "base")
    written = forward(model, controller, tokens, "write")
    read = forward(model, controller, tokens, "read")
    torch.testing.assert_close(base, original, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(written, original, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(read, original, rtol=1e-5, atol=1e-6)


def test_write_changes_only_runtime_fastweights_and_has_fixed_footprint():
    model = tiny_qwen()
    original_parameters = [(parameter, parameter.detach().clone()) for parameter in model.parameters()]
    controller = SpatialQwenMemory(model, settings())
    registered_parameters = [(parameter, parameter.detach().clone()) for parameter in model.parameters()]
    before = snapshot(controller)
    forward(model, controller, [1, 2, 3, 4, 5, 6, 7, 8], "write")
    after = snapshot(controller)
    assert after and different_state(before, after)
    footprint = controller.memory_bytes
    assert footprint > 0
    for parameter, value in original_parameters + registered_parameters:
        assert torch.equal(parameter, value)
        assert parameter.grad is None
    forward(model, controller, list(range(10, 30)), "write")
    assert different_state(after, snapshot(controller))
    assert controller.memory_bytes == footprint


def test_queries_do_not_write_and_another_observation_can_follow():
    model, controller = engine()
    forward(model, controller, [1, 2, 3, 4, 5, 6], "write")
    after_first_observation = snapshot(controller)
    first_answer = forward(model, controller, [31, 32, 33], "read")
    forward(model, controller, [41, 42], "read")
    second_answer = forward(model, controller, [31, 32, 33], "read")
    assert_same_state(snapshot(controller), after_first_observation)
    assert torch.equal(first_answer, second_answer)
    forward(model, controller, [11, 12, 13, 14, 15, 16], "write")
    assert different_state(snapshot(controller), after_first_observation)


def test_different_observations_produce_different_parameter_memory():
    model, controller = engine()
    forward(model, controller, [1, 2, 3, 4, 5, 6], "write")
    first = snapshot(controller)
    controller.reset()
    forward(model, controller, [11, 12, 13, 14, 15, 16], "write")
    assert different_state(first, snapshot(controller))


def test_cached_question_decode_matches_full_read_without_writing_memory():
    model, controller = engine()
    forward(model, controller, [1, 2, 3, 4, 5, 6], "write")
    saved_state = snapshot(controller)
    expected = forward(model, controller, [31, 32, 33, 34], "read")[:, -1]
    with torch.no_grad(), controller.context(mode="read"):
        prefix = model(
            input_ids=torch.tensor([[31, 32, 33]]),
            cache_position=torch.arange(3),
            use_cache=True,
        )
        last = model(
            input_ids=torch.tensor([[34]]),
            past_key_values=prefix.past_key_values,
            cache_position=torch.tensor([3]),
            use_cache=True,
        )
    torch.testing.assert_close(last.logits[:, -1], expected, rtol=1e-5, atol=1e-6)
    assert_same_state(snapshot(controller), saved_state)


def test_saved_memory_restores_logits_and_the_next_update(tmp_path):
    model, controller = engine()
    probe = [31, 32, 33]
    initial_logits = forward(model, controller, probe, "read")
    forward(model, controller, [1, 2, 3, 4, 5, 6], "write")
    saved_state = snapshot(controller)
    remembered_logits = forward(model, controller, probe, "read")
    assert not torch.equal(remembered_logits, initial_logits)
    checkpoint = tmp_path / "fastweights.pt"
    torch.save(saved_state, checkpoint)
    forward(model, controller, [11, 12, 13, 14], "write")
    continued_state = snapshot(controller)

    controller.reset()
    assert torch.equal(forward(model, controller, probe, "read"), initial_logits)
    controller.load_memory_state_dict(torch.load(checkpoint, weights_only=True))
    assert_same_state(snapshot(controller), saved_state)
    assert torch.equal(forward(model, controller, probe, "read"), remembered_logits)
    forward(model, controller, [11, 12, 13, 14], "write")
    assert_same_state(snapshot(controller), continued_state)


def test_nonzero_memory_branch_changes_readout_but_base_is_independent():
    model, controller = engine(ttt_scale_init=1.0)
    probe = [31, 32, 33]
    base_before = forward(model, controller, probe, "base")
    read_before = forward(model, controller, probe, "read")
    forward(model, controller, [1, 2, 3, 4, 5, 6], "write")
    read_after = forward(model, controller, probe, "read")
    assert (read_after - read_before).abs().max().item() > 1e-7
    assert torch.equal(forward(model, controller, probe, "base"), base_before)


def test_real_video_pixels_are_written_without_text_targets():
    model, controller = engine(use_conv=True)
    input_ids = torch.tensor([[123, 126, 126, 126, 126, 124]])
    grid = torch.tensor([[1, 4, 4]])
    video_mask = input_ids == 126
    torch.manual_seed(19)
    first_pixels = torch.randn(16, 3 * 2 * 16 * 16)
    second_pixels = torch.randn_like(first_pixels)

    def write(pixels):
        with torch.no_grad(), controller.context(
            mode="write", video_mask=video_mask, video_grid_thw=grid,
        ):
            model(
                input_ids=input_ids,
                pixel_values_videos=pixels,
                video_grid_thw=grid,
                use_cache=False,
            )

    write(first_pixels)
    first_state = snapshot(controller)
    controller.reset()
    write(second_pixels)
    assert different_state(first_state, snapshot(controller))


def test_spatial_convolution_checks_merged_grid_token_count():
    model, controller = engine(use_conv=True)
    with pytest.raises(ValueError, match="(?i)(grid|token|mask)"):
        forward(
            model, controller, [1, 2, 3, 4, 5, 6], "write",
            video_mask=torch.tensor([[False, True, True, True, True, False]]),
            video_grid_thw=torch.tensor([[1, 2, 2]]),
        )


def test_spatial_convolution_affects_visual_memory_write():
    model_on, controller_on = engine(use_conv=True)
    model_off, controller_off = engine(use_conv=False)
    # Identical shared parameters isolate the effect of the convolution path.
    source = model_on.state_dict()
    target = model_off.state_dict()
    shared = {key: value for key, value in source.items() if key in target}
    model_off.load_state_dict(deepcopy(shared), strict=False)
    controller_on.reset()
    controller_off.reset()
    context = dict(
        video_mask=torch.tensor([[False, True, True, True, True, False]]),
        video_grid_thw=torch.tensor([[1, 4, 4]]),
    )
    forward(model_on, controller_on, [1, 2, 3, 4, 5, 6], "write", **context)
    forward(model_off, controller_off, [1, 2, 3, 4, 5, 6], "write", **context)
    without_neighbors = snapshot(controller_off)
    assert_same_state(snapshot(controller_on), without_neighbors)

    # Bootstrap convolution is an identity; emulate learned neighboring-token mixing.
    controller_on.reset()
    with torch.no_grad():
        for layer in controller_on.layers.values():
            for name in ("conv_q", "conv_k", "conv_v"):
                getattr(layer, name).weight.fill_(1 / 27)
    forward(model_on, controller_on, [1, 2, 3, 4, 5, 6], "write", **context)
    assert different_state(snapshot(controller_on), without_neighbors)
