"""Spatial-TTT reference layers for the installed Transformers Qwen3-VL.

Adapted from THU-SI/Spatial-TTT, commit e2e33a62 (Apache-2.0).
See THIRD_PARTY_NOTICES.md. This implementation uses PyTorch SDPA, explicit
fast-weight state, and bounded input clips. It does not retain video KV across
calls: only fast weights carry observations into a later parameter-only query.
The official whole-model streaming/cache protocol is therefore not reproduced.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ttt_frame.spatial_memory import FastWeightConfig, FastWeightState, SwiGLUFastWeightMemory


@dataclass
class SpatialModelConfig:
    num_heads: int = 4
    chunk_size: int = 2648
    window_size: int = 2648
    base_lr: float = 1e-3
    use_muon: bool = True
    use_momentum: bool = True
    use_conv: bool = True
    ttt_scale_init: float = 0.0
    seed: int = 0
    layers: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.num_heads < 1 or self.chunk_size < 1 or self.window_size < self.chunk_size:
            raise ValueError("require positive heads/chunk and window_size >= chunk_size")
        if self.base_lr < 0 or not torch.isfinite(torch.tensor(self.base_lr)):
            raise ValueError("base_lr must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(self.ttt_scale_init)):
            raise ValueError("ttt_scale_init must be finite")
        if self.layers is not None:
            self.layers = tuple(self.layers)
            if not self.layers or len(set(self.layers)) != len(self.layers):
                raise ValueError("layers must be nonempty and unique")


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _rope(x, cos, sin):
    return x * cos[:, None] + _rotate_half(x) * sin[:, None]


class SpatialAttention(nn.Module):
    """Original Q/K/V -> parallel SWA and spatial SwiGLU memory -> original O.

    ``state`` is runtime episode data, deliberately outside module state_dict.
    The module parameters describe the learned write/read rule, not an episode.
    """

    def __init__(self, attn_layer, config: SpatialModelConfig, spatial_merge_size: int = 2):
        super().__init__()
        self.attn_layer = attn_layer
        self.config = attn_layer.config
        self.settings = config
        self.layer_idx = attn_layer.layer_idx
        self.head_dim = attn_layer.head_dim
        self.num_q_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.spatial_merge_size = spatial_merge_size
        if self.num_q_heads * self.head_dim != self.hidden_size:
            raise ValueError("reference Spatial-TTT requires Q output width == hidden_size")
        if self.hidden_size % config.num_heads or self.hidden_size // config.num_heads < self.head_dim:
            raise ValueError("fast head width must divide hidden_size and cover the RoPE head width")
        self.memory = SwiGLUFastWeightMemory(FastWeightConfig(
            dim=self.hidden_size, num_heads=config.num_heads, chunk_size=config.chunk_size,
            base_lr=config.base_lr, use_muon=config.use_muon,
            use_momentum=config.use_momentum, ttt_scale_init=config.ttt_scale_init,
            seed=config.seed + self.layer_idx,
        ))
        self.q_scale = nn.Parameter(torch.ones(self.hidden_size))
        self.q_offset = nn.Parameter(torch.zeros(self.hidden_size))
        self.k_scale = nn.Parameter(torch.ones(self.hidden_size))
        self.k_offset = nn.Parameter(torch.zeros(self.hidden_size))
        if config.use_conv:
            for name in ("conv_q", "conv_k", "conv_v"):
                conv = nn.Conv3d(self.hidden_size, self.hidden_size, 3, padding=1,
                                 padding_mode="replicate", groups=self.hidden_size, bias=False)
                nn.init.dirac_(conv.weight, groups=self.hidden_size)
                setattr(self, name, conv)
        # New slow parameters use FP32; preserve the original attention's dtype.
        device = attn_layer.q_proj.weight.device
        self.memory.to(device=device, dtype=torch.float32)
        for name, p in self.named_parameters():
            if not name.startswith("attn_layer."):
                p.data = p.data.to(device=device, dtype=torch.float32)
        self.state: FastWeightState | None = None
        self._mode = "read"
        self._video_mask = None
        self._video_grid = None
        self._flush = True

    @property
    def q_proj(self):
        return self.attn_layer.q_proj

    @property
    def k_proj(self):
        return self.attn_layer.k_proj

    @property
    def v_proj(self):
        return self.attn_layer.v_proj

    def _conv(self, x: torch.Tensor, conv: nn.Conv3d) -> torch.Tensor:
        mask, grid = self._video_mask, self._video_grid
        if mask is None:
            if grid is not None:
                raise ValueError("video_grid_thw requires video_mask")
            return x
        if grid is None:
            raise ValueError("video_mask requires video_grid_thw")
        if x.shape[0] != 1:
            raise ValueError("video convolution currently supports one stream at a time")
        mask = mask.to(device=x.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != x.shape[:2]:
            raise ValueError("video_mask shape must match [batch, sequence]")
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError("video_grid_thw must have shape [videos, 3]")
        shapes = []
        for row in grid.tolist():
            t, h, w = map(int, row)
            m = self.spatial_merge_size
            if min(t, h, w) < 1 or h % m or w % m:
                raise ValueError("invalid video grid for spatial merge size")
            shapes.append((t, h // m, w // m))
        positions = mask[0].nonzero(as_tuple=False).flatten()
        if sum(t*h*w for t, h, w in shapes) != positions.numel():
            raise ValueError("video token count does not match video_grid_thw")
        result, start = x.clone(), 0
        for t, h, w in shapes:
            n = t*h*w
            pos = positions[start:start+n]
            volume = x[0, pos].reshape(t, h, w, -1).permute(3, 0, 1, 2).unsqueeze(0)
            transformed = conv(volume.to(conv.weight.dtype))
            result[0, pos] = transformed.squeeze(0).permute(1, 2, 3, 0).reshape(n, -1).to(x.dtype)
            start += n
        return result

    def _fast_qkv(self, q, k, v, cos, sin):
        batch, length = q.shape[:2]
        q = q.reshape(batch, length, -1).float() * self.q_scale + self.q_offset
        repetitions = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repetitions, dim=2).reshape(batch, length, -1).float()
        v = v.repeat_interleave(repetitions, dim=2).reshape(batch, length, -1).float()
        k = k * self.k_scale + self.k_offset
        if self.settings.use_conv:
            q, k, v = (self._conv(x, conv) for x, conv in (
                (q, self.conv_q), (k, self.conv_k), (v, self.conv_v)))
        heads = self.settings.num_heads
        def prepare(x, normalize):
            x = F.silu(x).reshape(batch, length, heads, -1).transpose(1, 2)
            if normalize:
                # Match the official reference l2_norm epsilon, including partial RoPE.
                x = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
                x = torch.cat((_rope(x[..., :self.head_dim], cos.float(), sin.float()),
                               x[..., self.head_dim:]), dim=-1)
            return x.transpose(1, 2).reshape(batch, length, -1)
        return prepare(q, True), prepare(k, True), prepare(v, False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        if self._mode == "base":
            return self.attn_layer(hidden_states, position_embeddings=position_embeddings,
                                   attention_mask=attention_mask, past_key_values=past_key_values,
                                   cache_position=cache_position, **kwargs)
        if self._mode == "write" and past_key_values is not None:
            raise ValueError("write with use_cache=False; video KV must not become persistent memory")
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, length, self.num_q_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
        q, k = self.attn_layer.q_norm(q), self.attn_layer.k_norm(k)
        cos, sin = position_embeddings
        fast_q, fast_k, fast_v = self._fast_qkv(q, k, v, cos, sin)
        query = _rope(q.transpose(1, 2), cos, sin)
        key, value = _rope(k.transpose(1, 2), cos, sin), v.transpose(1, 2)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx,
                                               {"cos": cos, "sin": sin, "cache_position": cache_position})
        past_length = key.shape[2] - length
        qi = torch.arange(length, device=hidden_states.device)[:, None] + past_length
        ki = torch.arange(key.shape[2], device=hidden_states.device)[None, :]
        allowed = (ki <= qi) & (ki > qi - self.settings.window_size)
        mask = torch.zeros_like(allowed, dtype=query.dtype).masked_fill(~allowed, float("-inf"))
        mask = mask[None, None]
        if attention_mask is not None:
            if attention_mask.ndim != 4:
                raise ValueError("expected Transformers' four-dimensional causal attention mask")
            given = attention_mask[..., -length:, :key.shape[2]]
            mask = mask.masked_fill(~given, float("-inf")) if given.dtype == torch.bool else mask + given
        repetition = self.num_q_heads // self.num_kv_heads
        local = F.scaled_dot_product_attention(
            query, key.repeat_interleave(repetition, dim=1), value.repeat_interleave(repetition, dim=1),
            attn_mask=mask, dropout_p=0.0, scale=self.attn_layer.scaling,
        ).transpose(1, 2).reshape(batch, length, -1)
        memory_out, next_state = self.memory(
            fast_q, fast_k, fast_v, hidden_states, self.state,
            write=self._mode == "write", flush=self._flush if self._mode == "write" else False,
        )
        if self._mode == "write":
            self.state = next_state
        return self.attn_layer.o_proj(local + memory_out.to(local.dtype)), None


class SpatialQwenMemory:
    """Controller for explicit write/read/base contexts and episode snapshots.

    New slow parameters remain differentiable for offline training. Runtime
    ingestion should use torch.no_grad(); snapshots contain only episode state.
    Do not use gradient checkpointing with persistent write contexts, because
    backward recomputation would write the same observations twice.
    """

    def __init__(self, model, config: SpatialModelConfig):
        if getattr(model.config, "model_type", None) != "qwen3_vl":
            raise ValueError("SpatialQwenMemory currently supports Qwen3-VL only")
        self.model, self.config = model, config
        decoder = model.model.language_model.layers
        indices = config.layers if config.layers is not None else tuple(i for i in range(len(decoder)) if i % 4 != 3)
        if any(i < 0 or i >= len(decoder) for i in indices):
            raise ValueError("TTT layer index out of range")
        if any(isinstance(layer.self_attn, SpatialAttention) for layer in decoder):
            raise ValueError("model already contains Spatial-TTT layers")
        model.requires_grad_(False)
        self.layers = {}
        with torch.random.fork_rng(devices=[]):
            # New modules initialize on CPU before transfer; preserve CUDA RNG.
            torch.random.default_generator.manual_seed(config.seed)
            for idx in indices:
                layer = SpatialAttention(decoder[idx].self_attn, config,
                                         model.config.vision_config.spatial_merge_size)
                decoder[idx].self_attn = layer
                self.layers[idx] = layer
        self._active = False

    @contextmanager
    def context(self, mode="read", *, video_mask=None, video_grid_thw=None, flush=True):
        if mode not in {"write", "read", "base"}:
            raise ValueError("mode must be write, read, or base")
        if self._active:
            raise RuntimeError("memory contexts cannot overlap")
        if mode == "write" and getattr(self.model, "is_gradient_checkpointing", False):
            raise ValueError("disable gradient checkpointing before stateful write forwards")
        self._active = True
        for layer in self.layers.values():
            layer._mode, layer._video_mask, layer._video_grid, layer._flush = mode, video_mask, video_grid_thw, flush
        try:
            yield self
        finally:
            for layer in self.layers.values():
                layer._mode, layer._video_mask, layer._video_grid = "read", None, None
            self._active = False

    def reset(self):
        if self._active:
            raise RuntimeError("cannot reset inside a model forward context")
        for layer in self.layers.values():
            layer.state = None

    def detach(self):
        for layer in self.layers.values():
            if layer.state is not None:
                layer.state = layer.state.detached()

    def flush(self):
        for layer in self.layers.values():
            if layer.state is not None:
                layer.state = layer.memory.flush(layer.state)

    @property
    def memory_bytes(self):
        return sum(w.numel() * w.element_size() for layer in self.layers.values()
                   if layer.state is not None for w in (layer.state.w0, layer.state.w1, layer.state.w2))

    @property
    def runtime_state_bytes(self):
        return sum(16 + sum(t.numel() * t.element_size() for t in vars(layer.state).values()
                            if torch.is_tensor(t))
                   for layer in self.layers.values() if layer.state is not None)

    def memory_state_dict(self):
        return {f"layers.{idx}.{key}": value
                for idx, layer in self.layers.items() if layer.state is not None
                for key, value in layer.state.to_tensor_dict().items()}

    def _decode_memory_state_dict(self, tensors):
        if not tensors:
            return {idx: None for idx in self.layers}
        groups = {idx: {} for idx in self.layers}
        for name, tensor in tensors.items():
            parts = name.split(".", 2)
            if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit() or int(parts[1]) not in groups:
                raise ValueError(f"unexpected memory key: {name}")
            groups[int(parts[1])][parts[2]] = tensor
        restored = {}
        for idx, layer in self.layers.items():
            if not groups[idx]:
                raise ValueError("memory must include every TTT layer")
            device = layer.q_proj.weight.device
            state = FastWeightState.from_tensor_dict({k: v.to(device) for k, v in groups[idx].items()})
            state.validate(layer.memory.config, batch_size=1)
            restored[idx] = state
        return restored

    def validate_memory_state_dict(self, tensors):
        self._decode_memory_state_dict(tensors)

    def load_memory_state_dict(self, tensors):
        if self._active:
            raise RuntimeError("cannot load inside a model forward context")
        restored = self._decode_memory_state_dict(tensors)
        for idx, state in restored.items():
            self.layers[idx].state = state

    def slow_state_dict(self):
        """Added write/read-rule parameters, excluding the original base model."""
        return {f"layers.{idx}.{name}": value.detach().cpu().contiguous()
                for idx, layer in self.layers.items() for name, value in layer.state_dict().items()
                if not name.startswith("attn_layer.")}

    def load_slow_state_dict(self, tensors):
        expected = self.slow_state_dict()
        if tensors.keys() != expected.keys() or any(
            tensors[k].shape != expected[k].shape or not torch.isfinite(tensors[k]).all() for k in expected
        ):
            raise ValueError("invalid spatial slow-parameter checkpoint")
        with torch.no_grad():
            for idx, layer in self.layers.items():
                for name, parameter in layer.named_parameters():
                    if not name.startswith("attn_layer."):
                        parameter.copy_(tensors[f"layers.{idx}.{name}"])
        self.reset()


def load_official_checkpoint(controller: SpatialQwenMemory, path: str | Path):
    """Load a full official dense Spatial-TTT checkpoint, including tuned LLM.

    Only the single safetensors format is accepted. The public nano repository's
    stale shard index is intentionally bypassed. No download occurs here.
    """
    from safetensors import safe_open

    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    expected = controller.model.state_dict()
    mapping = {name: name.replace(".self_attn.memory.", ".self_attn.") for name in expected}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        available = set(source.keys())
        # Safetensors may omit one of tied input/output embedding tensors.
        for local, remote in list(mapping.items()):
            if remote not in available and controller.model.config.tie_word_embeddings:
                alternate = {"lm_head.weight": "model.language_model.embed_tokens.weight",
                             "model.language_model.embed_tokens.weight": "lm_head.weight"}.get(remote)
                if alternate in available:
                    mapping[local] = alternate
        missing = set(mapping.values()) - available
        extra = available - set(mapping.values())
        if missing or extra:
            raise ValueError(f"official checkpoint key mismatch; missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}")
        for local, remote in mapping.items():
            if list(expected[local].shape) != source.get_slice(remote).get_shape():
                raise ValueError(f"official checkpoint shape mismatch: {remote}")
        for remote in set(mapping.values()):
            if not torch.isfinite(source.get_tensor(remote)).all():
                raise ValueError(f"nonfinite official checkpoint tensor: {remote}")
        with torch.no_grad():
            for local, remote in mapping.items():
                tensor = source.get_tensor(remote)
                expected[local].copy_(tensor)
    controller.reset()


__all__ = ["SpatialModelConfig", "SpatialAttention", "SpatialQwenMemory", "load_official_checkpoint"]
