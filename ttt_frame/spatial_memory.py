"""Differentiable, streaming SwiGLU fast weights used by Spatial-TTT.

Reference: THU-SI/Spatial-TTT, commit e2e33a62, ``ttt_operation.py`` and
``causal_swa_lact.py``. This is the post-normalized, full-rank formulation:
each chunk reads the previous weights, then ascends the associative objective
``sum(f_W(K) * V)``. It is *not* an MSE reconstruction update. Q/K/V preparation
(spatial convolution, SiLU, normalization and RoPE) belongs to the model wrapper.

The recurrent state is separate from trainable initial weights. Nothing is
detached implicitly, so an outer training loss can differentiate through writes.
Inference callers should use ``torch.no_grad()`` and detach stored state. A
partial chunk is bounded by ``chunk_size`` and may be explicitly flushed before
revoking input. Reads use only ``w0/w1/w2`` after flushing; momentum also carries
input-dependent history and is retained only to support future writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FastWeightConfig:
    dim: int
    num_heads: int = 4
    inter_multi: float = 1.0
    chunk_size: int = 2648
    base_lr: float = 1e-3
    use_muon: bool = True
    use_momentum: bool = True
    ttt_scale_init: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("dim", "num_heads", "chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if not math.isfinite(self.inter_multi) or self.inter_multi <= 0:
            raise ValueError("inter_multi must be finite and positive")
        if int(self.dim // self.num_heads * self.inter_multi) < 1:
            raise ValueError("inter_multi produces an empty intermediate dimension")
        if not math.isfinite(self.base_lr) or self.base_lr < 0:
            raise ValueError("base_lr must be finite and nonnegative")
        if not math.isfinite(self.ttt_scale_init):
            raise ValueError("ttt_scale_init must be finite")


@dataclass
class FastWeightState:
    """FP32 runtime tensors; the leading dimension combines batch and heads.

    ``tokens`` counts ingested tokens and ``updates`` counts committed chunks
    (including a chunk committed with a disabled learning rate). Norm targets
    remain those of the initial weights throughout a memory trajectory.
    """

    w0: Tensor
    w1: Tensor
    w2: Tensor
    w0_norm: Tensor
    w1_norm: Tensor
    w2_norm: Tensor
    dw0_momentum: Tensor | None = None
    dw1_momentum: Tensor | None = None
    dw2_momentum: Tensor | None = None
    pending_k: Tensor | None = None
    pending_v: Tensor | None = None
    pending_lr0: Tensor | None = None
    pending_lr1: Tensor | None = None
    pending_lr2: Tensor | None = None
    pending_momentum: Tensor | None = None
    updates: int = 0
    tokens: int = 0

    @property
    def pending_tokens(self) -> int:
        return 0 if self.pending_k is None else self.pending_k.shape[1]

    def clone(self) -> FastWeightState:
        """Copy tensor storage without breaking the offline training graph."""
        return replace(self, **{
            f.name: value.clone()
            for f in fields(self)
            if isinstance(value := getattr(self, f.name), Tensor)
        })

    def detached(self) -> FastWeightState:
        """Detach the graph; storage is shared (writes here are functional)."""
        return replace(self, **{
            f.name: value.detach()
            for f in fields(self)
            if isinstance(value := getattr(self, f.name), Tensor)
        })

    def to_tensor_dict(self) -> dict[str, Tensor]:
        """Return detached, independent tensors suitable for safetensors."""
        result = {
            f.name: value.detach().contiguous().clone()
            for f in fields(self)
            if isinstance(value := getattr(self, f.name), Tensor)
        }
        result["updates"] = torch.tensor(self.updates, dtype=torch.int64)
        result["tokens"] = torch.tensor(self.tokens, dtype=torch.int64)
        return result

    @classmethod
    def from_tensor_dict(cls, tensors: Mapping[str, Tensor]) -> FastWeightState:
        """Load tensor-only state and reject malformed/nonfinite payloads."""
        allowed = {f.name for f in fields(cls)}
        required = {"w0", "w1", "w2", "w0_norm", "w1_norm", "w2_norm", "updates", "tokens"}
        if set(tensors) - allowed or required - set(tensors):
            raise ValueError("Invalid fast-weight state fields")
        if any(not isinstance(tensor, Tensor) for tensor in tensors.values()):
            raise ValueError("Fast-weight state values must all be tensors")
        values: dict = dict(tensors)
        for name in ("updates", "tokens"):
            value = values[name]
            if value.dtype != torch.int64 or value.ndim != 0:
                raise ValueError(f"{name} must be an int64 scalar tensor")
            values[name] = int(value.item())
        state = cls(**values)
        state.validate()
        return state

    def validate(
        self, config: FastWeightConfig | None = None, batch_size: int | None = None
    ) -> None:
        """Validate serialized state; not run on every streaming model call."""
        for name in ("updates", "tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.w0.ndim != 3 or any(size < 1 for size in self.w0.shape):
            raise ValueError("w0 must have shape [batch * heads, intermediate, head_dim]")
        bh, intermediate, head_dim = self.w0.shape
        expected = {
            "w0": (bh, intermediate, head_dim),
            "w1": (bh, head_dim, intermediate),
            "w2": (bh, intermediate, head_dim),
            "w0_norm": (bh, intermediate, 1),
            "w1_norm": (bh, head_dim, 1),
            "w2_norm": (bh, intermediate, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Invalid shape for {name}; expected {shape}")
            if name.endswith("_norm") and bool((value < 0).any()):
                raise ValueError(f"{name} must be nonnegative")
        momentum_names = ("dw0_momentum", "dw1_momentum", "dw2_momentum")
        has_momentum = [getattr(self, name) is not None for name in momentum_names]
        if any(has_momentum) != all(has_momentum):
            raise ValueError("All three momentum tensors must be present together")
        if all(has_momentum):
            for name, weight in zip(momentum_names, (self.w0, self.w1, self.w2)):
                if getattr(self, name).shape != weight.shape:
                    raise ValueError(f"Invalid shape for {name}")
        pending_names = ("pending_k", "pending_v", "pending_lr0", "pending_lr1", "pending_lr2")
        has_pending = [getattr(self, name) is not None for name in pending_names]
        if any(has_pending) != all(has_pending):
            raise ValueError("Pending keys, values and all three rates must be present together")
        if all(has_pending):
            if self.pending_k.ndim != 3:
                raise ValueError("pending_k must be three-dimensional")
            length = self.pending_tokens
            if length < 1 or length > self.tokens:
                raise ValueError("Invalid pending token count")
            for name in pending_names + ("pending_momentum",):
                value = getattr(self, name)
                if value is None:
                    continue
                width = head_dim if name in ("pending_k", "pending_v") else 1
                if value.shape != (bh, length, width):
                    raise ValueError(f"Invalid shape for {name}")
            if (self.pending_momentum is not None) != all(has_momentum):
                raise ValueError("Pending momentum must match optimizer momentum")
        elif self.pending_momentum is not None:
            raise ValueError("pending_momentum requires pending tokens")
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None or isinstance(value, int):
                continue
            if not isinstance(value, Tensor) or value.dtype != torch.float32:
                raise ValueError(f"{f.name} must be a float32 tensor")
            if value.device != self.w0.device:
                raise ValueError("All runtime tensors must be on the same device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{f.name} contains nonfinite values")
            if f.name.startswith("pending_lr") and bool((value < 0).any()):
                raise ValueError("Pending learning rates must be nonnegative")
        if config is not None:
            d = config.dim // config.num_heads
            if head_dim != d or intermediate != int(d * config.inter_multi):
                raise ValueError("State dimensions do not match fast-weight config")
            if bh % config.num_heads or (batch_size is not None and bh != batch_size * config.num_heads):
                raise ValueError("State batch/head count does not match config")
            if all(has_momentum) != config.use_momentum:
                raise ValueError("State optimizer momentum does not match config")
            if self.pending_tokens >= config.chunk_size:
                raise ValueError("Pending state must be shorter than chunk_size")


def swiglu_read(w0: Tensor, w1: Tensor, w2: Tensor, q: Tensor) -> Tensor:
    """Evaluate f_W(Q); Q is [batch * heads, tokens, head_dim]."""
    qt = q.transpose(1, 2)
    return torch.bmm(w1, F.silu(torch.bmm(w0, qt)) * torch.bmm(w2, qt)).transpose(1, 2)


def swiglu_fast_weight_gradients(
    w0: Tensor, w1: Tensor, w2: Tensor, k: Tensor, v: Tensor,
    lr0: Tensor, lr1: Tensor, lr2: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Hand-derived ascent updates for sum(f_W(K) * V), with per-token rates.

    Unlike ``autograd.grad``, these operations also work inside ``no_grad``;
    they remain differentiable when outer-loop training enables gradients.
    """
    gate_pre = torch.bmm(w0, k.transpose(1, 2))
    hidden_pre = torch.bmm(w2, k.transpose(1, 2))
    gate = F.silu(gate_pre)
    hidden = gate * hidden_pre
    value_t = v.transpose(1, 2)
    dhidden = torch.bmm(w1.transpose(1, 2), value_t)
    sigma = torch.sigmoid(gate_pre)
    dgate = dhidden * hidden_pre * sigma * (1 + gate_pre * (1 - sigma))
    dw0 = torch.bmm(dgate, k * lr0)
    dw1 = torch.bmm(value_t, hidden.transpose(1, 2) * lr1)
    dw2 = torch.bmm(dhidden * gate, k * lr2)
    return dw0, dw1, dw2


def zeropower_via_newtonschulz5(gradient: Tensor) -> Tensor:
    """Official five-step Muon polynomial, including BF16 intermediates.

    The returned tensor is FP32 for accumulation into runtime weights. Muon
    discards most of a gradient's overall magnitude; ``base_lr`` therefore is
    not an ordinary step-size knob when Muon is enabled.
    """
    x = gradient.to(torch.bfloat16)
    transpose = gradient.shape[1] > gradient.shape[2]
    if transpose:
        x = x.transpose(1, 2)
    x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ):
        gram = x @ x.transpose(1, 2)
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.transpose(1, 2) if transpose else x).float()


class SwiGLUFastWeightMemory(nn.Module):
    """Spatial-TTT's trainable initialization, write controls and read gate.

    The default zero gate preserves the pretrained model's initial output but
    must be trained before memory can affect answers. Setting a nonzero gate
    alone does not make newly initialized memory semantically useful.
    """

    def __init__(self, config: FastWeightConfig):
        super().__init__()
        self.config = config
        self.num_fw_heads = config.num_heads
        self.fw_head_dim = config.dim // config.num_heads
        self.inter_dim = int(self.fw_head_dim * config.inter_multi)
        h, d, i = self.num_fw_heads, self.fw_head_dim, self.inter_dim
        # Reproducible initialization without changing the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.w0 = nn.Parameter(torch.randn(h, i, d) / math.sqrt(d))
            self.w2 = nn.Parameter(torch.randn(h, i, d) / math.sqrt(d))
            self.w1 = nn.Parameter(torch.randn(h, d, i) / math.sqrt(i))
            self.lr_proj = nn.Linear(config.dim, 3 * h)
            self.ttt_scale_proj = nn.Linear(config.dim, h)
            nn.init.zeros_(self.ttt_scale_proj.weight)
            nn.init.constant_(self.ttt_scale_proj.bias, config.ttt_scale_init)
            self.ttt_norm = nn.RMSNorm(d, eps=1e-5)
            self.momentum_proj = (
                nn.Sequential(nn.Linear(config.dim, h), nn.Sigmoid())
                if config.use_momentum else None
            )
        self.base_lr_inv = (
            config.base_lr + math.log(-math.expm1(-config.base_lr))
            if config.base_lr > 0 else -math.inf
        )

    def initial_state(self, batch_size: int = 1) -> FastWeightState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        w0, w1, w2 = (w.float().repeat(batch_size, 1, 1) for w in (self.w0, self.w1, self.w2))
        return FastWeightState(
            w0=w0, w1=w1, w2=w2,
            w0_norm=w0.norm(dim=2, keepdim=True),
            w1_norm=w1.norm(dim=2, keepdim=True),
            w2_norm=w2.norm(dim=2, keepdim=True),
            dw0_momentum=torch.zeros_like(w0) if self.config.use_momentum else None,
            dw1_momentum=torch.zeros_like(w1) if self.config.use_momentum else None,
            dw2_momentum=torch.zeros_like(w2) if self.config.use_momentum else None,
        )

    def _heads(self, tensor: Tensor, width: int) -> Tensor:
        b, t, _ = tensor.shape
        return tensor.reshape(b, t, self.num_fw_heads, width).transpose(1, 2).reshape(
            b * self.num_fw_heads, t, width
        ).float()

    @staticmethod
    def _linear_fp32(layer: nn.Linear, hidden: Tensor) -> Tensor:
        return F.linear(hidden.float(), layer.weight.float(), layer.bias.float())

    def _write_chunk(self, state: FastWeightState) -> FastWeightState:
        if state.pending_k is None:
            return state
        if self.config.base_lr == 0:
            # Avoid both the epsilon norm drift and residual momentum at lr=0.
            return replace(state, updates=state.updates + 1)
        rates = (state.pending_lr0, state.pending_lr1, state.pending_lr2)
        updates = swiglu_fast_weight_gradients(
            state.w0, state.w1, state.w2, state.pending_k, state.pending_v, *rates
        )
        replacements = {}
        for idx, (gradient, rate) in enumerate(zip(updates, rates)):
            weight = getattr(state, f"w{idx}")
            active = (rate != 0).any(dim=1, keepdim=True)
            if self.config.use_momentum:
                old_momentum = getattr(state, f"dw{idx}_momentum")
                coefficient = state.pending_momentum.mean(dim=1, keepdim=True)
                gradient = gradient + old_momentum * coefficient
                replacements[f"dw{idx}_momentum"] = torch.where(active, gradient, old_momentum)
            if self.config.use_muon:
                gradient = zeropower_via_newtonschulz5(gradient)
            updated = weight + gradient
            updated = updated / (updated.norm(dim=2, keepdim=True) + 1e-5)
            updated = updated * getattr(state, f"w{idx}_norm")
            replacements[f"w{idx}"] = torch.where(active, updated, weight)
        return replace(state, **replacements, updates=state.updates + 1)

    @staticmethod
    def _clear_pending(state: FastWeightState) -> FastWeightState:
        return replace(
            state, pending_k=None, pending_v=None, pending_lr0=None,
            pending_lr1=None, pending_lr2=None, pending_momentum=None,
        )

    def flush(self, state: FastWeightState) -> FastWeightState:
        """Commit a tail chunk without reading; never mutate the supplied state."""
        if state.pending_tokens == 0:
            return state
        return self._clear_pending(self._write_chunk(state))

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, hidden_states: Tensor,
        state: FastWeightState | None = None, *, write: bool = True, flush: bool = False,
    ) -> tuple[Tensor, FastWeightState]:
        """Read queries and optionally write K/V in globally aligned chunks.

        Inputs are preprocessed [B,T,dim] tensors; outputs have the same shape
        and dtype as hidden_states. With ``flush=False``, arbitrary call splits
        are equivalent to a single call. ``write=False`` returns the same state
        object and ignores ``flush``: neither questions nor generated answers
        may change stored observations.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.dim:
            raise ValueError("hidden_states must have shape [batch, tokens, dim]")
        if any(x.shape != hidden_states.shape for x in (q, k, v)):
            raise ValueError("q, k, v and hidden_states must have identical shapes")
        if not all(x.is_floating_point() for x in (q, k, v, hidden_states)):
            raise ValueError("Inputs must be floating-point tensors")
        if any(x.device != self.w0.device for x in (q, k, v, hidden_states)):
            raise ValueError("Inputs and module must be on the same device")
        batch, length, _ = hidden_states.shape
        current = self.initial_state(batch) if state is None else state
        expected_w0 = (batch * self.num_fw_heads, self.inter_dim, self.fw_head_dim)
        if current.w0.shape != expected_w0 or current.w0.device != q.device:
            raise ValueError("State batch/head shape or device does not match inputs")
        if current.pending_tokens >= self.config.chunk_size:
            raise ValueError("Invalid state: pending chunk must be shorter than chunk_size")
        fast_q = self._heads(q, self.fw_head_dim)
        if not write:
            raw = swiglu_read(current.w0, current.w1, current.w2, fast_q)
        else:
            fast_k, fast_v = (self._heads(x, self.fw_head_dim) for x in (k, v))
            lr = self._linear_fp32(self.lr_proj, hidden_states)
            lr = F.softplus(lr + self.base_lr_inv) if self.config.base_lr > 0 else torch.zeros_like(lr)
            lr0, lr1, lr2 = self._heads(lr, 3).chunk(3, dim=-1)
            momentum = None
            if self.momentum_proj is not None:
                momentum = self._heads(
                    torch.sigmoid(self._linear_fp32(self.momentum_proj[0], hidden_states)), 1
                )
            outputs = []
            offset = 0
            while offset < length:
                end = min(length, offset + self.config.chunk_size - current.pending_tokens)
                outputs.append(swiglu_read(
                    current.w0, current.w1, current.w2, fast_q[:, offset:end]
                ))
                additions = {
                    "pending_k": fast_k[:, offset:end],
                    "pending_v": fast_v[:, offset:end],
                    "pending_lr0": lr0[:, offset:end],
                    "pending_lr1": lr1[:, offset:end],
                    "pending_lr2": lr2[:, offset:end],
                    "pending_momentum": None if momentum is None else momentum[:, offset:end],
                }
                for name, value in additions.items():
                    previous = getattr(current, name)
                    if value is not None:
                        # Own the bounded tail storage, not the entire input tensor.
                        additions[name] = value.clone() if previous is None else torch.cat((previous, value), dim=1)
                current = replace(current, **additions, tokens=current.tokens + end - offset)
                if current.pending_tokens == self.config.chunk_size:
                    current = self.flush(current)
                offset = end
            raw = torch.cat(outputs, dim=1) if outputs else fast_q
            if flush:
                current = self.flush(current)
        normalized = F.rms_norm(raw, (self.fw_head_dim,), self.ttt_norm.weight.float(), 1e-5)
        scale = self._heads(F.silu(self._linear_fp32(self.ttt_scale_proj, hidden_states)), 1)
        output = (normalized * scale).reshape(batch, self.num_fw_heads, length, self.fw_head_dim)
        output = output.transpose(1, 2).reshape(batch, length, self.config.dim)
        return output.to(hidden_states.dtype), current
