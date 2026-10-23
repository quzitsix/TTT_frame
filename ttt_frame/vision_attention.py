"""Optional, reversible batching of independent Qwen3-VL vision attention groups.

Transformers 4.57's vision SDPA path launches attention once per image or video
tubelet. Equal-sized groups can instead occupy the batch axis of one SDPA call.
No group attends to another, and neither weights nor the Spatial-TTT algorithm
change. Unequal sizes, training, and non-SDPA configurations use the original
forward. Enable explicitly and record this execution option in run provenance.
"""

from __future__ import annotations

from types import MethodType

import torch


class EqualLengthVisionSDPA:
    """Install an instance-local patch; ``remove()`` or context exit restores it.

    ``model`` may be a complete Qwen3-VL or its visual tower. Construction
    installs immediately. No Transformers class or installed package is edited.
    """

    protocol_name = "qwen3_vl_equal_length_vision_sdpa_v1"

    def __init__(self, model: torch.nn.Module):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
        )

        modules = [module for module in model.modules()
                   if isinstance(module, Qwen3VLVisionAttention)]
        if not modules:
            raise ValueError("no Qwen3-VL vision attention modules found")
        if any(hasattr(module, "_ttt_equal_length_sdpa") for module in modules):
            raise ValueError("vision attention optimization is already installed")
        self.optimized_calls = 0
        self.fallback_calls = 0
        self.module_count = len(modules)
        self._originals = []
        for module in modules:
            original = module.forward
            had_instance_forward = "forward" in vars(module)
            self._originals.append((module, original, had_instance_forward))
            module._ttt_equal_length_sdpa = self.protocol_name
            module.forward = MethodType(self._forward_for(original), module)

    def _forward_for(self, original):
        def forward(module, hidden_states, cu_seqlens, rotary_pos_emb=None,
                    position_embeddings=None, **kwargs):
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb_vision,
            )

            def fallback():
                self.fallback_calls += 1
                return original(hidden_states, cu_seqlens, rotary_pos_emb,
                                position_embeddings, **kwargs)

            if (module.training or module.config._attn_implementation != "sdpa"
                    or position_embeddings is None or cu_seqlens.ndim != 1):
                return fallback()
            boundaries = cu_seqlens.detach().cpu().tolist()
            groups = len(boundaries) - 1
            if (groups < 2 or boundaries[0] != 0
                    or boundaries[-1] != hidden_states.shape[0]):
                return fallback()
            lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
            length = lengths[0]
            if length <= 0 or any(value != length for value in lengths):
                return fallback()

            sequence_length = hidden_states.shape[0]
            q, k, v = (
                module.qkv(hidden_states)
                .reshape(sequence_length, 3, module.num_heads, -1)
                .permute(1, 0, 2, 3).unbind(0)
            )
            q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
            q, k, v = (tensor.reshape(groups, length, module.num_heads, -1)
                       .transpose(1, 2) for tensor in (q, k, v))
            output, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](
                module, q, k, v, attention_mask=None, scaling=module.scaling,
                dropout=0.0, is_causal=False, **kwargs,
            )
            self.optimized_calls += 1
            return module.proj(output.reshape(sequence_length, -1).contiguous())

        return forward

    def remove(self) -> None:
        """Restore instance forwards; repeated removal is harmless."""
        for module, original, had_instance_forward in self._originals:
            if had_instance_forward:
                module.forward = original
            else:
                del module.forward
            del module._ttt_equal_length_sdpa
        self._originals.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()


__all__ = ["EqualLengthVisionSDPA"]
