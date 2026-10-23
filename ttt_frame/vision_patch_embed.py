"""Optional linear execution of Qwen3-VL's nonoverlapping patch convolution.

The processor already flattens each C x T x H x W patch separately. Its Conv3d
has exactly one output position per patch, so a linear map with the same flattened
weight and bias computes the same operation. This avoids an unusually slow
native Conv3d fallback on some BF16/CUDA installations. Different kernels can round
floating-point results differently; this is algebraic, not bitwise equivalence.
Enable explicitly for all compared arms and record the execution protocol.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.nn import functional as F


class LinearVisionPatchEmbed:
    """Install on model instances only; context exit or ``remove()`` restores.

    Accepts a complete Qwen3-VL, its visual tower, or its patch embedding module.
    No weights, model classes, installed packages, or attention boundaries change.
    Training uses the original forward, keeping this option limited to inference.
    """

    protocol_name = "qwen3_vl_linear_patch_embed_v1"

    def __init__(self, model: torch.nn.Module):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )

        modules = [module for module in model.modules()
                   if isinstance(module, Qwen3VLVisionPatchEmbed)]
        if not modules:
            raise ValueError("no Qwen3-VL vision patch embedding modules found")
        if any(hasattr(module, "_ttt_linear_patch_embed") for module in modules):
            raise ValueError("linear vision patch embedding is already installed")
        for module in modules:
            conv = module.proj
            kernel = (module.temporal_patch_size, module.patch_size, module.patch_size)
            if (not isinstance(conv, torch.nn.Conv3d)
                    or tuple(conv.kernel_size) != kernel
                    or tuple(conv.stride) != kernel
                    or tuple(conv.padding) != (0, 0, 0)
                    or tuple(conv.dilation) != (1, 1, 1)
                    or conv.groups != 1
                    or conv.in_channels != module.in_channels
                    or conv.out_channels != module.embed_dim):
                raise ValueError("patch convolution does not have the required geometry")

        self.optimized_calls = 0
        self.fallback_calls = 0
        self.module_count = len(modules)
        self._originals = []
        for module in modules:
            original = module.forward
            self._originals.append((module, original, "forward" in vars(module)))
            module._ttt_linear_patch_embed = self.protocol_name
            module.forward = MethodType(self._forward_for(original), module)

    def _forward_for(self, original):
        def forward(module, hidden_states):
            if module.training:
                self.fallback_calls += 1
                return original(hidden_states)
            width = module.in_channels * module.temporal_patch_size * module.patch_size**2
            patches = hidden_states.reshape(-1, width).to(dtype=module.proj.weight.dtype)
            self.optimized_calls += 1
            return F.linear(patches, module.proj.weight.flatten(1), module.proj.bias)

        return forward

    def remove(self) -> None:
        """Restore the original instance forwards; repeated removal is harmless."""
        for module, original, had_instance_forward in self._originals:
            if had_instance_forward:
                module.forward = original
            else:
                del module.forward
            del module._ttt_linear_patch_embed
        self._originals.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()


__all__ = ["LinearVisionPatchEmbed"]
