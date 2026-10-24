"""Read-only Idea 3 diagnostics on native Spatial-TTT association weights.

Writing is unchanged. Questions supply native Q addresses, and independently
encoded, unlettered candidate texts supply native V directions. V is normalized
per head for comparable candidate scales; this is a diagnostic target choice,
not a trained text-to-visual mapping or the official answer decoder. The score
is the existing dot-product episode-minus-reset objective, never an MSE loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_frame.spatial_memory import FastWeightState, swiglu_read
from ttt_frame.spatial_probe import (
    CandidateAddressResult, _prepare_pair, _validate_search_options,
    association_delta, projected_inverse_address_search,
)


@dataclass(frozen=True)
class TextAddresses:
    queries: dict[int, Tensor]
    targets: dict[int, Tensor]
    query_tokens: int
    candidate_tokens: tuple[int, ...]
    position_offset: int
    prior_scores: tuple[float, ...] = ()


def _controller(engine_or_controller):
    return getattr(engine_or_controller, "controller", engine_or_controller)


class NativeQKVCapture:
    """Temporarily capture the final native Q/K/V token without changing it.

    ``last_visual`` selects the last token in the layer's video mask. Repeated
    writes overwrite records, retaining only the latest visual token per layer.
    The original methods are restored even when the model forward raises.
    """

    def __init__(self, engine_or_controller, *, selection: str = "last"):
        if selection not in {"last", "last_visual"}:
            raise ValueError("selection must be last or last_visual")
        self.controller = _controller(engine_or_controller)
        self.selection = selection
        self.records: dict[int, dict[str, Tensor]] = {}
        self._originals = {}
        self._active = False

    def __enter__(self):
        if self._active or any(getattr(layer, "_idea3_capture_active", False)
                               for layer in self.controller.layers.values()):
            raise RuntimeError("native QKV captures cannot overlap")
        self._active = True
        try:
            for index, layer in self.controller.layers.items():
                original = layer._fast_qkv
                self._originals[index] = ("_fast_qkv" in vars(layer), original)

                def capture(*args, _index=index, _layer=layer,
                            _original=original, **kwargs):
                    outputs = _original(*args, **kwargs)
                    if outputs[0].shape[0] != 1:
                        raise ValueError("native address capture requires batch size one")
                    position = outputs[0].shape[1] - 1
                    if self.selection == "last_visual":
                        mask = _layer._video_mask
                        if mask is None:
                            return outputs
                        selected = mask.reshape(-1).nonzero(as_tuple=False).flatten()
                        if selected.numel() == 0:
                            return outputs
                        position = int(selected[-1])
                    self.records[_index] = {
                        name: _layer.memory._heads(value[:, position:position + 1],
                                                    _layer.memory.fw_head_dim).detach().clone()
                        for name, value in zip(("q", "k", "v"), outputs)
                    }
                    return outputs

                layer._fast_qkv = capture
                layer._idea3_capture_active = True
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for index, (had_instance_method, original) in self._originals.items():
            layer = self.controller.layers[index]
            if had_instance_method:
                layer._fast_qkv = original
            else:
                delattr(layer, "_fast_qkv")
            if hasattr(layer, "_idea3_capture_active"):
                delattr(layer, "_idea3_capture_active")
        self._originals.clear()
        self._active = False


def snapshot_states(engine_or_controller, *, reset: bool = False,
                    device: torch.device | str | None = None) -> dict[int, FastWeightState]:
    """Copy only inference weights/norms; omit momentum and pending buffers.

    A current state with pending writes is rejected. ``reset=True`` constructs
    learned initial weights directly, avoiding allocation of zero momentum.
    These snapshots support reading only, not resumed online writing.
    """
    result = {}
    for index, layer in _controller(engine_or_controller).layers.items():
        state = None if reset else layer.state
        if state is not None and state.pending_tokens:
            raise ValueError("flush pending writes before taking a read-only snapshot")
        source = layer.memory if state is None else state
        weights = {}
        for name in ("w0", "w1", "w2"):
            value = getattr(source, name).detach()
            weights[name] = value.to(device=device or value.device, dtype=torch.float32).clone()
            norm = (value.norm(dim=2, keepdim=True) if state is None
                    else getattr(state, name + "_norm").detach())
            weights[name + "_norm"] = norm.to(device=device or value.device,
                                               dtype=torch.float32).clone()
        result[index] = FastWeightState(**weights,
                                        updates=0 if state is None else state.updates,
                                        tokens=0 if state is None else state.tokens)
    return result


def extract_text_addresses(engine, question: str,
                           option_texts: Sequence[str]) -> TextAddresses:
    """Encode plain text independently under learned initial Spatial states.

    No chat template, answer letters, candidate indices, padding, or shared
    candidate context enters encoding. The question-only query is captured at
    the final prompt token. Candidate targets use the final literal candidate
    token, without an assistant suffix. Prefix and candidate are tokenized
    separately to define the conditional likelihood boundary exactly. Priors
    are mean candidate-token log likelihoods under learned initial weights.
    All encodings use the recipient's stream position offset. Existing episode
    states are restored by identity.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    if isinstance(option_texts, str) or not option_texts or any(
        not isinstance(text, str) or not text.strip() for text in option_texts
    ):
        raise ValueError("option_texts must be a nonempty sequence of unlettered texts")
    controller = engine.controller
    if controller._active:
        raise RuntimeError("cannot extract addresses inside an active memory context")
    saved_states = {index: layer.state for index, layer in controller.layers.items()}
    offset = int(engine.position_offset)
    query_text = "Question: " + question.strip() + "\nCandidate action: "
    tokenizer = engine.processor.tokenizer
    prefix_ids = tokenizer(query_text, return_tensors="pt",
                           add_special_tokens=False)["input_ids"].to(engine.device)
    candidate_ids = [tokenizer(text.strip(), return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(engine.device)
                     for text in option_texts]
    sequences = [prefix_ids, *[torch.cat((prefix_ids, candidate), dim=1)
                               for candidate in candidate_ids]]
    records, lengths, priors = [], [], []
    try:
        for layer in controller.layers.values():
            layer.state = None
        for sequence_index, input_ids in enumerate(sequences):
            engine._clear_generation_state()
            if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
                raise ValueError("tokenizer must return one nonempty input sequence")
            option_length = (0 if sequence_index == 0
                             else candidate_ids[sequence_index - 1].shape[1])
            if sequence_index and not option_length:
                raise ValueError("candidate must tokenize to at least one token")
            positions = torch.arange(input_ids.shape[1], device=engine.device) + offset
            with torch.no_grad(), NativeQKVCapture(controller) as capture:
                with controller.context("read"):
                    output = engine.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                          position_ids=positions[None, None].expand(3, 1, -1),
                                          use_cache=False, logits_to_keep=option_length + 1)
                if sequence_index:
                    logits = output.logits[0, -(option_length + 1):-1].float()
                    labels = candidate_ids[sequence_index - 1][0]
                    if logits.shape[0] != labels.shape[0]:
                        raise RuntimeError("candidate logits do not match the token boundary")
                    prior = F.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).mean()
                    if not torch.isfinite(prior):
                        raise RuntimeError("candidate conditional likelihood is nonfinite")
                    priors.append(float(prior))
            if set(capture.records) != set(controller.layers):
                raise RuntimeError("not every Spatial layer produced a native address")
            records.append(capture.records)
            lengths.append(int(input_ids.shape[1]))
    finally:
        for index, layer in controller.layers.items():
            layer.state = saved_states[index]
        engine._clear_generation_state()
    queries = {index: records[0][index]["q"] for index in controller.layers}
    targets = {index: torch.stack([F.normalize(record[index]["v"], dim=-1, eps=1e-8)
                                   for record in records[1:]])
               for index in controller.layers}
    return TextAddresses(queries, targets, lengths[0], tuple(lengths[1:]), offset, tuple(priors))


def score_addresses(current: Mapping[int, FastWeightState],
                    reset: Mapping[int, FastWeightState], addresses: TextAddresses,
                    *, radius: float, steps: int = 3, step_size: float = 0.05,
                    regularization: float = 0.0) -> dict:
    """Return JSON-ready per-candidate means over head/token and Spatial layer.

    Search uses the existing summed per-layer objective; only aggregation divides
    by head/token count. ``search_objectives`` includes the displacement penalty.
    Empty memory produces exactly zero scores and displacement for every target.
    """
    keys = set(current)
    if not keys or keys != set(reset) or keys != set(addresses.queries) or keys != set(addresses.targets):
        raise ValueError("current/reset/query/target layer sets must be equal and nonempty")
    names = ("direct_scores", "search_scores", "direct_objectives", "search_objectives")
    per_layer, sums = {}, {name: [] for name in names}
    max_displacement = 0.0
    for index in sorted(keys):
        query, targets = addresses.queries[index], addresses.targets[index]
        result = batched_inverse_address_candidates(
            current[index], reset[index], query, targets, radius=radius,
            steps=steps, step_size=step_size, regularization=regularization,
        )
        divisor = query.shape[0] * (query.shape[1] if query.ndim == 3 else 1)
        entry = {}
        for name in names:
            values = getattr(result, name).detach().cpu() / divisor
            sums[name].append(values)
            entry[name] = values.tolist()
        entry["max_displacement"] = float(result.displacement_norms.max())
        max_displacement = max(max_displacement, entry["max_displacement"])
        per_layer[str(index)] = entry
    return {**{name: torch.stack(values).mean(0).tolist() for name, values in sums.items()},
            "per_layer": per_layer, "max_displacement": max_displacement,
            "layers": len(keys), "search": {"steps": steps, "radius": radius,
                                             "step_size": step_size,
                                             "regularization": regularization}}


def batched_inverse_address_candidates(
    current: FastWeightState, reset: FastWeightState, query: Tensor, targets: Tensor,
    *, radius: float, steps: int = 3, step_size: float = 0.05,
    regularization: float = 0.0,
) -> CandidateAddressResult:
    """Batch independent candidate searches as tokens under the same weights.

    The objective, projection and per-candidate best-iterate selection equal
    ``compare_inverse_address_candidates``. Each candidate owns its query and
    penalty; summing separable objectives only batches their gradient evaluation.
    GEMM layouts differ, so floating-point agreement is within tolerance rather
    than guaranteed bitwise. Neither current/reset states nor input addresses
    are modified. General multi-token queries retain separate candidate axes.
    """
    _validate_search_options(radius=radius, steps=steps, step_size=step_size,
                             regularization=regularization)
    if not isinstance(targets, Tensor) or not targets.is_floating_point():
        raise TypeError("targets must be a floating-point tensor")
    if (query.ndim not in (2, 3) or targets.ndim != query.ndim + 1
            or not targets.shape[0] or tuple(targets.shape[1:]) != tuple(query.shape)):
        raise ValueError("targets must have shape [candidates, *query.shape]")
    prepared, _, squeezed = _prepare_pair(current, reset, query, targets[0])
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("targets must contain only finite values")
    candidates = targets.shape[0]
    heads, tokens, width = prepared.shape
    origin = prepared.detach()[:, None].expand(heads, candidates, tokens, width).clone()
    fixed = targets.detach().to(device=prepared.device, dtype=prepared.dtype)
    if squeezed:
        fixed = fixed.unsqueeze(2)
    fixed = fixed.permute(1, 0, 2, 3).contiguous()
    current_weights = [getattr(current, name).detach() for name in ("w0", "w1", "w2")]
    reset_weights = [getattr(reset, name).detach() for name in ("w0", "w1", "w2")]

    def evaluate(addresses):
        flat = addresses.reshape(heads, candidates * tokens, width)
        delta = swiglu_read(*current_weights, flat) - swiglu_read(*reset_weights, flat)
        delta = delta.reshape(heads, candidates, tokens, fixed.shape[-1])
        scores = (delta * fixed).sum(dim=(0, 2, 3))
        penalty = regularization * (addresses - origin).square().sum(dim=(0, 2, 3))
        return scores, scores - penalty

    with torch.no_grad():
        direct, _ = evaluate(origin)
        if not bool(torch.isfinite(direct).all()):
            raise RuntimeError("direct inverse-address score is nonfinite")
        best, searched = origin.clone(), origin.clone()
        best_scores, best_objectives = direct.clone(), direct.clone()
    for _ in range(steps):
        with torch.enable_grad():
            searched = searched.detach().requires_grad_(True)
            _, objectives = evaluate(searched)
            if not bool(torch.isfinite(objectives).all()):
                raise RuntimeError("inverse-address objective became nonfinite")
            (gradient,) = torch.autograd.grad(objectives.sum(), searched)
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("inverse-address gradient became nonfinite")
        with torch.no_grad():
            displacement = searched + step_size * gradient - origin
            norms = displacement.norm(dim=-1, keepdim=True)
            scale = torch.clamp(radius / norms.clamp_min(torch.finfo(prepared.dtype).eps), max=1.0)
            searched = origin + displacement * scale
            candidate_scores, candidate_objectives = evaluate(searched)
            if not bool(torch.isfinite(candidate_objectives).all()):
                raise RuntimeError("inverse-address objective became nonfinite")
            improved = candidate_objectives > best_objectives
            best = torch.where(improved[None, :, None, None], searched, best)
            best_scores = torch.where(improved, candidate_scores, best_scores)
            best_objectives = torch.where(improved, candidate_objectives, best_objectives)
    displacement = (best - origin).norm(dim=-1).permute(1, 0, 2)
    best = best.permute(1, 0, 2, 3)
    return CandidateAddressResult(
        direct_scores=direct.detach(), search_scores=best_scores.detach(),
        direct_objectives=direct.detach().clone(), search_objectives=best_objectives.detach(),
        queries=(best.squeeze(2) if squeezed else best).detach(),
        displacement_norms=(displacement.squeeze(2) if squeezed else displacement).detach(),
    )


def visual_key_diagnostic(current: Mapping[int, FastWeightState],
                          reset: Mapping[int, FastWeightState], addresses: TextAddresses,
                          visual_records: Mapping[int, Mapping[str, Tensor]], *,
                          radius: float = 0.5, steps: int = 3,
                          step_size: float = 1.0, regularization: float = 0.01) -> dict:
    """Compare last observed visual K and text Q against the same visual V.

    This is a within-prefix association check, not future-action accuracy, and a
    final-frame probe cannot establish retention of older facts. Values are
    normalized per head in both comparisons, matching the candidate diagnostic.
    """
    if set(current) != set(reset) or set(current) != set(visual_records) or set(current) != set(addresses.queries):
        raise ValueError("visual diagnostic layer sets must match")
    per_layer = {}
    for index in sorted(current):
        visual = visual_records[index]
        target = F.normalize(visual["v"], dim=-1, eps=1e-8)
        search = projected_inverse_address_search(
            current[index], reset[index], addresses.queries[index], target,
            radius=radius, steps=steps, step_size=step_size, regularization=regularization,
        )
        divisor = target.shape[0] * target.shape[1]
        per_layer[str(index)] = {
            "visual_key": float(association_delta(current[index], reset[index], visual["k"],
                                                   target, reduction="mean").detach()),
            "text_key": float(association_delta(current[index], reset[index], addresses.queries[index],
                                                 target, reduction="mean").detach()),
            "inverse_text_key": float(search.search_score) / divisor,
            "inverse_text_objective": float(search.search_objective) / divisor,
            "max_displacement": float(search.displacement_norm.max()),
        }
    if not per_layer:
        raise ValueError("visual diagnostic requires nonempty states")
    return {"per_layer": per_layer,
            "max_displacement": max(row["max_displacement"] for row in per_layer.values()),
            **{name: sum(row[name] for row in per_layer.values()) / len(per_layer)
               for name in ("visual_key", "text_key", "inverse_text_key", "inverse_text_objective")}}
