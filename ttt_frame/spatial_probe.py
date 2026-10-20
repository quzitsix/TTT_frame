"""CPU-friendly probes for reading frozen Spatial-TTT fast weights.

The Spatial memory is trained online with a dot-product association objective:
writing ``(k, v)`` increases ``f_W(k) dot v``.  This module uses the same
quantity to ask whether an episode changed the memory at a proposed address.
It deliberately does not use an MSE reconstruction objective.

The helpers operate on head-flattened tensors.  A query and target may have
shape ``[batch * heads, tokens, width]`` or, for one token per head,
``[batch * heads, width]``.  Searches optimize only a temporary query tensor;
the current and reset :class:`~ttt_frame.spatial_memory.FastWeightState`
objects are detached and never modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ttt_frame.spatial_memory import FastWeightState, swiglu_read


Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True)
class InverseAddressResult:
    """Direct and searched association scores for one candidate target."""

    direct_score: Tensor
    search_score: Tensor
    direct_objective: Tensor
    search_objective: Tensor
    query: Tensor
    displacement_norm: Tensor
    steps: int


@dataclass(frozen=True)
class CandidateAddressResult:
    """Candidate-wise results; the leading dimension follows target order."""

    direct_scores: Tensor
    search_scores: Tensor
    direct_objectives: Tensor
    search_objectives: Tensor
    queries: Tensor
    displacement_norms: Tensor


def _as_token_tensor(tensor: Tensor, *, name: str) -> tuple[Tensor, bool]:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if tensor.ndim == 2:
        return tensor.unsqueeze(1), True
    if tensor.ndim == 3:
        return tensor, False
    raise ValueError(f"{name} must have shape [batch * heads, width] or "
                     "[batch * heads, tokens, width]")


def _check_states(current: FastWeightState, reset: FastWeightState) -> None:
    for name in ("w0", "w1", "w2"):
        current_weight = getattr(current, name)
        reset_weight = getattr(reset, name)
        if current_weight.shape != reset_weight.shape:
            raise ValueError(f"current and reset {name} shapes do not match")
        if current_weight.device != reset_weight.device:
            raise ValueError("current and reset states must use the same device")
        if current_weight.dtype != reset_weight.dtype:
            raise ValueError("current and reset states must use the same dtype")
    if current.w0.ndim != 3 or current.w1.ndim != 3 or current.w2.ndim != 3:
        raise ValueError("fast weights must be three-dimensional")
    if current.w0.shape != current.w2.shape:
        raise ValueError("w0 and w2 shapes must match")
    if current.w0.shape[0] != current.w1.shape[0]:
        raise ValueError("fast weights must have the same head-flattened batch")
    if current.w0.shape[1] != current.w1.shape[2]:
        raise ValueError("fast-weight intermediate dimensions do not match")


def _prepare_pair(
    current: FastWeightState,
    reset: FastWeightState,
    query: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, bool]:
    _check_states(current, reset)
    query, squeezed = _as_token_tensor(query, name="query")
    target, target_squeezed = _as_token_tensor(target, name="target")
    if squeezed != target_squeezed or query.shape[:2] != target.shape[:2]:
        raise ValueError("query and target must have matching head and token dimensions")
    expected_query = (current.w0.shape[0], query.shape[1], current.w0.shape[2])
    expected_target = (current.w1.shape[0], query.shape[1], current.w1.shape[1])
    if tuple(query.shape) != expected_query:
        raise ValueError(f"query shape must be {expected_query}, got {tuple(query.shape)}")
    if tuple(target.shape) != expected_target:
        raise ValueError(f"target shape must be {expected_target}, got {tuple(target.shape)}")
    device, dtype = current.w0.device, current.w0.dtype
    query = query.to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(query).all()):
        raise ValueError("query must contain only finite values")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target must contain only finite values")
    return query, target, squeezed


def _association_delta_prepared(
    current: FastWeightState,
    reset: FastWeightState,
    query: Tensor,
    target: Tensor,
    reduction: Reduction,
) -> Tensor:
    current_read = swiglu_read(
        current.w0.detach(), current.w1.detach(), current.w2.detach(), query
    )
    reset_read = swiglu_read(
        reset.w0.detach(), reset.w1.detach(), reset.w2.detach(), query
    )
    token_scores = ((current_read - reset_read) * target).sum(dim=-1)
    if reduction == "none":
        return token_scores
    if reduction == "mean":
        return token_scores.mean()
    if reduction == "sum":
        return token_scores.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def association_delta(
    current: FastWeightState,
    reset: FastWeightState,
    query: Tensor,
    target: Tensor,
    *,
    reduction: Reduction = "sum",
) -> Tensor:
    """Return the episode-minus-reset dot-product association.

    The score is ``<f_current(query), target> - <f_reset(query), target>``.
    With ``reduction='none'`` the result has shape ``[batch * heads, tokens]``
    (or ``[batch * heads]`` when two-dimensional inputs were supplied).
    """

    query, target, squeezed = _prepare_pair(current, reset, query, target)
    result = _association_delta_prepared(current, reset, query, target, reduction)
    if reduction == "none" and squeezed:
        result = result.squeeze(1)
    return result


def _validate_search_options(
    *, radius: float, steps: int, step_size: float, regularization: float
) -> None:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    for name, value in (
        ("radius", radius),
        ("step_size", step_size),
        ("regularization", regularization),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if steps and step_size == 0:
        raise ValueError("step_size must be positive when steps is nonzero")


def _project_rows(query: Tensor, origin: Tensor, radius: float) -> Tensor:
    displacement = query - origin
    norms = displacement.norm(dim=-1, keepdim=True)
    scale = torch.clamp(radius / norms.clamp_min(torch.finfo(query.dtype).eps), max=1.0)
    return origin + displacement * scale


def projected_inverse_address_search(
    current: FastWeightState,
    reset: FastWeightState,
    query: Tensor,
    target: Tensor,
    *,
    radius: float,
    steps: int = 3,
    step_size: float = 0.05,
    regularization: float = 0.0,
) -> InverseAddressResult:
    """Maximize episodic association by changing only a temporary address.

    Each head/token address is projected onto an L2 ball of ``radius`` around
    its direct query.  The optimized objective is the summed association delta
    minus ``regularization * ||query - direct_query||^2``.  Reported
    ``direct_score`` and ``search_score`` are the unregularized association
    deltas, so a caller can distinguish evidence fit from the distance penalty.
    """

    _validate_search_options(
        radius=radius, steps=steps, step_size=step_size,
        regularization=regularization,
    )
    prepared_query, prepared_target, squeezed = _prepare_pair(
        current, reset, query, target
    )
    origin = prepared_query.detach().clone()
    target_fixed = prepared_target.detach()

    with torch.no_grad():
        direct_score = _association_delta_prepared(
            current, reset, origin, target_fixed, "sum"
        )
        if not bool(torch.isfinite(direct_score)):
            raise RuntimeError("direct inverse-address score is nonfinite")
    best_query = origin.clone()
    best_score = direct_score.detach().clone()
    best_objective = direct_score.detach().clone()
    searched = origin.clone()
    for _ in range(steps):
        # Spatial ingestion and readout normally run under ``torch.no_grad``.
        # Re-enable gradients only for the disposable address tensor.
        with torch.enable_grad():
            searched = searched.detach().requires_grad_(True)
            score = _association_delta_prepared(
                current, reset, searched, target_fixed, "sum"
            )
            penalty = regularization * (searched - origin).square().sum()
            objective = score - penalty
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("inverse-address objective became nonfinite")
            (gradient,) = torch.autograd.grad(objective, searched)
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("inverse-address gradient became nonfinite")
        with torch.no_grad():
            searched = _project_rows(searched + step_size * gradient, origin, radius)
            candidate_score = _association_delta_prepared(
                current, reset, searched, target_fixed, "sum"
            )
            candidate_penalty = regularization * (searched - origin).square().sum()
            candidate_objective = candidate_score - candidate_penalty
            if not bool(torch.isfinite(candidate_objective)):
                raise RuntimeError("inverse-address objective became nonfinite")
            if bool(candidate_objective > best_objective):
                best_query = searched.detach().clone()
                best_score = candidate_score.detach().clone()
                best_objective = candidate_objective.detach().clone()

    with torch.no_grad():
        displacement = (best_query - origin).norm(dim=-1)
        output_query = best_query.squeeze(1) if squeezed else best_query
        output_displacement = displacement.squeeze(1) if squeezed else displacement
    return InverseAddressResult(
        direct_score=direct_score.detach(),
        search_score=best_score,
        direct_objective=direct_score.detach(),
        search_objective=best_objective,
        query=output_query.detach(),
        displacement_norm=output_displacement.detach(),
        steps=steps,
    )


def compare_inverse_address_candidates(
    current: FastWeightState,
    reset: FastWeightState,
    query: Tensor,
    targets: Tensor,
    *,
    radius: float,
    steps: int = 3,
    step_size: float = 0.05,
    regularization: float = 0.0,
) -> CandidateAddressResult:
    """Run the same inverse-address search independently for each target.

    ``targets`` adds a candidate dimension before the query shape.  No
    candidate shares an optimized address with another, so permuting targets
    only permutes the returned rows.
    """

    if not isinstance(targets, Tensor) or not targets.is_floating_point():
        raise TypeError("targets must be a floating-point tensor")
    expected_ndim = query.ndim + 1
    if query.ndim not in (2, 3) or targets.ndim != expected_ndim:
        raise ValueError("targets must have shape [candidates, *query.shape]")
    if targets.shape[0] < 1 or tuple(targets.shape[1:]) != tuple(query.shape):
        raise ValueError("targets must have a nonempty candidate axis and match query.shape")
    results = [
        projected_inverse_address_search(
            current, reset, query, target,
            radius=radius, steps=steps, step_size=step_size,
            regularization=regularization,
        )
        for target in targets
    ]
    return CandidateAddressResult(
        direct_scores=torch.stack([result.direct_score for result in results]),
        search_scores=torch.stack([result.search_score for result in results]),
        direct_objectives=torch.stack([result.direct_objective for result in results]),
        search_objectives=torch.stack([result.search_objective for result in results]),
        queries=torch.stack([result.query for result in results]),
        displacement_norms=torch.stack([result.displacement_norm for result in results]),
    )


__all__ = [
    "CandidateAddressResult",
    "InverseAddressResult",
    "association_delta",
    "compare_inverse_address_candidates",
    "projected_inverse_address_search",
]
