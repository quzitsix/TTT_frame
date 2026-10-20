"""Coordinate transport preflight for visual keys and SwiGLU fast weights.

The public convention is deliberately explicit:

* Samples are stored as **row vectors** in tensors shaped ``[samples, dim]``.
* ``transport`` is the corresponding column-vector map, so
  ``x_new_column = transport @ x_old_column`` and, in row-vector code,
  ``x_new = x_old @ transport.T``.
* The Spatial-TTT SwiGLU input weights have shape
  ``[..., hidden, dim]``.  Transporting both input paths with
  ``weight_new = weight_old @ transport.T`` preserves the function when an
  input moves from ``x_old`` to ``x_new`` and ``transport`` is orthogonal.

This module is independent of the Spatial model wrapper.  It is intended to
test the key-space assumption before changing the streaming model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlignmentMetrics:
    """Alignment error before and after applying a candidate transport."""

    match_count: int
    dimension: int
    identity_mse: float
    transported_mse: float
    identity_rmse: float
    transported_rmse: float
    relative_rmse_improvement: float


@dataclass(frozen=True)
class FitHeldoutMetrics:
    """Separate fit and held-out measurements for one fitted transport."""

    fit: AlignmentMetrics
    heldout: AlignmentMetrics | None


@dataclass(frozen=True)
class KeyTransportDecision:
    """Safe result of fitting and applying the preflight rejection rules.

    ``transport`` is always safe to apply: it is the identity on rejection.
    ``candidate_transport`` retains a fitted but rejected matrix for diagnosis.
    """

    accepted: bool
    reason: str
    transport: Tensor
    candidate_transport: Tensor | None
    metrics: FitHeldoutMetrics | None


def _validate_scalar(
    value: Real,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_match_pair(
    old_keys: Tensor,
    new_keys: Tensor,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, int]:
    if not isinstance(old_keys, Tensor) or not isinstance(new_keys, Tensor):
        raise ValueError(f"{name} keys must be torch tensors")
    if old_keys.ndim != 2 or new_keys.ndim != 2:
        raise ValueError(f"{name} keys must have shape [matches, dimension]")
    if old_keys.shape != new_keys.shape:
        raise ValueError(f"{name} old/new keys must have identical shapes")
    matches, dimension = old_keys.shape
    if dimension < 1 or (matches < 1 and not allow_empty):
        raise ValueError(f"{name} keys must contain at least one match and one feature")
    if not old_keys.is_floating_point() or not new_keys.is_floating_point():
        raise ValueError(f"{name} keys must use a floating-point dtype")
    if old_keys.dtype != new_keys.dtype or old_keys.device != new_keys.device:
        raise ValueError(f"{name} old/new keys must have the same dtype and device")
    if not bool(torch.isfinite(old_keys).all()) or not bool(torch.isfinite(new_keys).all()):
        raise ValueError(f"{name} keys contain nonfinite values")
    return matches, dimension


def _solver_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _identity_for(keys: Tensor) -> Tensor:
    return torch.eye(keys.shape[-1], dtype=_solver_dtype(keys), device=keys.device)


def _validate_transport_matrix(
    transport: Tensor,
    dimension: int,
    *,
    device: torch.device,
    require_orthogonal: bool,
    orthogonal_atol: float,
) -> None:
    if not isinstance(transport, Tensor):
        raise ValueError("transport must be a torch tensor")
    if transport.ndim < 2 or transport.shape[-2:] != (dimension, dimension):
        raise ValueError(
            f"transport must end in shape [{dimension}, {dimension}]"
        )
    if not transport.is_floating_point():
        raise ValueError("transport must use a floating-point dtype")
    if transport.device != device:
        raise ValueError("transport and inputs must be on the same device")
    if not bool(torch.isfinite(transport).all()):
        raise ValueError("transport contains nonfinite values")
    if require_orthogonal:
        check = transport.double() if transport.dtype == torch.float64 else transport.float()
        identity = torch.eye(dimension, dtype=check.dtype, device=check.device)
        gram = check.transpose(-1, -2) @ check
        if not torch.allclose(gram, identity.expand_as(gram), atol=orthogonal_atol, rtol=1e-5):
            raise ValueError("transport must be orthogonal")


@torch.no_grad()
def regularized_orthogonal_procrustes(
    old_keys: Tensor,
    new_keys: Tensor,
    *,
    regularization: float = 0.0,
) -> Tensor:
    """Fit the orthogonal column-vector map from old keys to new keys.

    Inputs are row-vector matrices ``X`` and ``Y``.  The returned ``R`` solves

    ``min_R ||X @ R.T - Y||_F^2 + regularization * ||R - I||_F^2``
    subject to ``R.T @ R = I``.

    The regularizer is implemented exactly by taking the SVD of
    ``X.T @ Y + regularization * I``.  Solves run in FP32 (FP64 when the input
    is FP64), and the result is detached because the preflight does not
    differentiate through the SVD.
    """

    _, dimension = _validate_match_pair(
        old_keys, new_keys, name="fit", allow_empty=False
    )
    regularization = _validate_scalar(
        regularization, "regularization", minimum=0.0
    )
    dtype = _solver_dtype(old_keys)
    old = old_keys.detach().to(dtype=dtype)
    new = new_keys.detach().to(dtype=dtype)
    cross = old.transpose(0, 1) @ new
    if regularization:
        cross = cross + regularization * torch.eye(
            dimension, dtype=dtype, device=old.device
        )
    left, _, right_h = torch.linalg.svd(cross, full_matrices=False)
    # Q = left @ right_h is the row-vector map X @ Q.  Return R = Q.T so the
    # public matrix obeys the column-vector convention x_new = R @ x_old.
    return (left @ right_h).transpose(0, 1).contiguous()


def apply_key_transport(keys: Tensor, transport: Tensor) -> Tensor:
    """Apply ``x_new = R x_old`` to keys represented as row vectors."""

    if not isinstance(keys, Tensor) or keys.ndim < 1 or keys.shape[-1] < 1:
        raise ValueError("keys must be a tensor ending in a nonempty feature dimension")
    if not keys.is_floating_point() or not bool(torch.isfinite(keys).all()):
        raise ValueError("keys must contain finite floating-point values")
    _validate_transport_matrix(
        transport,
        keys.shape[-1],
        device=keys.device,
        require_orthogonal=False,
        orthogonal_atol=0.0,
    )
    dtype = torch.promote_types(keys.dtype, transport.dtype)
    try:
        return torch.matmul(
            keys.to(dtype=dtype), transport.to(dtype=dtype).transpose(-1, -2)
        )
    except RuntimeError as error:
        raise ValueError("keys and transport have incompatible batch dimensions") from error


def alignment_metrics(
    old_keys: Tensor,
    new_keys: Tensor,
    transport: Tensor,
) -> AlignmentMetrics:
    """Measure identity and transported errors for one set of matches."""

    matches, dimension = _validate_match_pair(
        old_keys, new_keys, name="metric", allow_empty=False
    )
    _validate_transport_matrix(
        transport,
        dimension,
        device=old_keys.device,
        require_orthogonal=False,
        orthogonal_atol=0.0,
    )
    if transport.ndim != 2:
        raise ValueError("alignment metrics require one two-dimensional transport")
    dtype = torch.promote_types(_solver_dtype(old_keys), transport.dtype)
    old = old_keys.detach().to(dtype=dtype)
    new = new_keys.detach().to(dtype=dtype)
    moved = apply_key_transport(old, transport.detach().to(dtype=dtype))
    identity_mse = float((old - new).square().mean().item())
    transported_mse = float((moved - new).square().mean().item())
    identity_rmse = math.sqrt(identity_mse)
    transported_rmse = math.sqrt(transported_mse)
    epsilon = torch.finfo(dtype).eps
    relative_improvement = (
        identity_rmse - transported_rmse
    ) / max(identity_rmse, epsilon)
    return AlignmentMetrics(
        match_count=matches,
        dimension=dimension,
        identity_mse=identity_mse,
        transported_mse=transported_mse,
        identity_rmse=identity_rmse,
        transported_rmse=transported_rmse,
        relative_rmse_improvement=relative_improvement,
    )


def fit_heldout_metrics(
    transport: Tensor,
    fit_old: Tensor,
    fit_new: Tensor,
    heldout_old: Tensor | None = None,
    heldout_new: Tensor | None = None,
) -> FitHeldoutMetrics:
    """Return fit and held-out errors without pooling the two splits."""

    if (heldout_old is None) != (heldout_new is None):
        raise ValueError("heldout_old and heldout_new must be supplied together")
    fit = alignment_metrics(fit_old, fit_new, transport)
    heldout = None
    if heldout_old is not None and heldout_new is not None:
        heldout = alignment_metrics(heldout_old, heldout_new, transport)
        if heldout.dimension != fit.dimension:
            raise ValueError("fit and held-out keys must use the same dimension")
    return FitHeldoutMetrics(fit=fit, heldout=heldout)


def fit_key_transport(
    fit_old: Tensor,
    fit_new: Tensor,
    *,
    heldout_old: Tensor | None = None,
    heldout_new: Tensor | None = None,
    regularization: float = 0.0,
    min_fit_matches: int = 3,
    min_heldout_matches: int = 1,
    min_relative_improvement: float = 0.0,
    max_heldout_rmse: float | None = None,
    scene_cut: bool = False,
) -> KeyTransportDecision:
    """Fit a transport and return identity when preflight checks reject it.

    Rejection is deterministic and ordered: explicit scene cut, too few fit
    matches, too few held-out matches, insufficient held-out improvement, then
    excessive held-out RMSE.  When no held-out split is supplied, the last two
    checks use the fit split and ``max_heldout_rmse`` acts as a generic maximum.
    """

    fit_count, dimension = _validate_match_pair(
        fit_old, fit_new, name="fit", allow_empty=True
    )
    if isinstance(min_fit_matches, bool) or not isinstance(min_fit_matches, int):
        raise ValueError("min_fit_matches must be a positive integer")
    if isinstance(min_heldout_matches, bool) or not isinstance(min_heldout_matches, int):
        raise ValueError("min_heldout_matches must be a positive integer")
    if min_fit_matches < 1 or min_heldout_matches < 1:
        raise ValueError("minimum match counts must be positive")
    regularization = _validate_scalar(
        regularization, "regularization", minimum=0.0
    )
    min_relative_improvement = _validate_scalar(
        min_relative_improvement, "min_relative_improvement"
    )
    if max_heldout_rmse is not None:
        max_heldout_rmse = _validate_scalar(
            max_heldout_rmse, "max_heldout_rmse", minimum=0.0
        )
    if not isinstance(scene_cut, bool):
        raise ValueError("scene_cut must be a boolean")
    if (heldout_old is None) != (heldout_new is None):
        raise ValueError("heldout_old and heldout_new must be supplied together")

    heldout_count = None
    if heldout_old is not None and heldout_new is not None:
        heldout_count, heldout_dimension = _validate_match_pair(
            heldout_old, heldout_new, name="held-out", allow_empty=True
        )
        if heldout_dimension != dimension:
            raise ValueError("fit and held-out keys must use the same dimension")
        if heldout_old.dtype != fit_old.dtype or heldout_old.device != fit_old.device:
            raise ValueError("fit and held-out keys must have the same dtype and device")

    identity = _identity_for(fit_old)

    def reject(reason: str) -> KeyTransportDecision:
        return KeyTransportDecision(False, reason, identity, None, None)

    if scene_cut:
        return reject("scene_cut")
    if fit_count < min_fit_matches:
        return reject("insufficient_fit_matches")
    if heldout_count is not None and heldout_count < min_heldout_matches:
        return reject("insufficient_heldout_matches")

    candidate = regularized_orthogonal_procrustes(
        fit_old, fit_new, regularization=regularization
    )
    metrics = fit_heldout_metrics(
        candidate, fit_old, fit_new, heldout_old, heldout_new
    )
    decision_metrics = metrics.heldout or metrics.fit
    if decision_metrics.relative_rmse_improvement < min_relative_improvement:
        return KeyTransportDecision(
            False,
            "improvement_below_threshold",
            identity,
            candidate,
            metrics,
        )
    if (
        max_heldout_rmse is not None
        and decision_metrics.transported_rmse > max_heldout_rmse
    ):
        return KeyTransportDecision(
            False,
            "rmse_above_threshold",
            identity,
            candidate,
            metrics,
        )
    return KeyTransportDecision(True, "accepted", candidate, candidate, metrics)


def transport_swiglu_input_weights(
    w0: Tensor,
    w2: Tensor,
    transport: Tensor,
    *,
    orthogonal_atol: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Synchronously transport the two SwiGLU input-coordinate matrices.

    ``w1`` is intentionally absent because it maps the hidden activation back
    to the fixed value space.  Inputs are never mutated.  A single ``[d, d]``
    matrix broadcasts over heads, while ``[..., d, d]`` supports one matrix per
    leading batch/head index.
    """

    if not isinstance(w0, Tensor) or not isinstance(w2, Tensor):
        raise ValueError("w0 and w2 must be torch tensors")
    if w0.ndim < 2 or w2.ndim < 2 or w0.shape != w2.shape:
        raise ValueError("w0 and w2 must have identical [..., hidden, dimension] shapes")
    if not w0.is_floating_point() or not w2.is_floating_point():
        raise ValueError("w0 and w2 must use a floating-point dtype")
    if w0.dtype != w2.dtype or w0.device != w2.device:
        raise ValueError("w0 and w2 must have the same dtype and device")
    if not bool(torch.isfinite(w0).all()) or not bool(torch.isfinite(w2).all()):
        raise ValueError("w0 and w2 contain nonfinite values")
    orthogonal_atol = _validate_scalar(
        orthogonal_atol, "orthogonal_atol", minimum=0.0
    )
    dimension = w0.shape[-1]
    _validate_transport_matrix(
        transport,
        dimension,
        device=w0.device,
        require_orthogonal=True,
        orthogonal_atol=orthogonal_atol,
    )
    try:
        transform_t = transport.to(dtype=w0.dtype).transpose(-1, -2)
        return torch.matmul(w0, transform_t), torch.matmul(w2, transform_t)
    except RuntimeError as error:
        raise ValueError("weights and transport have incompatible batch dimensions") from error
