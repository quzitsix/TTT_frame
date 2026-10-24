"""A shared, train-only semantic MCQ readout for Idea 01 forecasts.

The memory forecasts visual features without seeing candidate text or labels.
This separate supervised ridge probe maps its pooled forecasts into a frozen
language embedding space. Every arm uses the same probe and candidate cosine
scores; option letters are never features. This is not VLM generation, an
outer-trained TTT module, or an assertion that raw visual/text features align.

Training rows are case x arm, but the independent sample count is the number
of cases. The objective is mean squared error + lambda * squared weight norm;
the dual solve consequently uses ``n_rows * lambda``. Repeating every arm does
not strengthen the supervision or change the regularization convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _finite_cpu(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def pool_forecasts(forecasts: Tensor) -> Tensor:
    """Mean over spatial tokens, then concatenate horizons in their given order.

    ``[..., horizon, spatial_token, feature] -> [..., horizon * feature]``.
    Four 256-dimensional horizons give a 1024-dimensional readout input.
    """
    if not isinstance(forecasts, Tensor) or forecasts.ndim < 3:
        raise ValueError("forecasts must end in horizon, spatial_token, feature axes")
    if any(size < 1 for size in forecasts.shape[-3:]):
        raise ValueError("forecast axes must be nonempty")
    return _finite_cpu(forecasts, "forecasts").mean(dim=-2).flatten(start_dim=-2)


def candidate_prompt(question: str, candidate: str) -> str:
    """Encode one candidate independently, without a label or option position."""
    if not all(isinstance(item, str) and item.strip() for item in (question, candidate)):
        raise ValueError("question and candidate must be nonempty text")
    return f"问题：{question.strip()}\n候选动作：{candidate.strip()}"


@dataclass
class SharedRidgeReadout:
    feature_mean: Tensor
    feature_scale: Tensor
    text_mean: Tensor
    text_scale: Tensor
    coefficients: Tensor
    target_mean: Tensor
    ridge_lambda: float
    independent_train_cases: int
    train_rows: int
    train_ids: tuple[str, ...]
    arm_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        train_features: Tensor,
        train_options: Tensor,
        train_correct_indices: Tensor,
        *,
        ridge_lambda: float,
        train_ids: Sequence[str] | None = None,
        arm_names: Sequence[str] | None = None,
    ) -> "SharedRidgeReadout":
        """Fit once using ``[case, arm, feature]`` and ``[case, choice, text]``.

        Pass only training cases. Selection on validation belongs to the caller;
        this function never opens a manifest, split, answer key, or media file.
        All arms must be present for every case and receive exactly equal weight.
        Text whitening uses every training candidate once, independent of labels;
        the target is the correctly labelled candidate in that same space.
        """
        x = _finite_cpu(train_features, "train_features")
        candidates = _finite_cpu(train_options, "train_options")
        if x.ndim != 3 or candidates.ndim != 3 or x.shape[0] != candidates.shape[0]:
            raise ValueError("expected features [case, arm, feature] and options [case, choice, text]")
        n, arms, width = x.shape
        if n < 2 or min(arms, width, candidates.shape[2]) < 1 or candidates.shape[1] < 2:
            raise ValueError("need two cases/choices and nonempty feature and arm dimensions")
        if isinstance(ridge_lambda, bool) or not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be finite and positive")
        if (not isinstance(train_correct_indices, Tensor)
                or train_correct_indices.shape != (n,)
                or train_correct_indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError("correct indices must be a length-case integer tensor")
        labels = train_correct_indices.detach().cpu().long()
        if not bool(((labels >= 0) & (labels < candidates.shape[1])).all()):
            raise ValueError("correct index is outside available options")
        ids = tuple(train_ids) if train_ids is not None else ()
        names = tuple(arm_names) if arm_names is not None else ()
        if ids and (len(ids) != n or len(set(ids)) != n):
            raise ValueError("train_ids must uniquely identify all independent cases")
        if names and (len(names) != arms or len(set(names)) != arms):
            raise ValueError("arm_names must uniquely identify each arm")

        rows = x.reshape(n * arms, width)
        feature_mean = rows.mean(dim=0)
        feature_scale = rows.std(dim=0, correction=0).clamp_min(1e-6)
        standardized = (rows - feature_mean) / feature_scale
        text_rows = candidates.flatten(0, 1)
        text_mean = text_rows.mean(dim=0)
        text_scale = text_rows.std(dim=0, correction=0).clamp_min(1e-6)
        transformed = F.normalize((candidates - text_mean) / text_scale, dim=-1)
        targets = transformed[torch.arange(n), labels]
        target_mean = targets.mean(dim=0)
        repeated_targets = (targets - target_mean).repeat_interleave(arms, dim=0)
        gram = standardized @ standardized.T
        gram.diagonal().add_(rows.shape[0] * float(ridge_lambda))
        coefficients = standardized.T @ torch.linalg.solve(gram, repeated_targets)
        return cls(
            feature_mean, feature_scale, text_mean, text_scale, coefficients,
            target_mean, float(ridge_lambda), n, n * arms, ids, names,
        )

    def transform_options(self, options: Tensor) -> Tensor:
        candidates = _finite_cpu(options, "options")
        if candidates.ndim != 3 or candidates.shape[-1] != self.text_mean.numel():
            raise ValueError("options must have shape [case, choice, text_embedding_dimension]")
        return F.normalize((candidates - self.text_mean) / self.text_scale, dim=-1)

    def predict_scores(self, features: Tensor, options: Tensor) -> Tensor:
        """Return cosine scores [case, arm, choice], or [case, choice]."""
        x = _finite_cpu(features, "features")
        candidates = self.transform_options(options)
        if (x.ndim not in (2, 3) or x.shape[-1] != self.feature_mean.numel()
                or x.shape[0] != candidates.shape[0]):
            raise ValueError("features must be [case, (arm,), feature] and match options")
        predicted = F.normalize(
            ((x - self.feature_mean) / self.feature_scale) @ self.coefficients + self.target_mean,
            dim=-1,
        )
        if x.ndim == 2:
            return torch.einsum("ne,nce->nc", predicted, candidates)
        return torch.einsum("nae,nce->nac", predicted, candidates)

    def blind_scores(self, options: Tensor) -> Tensor:
        """Train-fitted text prior, with no visual input and no option-index prior.

        Every case is scored against the mean correctly labelled training text.
        The mean is computed once per independent case, not per arm.
        """
        return torch.einsum(
            "e,nce->nc", F.normalize(self.target_mean, dim=0), self.transform_options(options)
        )

    def to_dict(self) -> dict:
        return {
            "kind": "idea01_shared_semantic_ridge_readout", "format_version": 1,
            "feature_mean": self.feature_mean.clone(), "feature_scale": self.feature_scale.clone(),
            "text_mean": self.text_mean.clone(), "text_scale": self.text_scale.clone(),
            "coefficients": self.coefficients.clone(), "target_mean": self.target_mean.clone(),
            "ridge_lambda": self.ridge_lambda,
            "independent_train_cases": self.independent_train_cases, "train_rows": self.train_rows,
            "train_ids": list(self.train_ids), "arm_names": list(self.arm_names),
            "objective": "mean_squared_error_plus_lambda_squared_weight_norm",
            "text_transform": "train_candidate_center_scale_then_l2_normalize",
            "readout": "predicted_text_cosine_to_each_candidate",
        }

    @classmethod
    def from_dict(cls, value: dict) -> "SharedRidgeReadout":
        if value.get("kind") != "idea01_shared_semantic_ridge_readout" or value.get("format_version") != 1:
            raise ValueError("unsupported readout checkpoint")
        names = ("feature_mean", "feature_scale", "text_mean", "text_scale", "coefficients", "target_mean")
        tensors = {name: _finite_cpu(value[name], name).clone() for name in names}
        width, text_width = tensors["feature_mean"].numel(), tensors["text_mean"].numel()
        expected = {"feature_mean": (width,), "feature_scale": (width,),
                    "text_mean": (text_width,), "text_scale": (text_width,),
                    "coefficients": (width, text_width), "target_mean": (text_width,)}
        if any(tensors[name].shape != shape for name, shape in expected.items()):
            raise ValueError("readout tensor dimensions do not match")
        if any(not bool((tensors[name] > 0).all()) for name in ("feature_scale", "text_scale")):
            raise ValueError("readout scales must be positive")
        return cls(**tensors, ridge_lambda=float(value["ridge_lambda"]),
                   independent_train_cases=int(value["independent_train_cases"]),
                   train_rows=int(value["train_rows"]), train_ids=tuple(value["train_ids"]),
                   arm_names=tuple(value["arm_names"]))
