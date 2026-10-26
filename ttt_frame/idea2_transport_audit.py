"""Real-video coordinate-transport preflight, independent of frozen runners.

The fixed PCA subspace is learned only from train-day visual tokens. Matching
never receives labels or object identities. Fit, gate, and final audit tokens
are disjoint; accepting on gate error cannot select the audit observations.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math

import torch
from torch.nn import functional as F

from ttt_frame.forecast_features import QwenImageFeatures
from ttt_frame.key_transport import alignment_metrics, regularized_orthogonal_procrustes


RANK = 16
COSINE_THRESHOLD = 0.7
MIN_MATCHES = 32
MIN_IMPROVEMENT = 0.15
REGULARIZATIONS = (0.0, 0.01, 0.1, 1.0, 10.0)


def pair_schedule() -> list[dict]:
    """Twelve representative adjacent pairs, fixed without inspecting pixels."""
    return [dict(role=role, old_seconds=float(t), new_seconds=float(t + .5))
            for role, times in (("context", (150, 450, 750, 1050, 1350, 1650)),
                                ("recent", (0, 5, 10, 15, 20, 25))) for t in times]


class DenseSpatialFeatures(QwenImageFeatures):
    """Expose every native merged spatial token, without 8-token pooling."""

    @torch.inference_mode()
    def encode(self, images: list) -> list[torch.Tensor]:
        batch = self.processor(images=images, return_tensors="pt")
        grid = batch["image_grid_thw"].cpu()
        if bool((grid[:, 0] != 1).any()):
            raise ValueError("each still image must have an independent temporal grid")
        tokens, _ = self.model(batch["pixel_values"].to(self.device, self.dtype), grid_thw=grid)
        offset, result = 0, []
        for _, h, w in grid.tolist():
            count = (h // self.model.spatial_merge_size) * (w // self.model.spatial_merge_size)
            projected = tokens[offset:offset + count].float() @ self.projection
            value = F.layer_norm(projected, (projected.shape[-1],), eps=1e-5)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite visual tokens")
            result.append(value.cpu())
            offset += count
        if offset != len(tokens):
            raise ValueError("native token count mismatch")
        return result


def stable_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def fit_pca(token_batches, rank: int = RANK) -> dict:
    count, total, second = 0, None, None
    for batch in token_batches:
        if batch.ndim != 2 or not bool(torch.isfinite(batch).all()):
            raise ValueError("PCA requires finite token matrices")
        x = batch.double()
        if total is None:
            total, second = torch.zeros(x.shape[1], dtype=x.dtype), torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype)
        total += x.sum(0)
        second += x.T @ x
        count += len(x)
    if count < rank + 1 or total is None:
        raise ValueError("too few training tokens")
    mean = total / count
    covariance = second / count - mean[:, None] * mean[None, :]
    values, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -rank:].flip(1)
    # Fix eigenvector signs for stable saved representations.
    idx = basis.abs().argmax(0)
    signs = torch.sign(basis[idx, torch.arange(rank)])
    basis *= signs
    return {"mean": mean.float(), "basis": basis.float(), "token_count": count,
            "explained_variance_ratio": float(values[-rank:].sum() / values.clamp_min(0).sum())}


def matched_indices(old: torch.Tensor, new: torch.Tensor, threshold: float = COSINE_THRESHOLD):
    if old.ndim != 2 or new.ndim != 2 or old.shape[1] != new.shape[1]:
        raise ValueError("matching requires compatible token matrices")
    if not 0 <= threshold <= 1 or not bool(torch.isfinite(old).all()) or not bool(torch.isfinite(new).all()):
        raise ValueError("invalid matching inputs")
    sim = F.normalize(old.float(), dim=-1) @ F.normalize(new.float(), dim=-1).T
    targets = sim.argmax(1)
    sources = sim.argmax(0)
    ids = torch.arange(len(old))
    keep = (sources[targets] == ids) & (sim[ids, targets] >= threshold)
    return ids[keep], targets[keep], sim[ids[keep], targets[keep]]


def split_correspondences(n: int, key: str):
    if n < MIN_MATCHES:
        raise ValueError("insufficient matching support")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(stable_seed(key)))
    fit_n = n // 2
    gate_n = (n - fit_n) // 2
    return order[:fit_n], order[fit_n:fit_n + gate_n], order[fit_n + gate_n:]


def project_keys(x: torch.Tensor, pca: dict) -> torch.Tensor:
    # Normalizing projected rows removes norm changes that no rotation can fit.
    return F.normalize((x.float() - pca["mean"]) @ pca["basis"], dim=-1)


def audit_matched_pair(old: torch.Tensor, new: torch.Tensor, pca: dict, *, key: str,
                       regularization: float, permuted: bool = False) -> dict:
    oi, ni, confidence = matched_indices(old, new)
    result = {"matches": len(oi), "old_tokens": len(old), "new_tokens": len(new),
              "mean_match_cosine": float(confidence.mean()) if len(oi) else None,
              "permuted": permuted, "eligible": False, "accepted": False}
    if len(oi) < MIN_MATCHES:
        return {**result, "reason": "insufficient_matches"}
    x, y = project_keys(old[oi], pca), project_keys(new[ni], pca)
    if permuted:
        # A nonzero cyclic shift is an exact derangement with no unchanged pair.
        shift = 1 + stable_seed(key + "/permutation") % (len(y) - 1)
        y = y.roll(int(shift), dims=0)
    fit, gate, audit = split_correspondences(len(x), key)
    spectrum = torch.linalg.svdvals(x[fit])
    ratio = float(spectrum[-1] / spectrum[0])
    result.update(fit_count=len(fit), gate_count=len(gate), audit_count=len(audit),
                  fit_rank_ratio=ratio)
    if ratio < 1e-4:
        return {**result, "reason": "rank_deficient"}
    rotation = regularized_orthogonal_procrustes(x[fit], y[fit], regularization=regularization)
    measurements = {name: asdict(alignment_metrics(x[index], y[index], rotation))
                    for name, index in (("fit", fit), ("gate", gate), ("audit", audit))}
    accepted = measurements["gate"]["relative_rmse_improvement"] >= MIN_IMPROVEMENT
    result.update(eligible=True, accepted=accepted, reason="accepted" if accepted else "gate_below_15pct",
                  metrics=measurements,
                  orthogonality_error=float((rotation.T @ rotation - torch.eye(RANK)).abs().max()))
    return result


def aggregate_pairs(rows: list[dict]) -> dict:
    eligible = [x for x in rows if x["eligible"]]
    result = {"pairs": len(rows), "eligible": len(eligible),
              "eligible_fraction": len(eligible) / len(rows) if rows else None,
              "accepted": sum(x["accepted"] for x in rows),
              "accepted_fraction_all": sum(x["accepted"] for x in rows) / len(rows) if rows else None,
              "accepted_fraction_eligible": sum(x["accepted"] for x in eligible) / len(eligible) if eligible else None}
    for split in ("fit", "gate", "audit"):
        if not eligible:
            result[split] = None
            continue
        n = sum(x["metrics"][split]["match_count"] for x in eligible)
        def pooled(field):
            return math.sqrt(sum(x["metrics"][split][field] * x["metrics"][split]["match_count"] for x in eligible) / n)
        identity, transported = pooled("identity_mse"), pooled("transported_mse")
        gated = math.sqrt(sum(x["metrics"][split]["transported_mse" if x["accepted"] else "identity_mse"] * x["metrics"][split]["match_count"] for x in eligible) / n)
        result[split] = {"correspondences": n, "identity_rmse": identity,
                         "candidate_rmse": transported, "gated_rmse": gated,
                         "candidate_relative_improvement": (identity-transported)/max(identity,1e-12),
                         "gated_relative_improvement": (identity-gated)/max(identity,1e-12)}
    return result
