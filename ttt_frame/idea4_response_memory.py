"""Finite-response diagnostics for an Idea1 teacher, not full Spatial-TTT.

The state is a fixed set of function values. The kernel is an RBF in a frozen
random projection, with train-fitted bandwidth/gain. In the conditioned arm,
the bandwidth and gain depend on the RMS of the previous response state.
Eight simultaneous token writes are averaged; this is an explicit batch
approximation to the original card's sequential per-token writes.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def teacher_read(w_in: Tensor, w_out: Tensor, query: Tensor) -> Tensor:
    return F.silu(query @ w_in.transpose(-1, -2)) @ w_out.transpose(-1, -2)


@torch.no_grad()
def teacher_write(w_in: Tensor, w_out: Tensor, key: Tensor, target: Tensor,
                  active: Tensor, learning_rate: float = 0.05) -> None:
    pre = key @ w_in.transpose(-1, -2)
    hidden = F.silu(pre)
    derivative = 2 * (hidden @ w_out.transpose(-1, -2) - target) / (key.shape[-2] * target.shape[-1])
    derivative = derivative * active
    grad_out = derivative.transpose(-1, -2) @ hidden
    sig = torch.sigmoid(pre)
    grad_in = ((derivative @ w_out) * sig * (1 + pre * (1 - sig))).transpose(-1, -2) @ key
    w_in.sub_(learning_rate * grad_in)
    w_out.sub_(learning_rate * grad_out)


def initial_teacher(batch: int, horizons: int, dim: int, hidden: int, device: str,
                    seed: int = 0) -> tuple[Tensor, Tensor]:
    wi = torch.randn(horizons, hidden, dim, generator=torch.Generator().manual_seed(seed)) / math.sqrt(dim)
    return wi.to(device).unsqueeze(0).repeat(batch, 1, 1, 1), torch.zeros(batch, horizons, dim, hidden, device=device)


class ResponseMemory(nn.Module):
    """Positive-definite fixed/response-conditioned RBF with a ridge reader."""
    def __init__(self, probes: Tensor, projection: Tensor, *, conditioned: bool,
                 ridge: float = 0.05):
        super().__init__()
        if probes.ndim != 2 or projection.ndim != 2 or probes.shape[1] != projection.shape[0]:
            raise ValueError("incompatible probes/projection")
        if ridge <= 0 or not torch.isfinite(probes).all() or not torch.isfinite(projection).all():
            raise ValueError("finite values and positive ridge required")
        self.register_buffer("probes", probes.detach().clone())
        self.register_buffer("projection", projection.detach().clone())
        self.register_buffer("eye", torch.eye(len(probes), device=probes.device))
        self.ridge = float(ridge)
        self.conditioned = bool(conditioned)
        self.log_tau = nn.Parameter(torch.tensor(0., device=probes.device))
        self.logit_eta = nn.Parameter(torch.tensor(-4., device=probes.device))
        self.tau_state = nn.Parameter(torch.tensor(0., device=probes.device), requires_grad=conditioned)
        self.eta_state = nn.Parameter(torch.tensor(0., device=probes.device), requires_grad=conditioned)

    def embed(self, x: Tensor) -> Tensor:
        return F.normalize(x @ self.projection, dim=-1)

    def kernel(self, state: Tensor, x: Tensor, y: Tensor) -> Tensor:
        c = torch.log1p(state.square().mean(dim=(-2, -1)).sqrt()).unsqueeze(-1).unsqueeze(-1)
        tau = torch.exp((self.log_tau + self.tau_state * c).clamp(-4., 4.))
        xx, yy = self.embed(x), self.embed(y)
        distance = (xx.square().sum(-1, keepdim=True) + yy.square().sum(-1).unsqueeze(-2)
                    - 2 * xx @ yy.transpose(-1, -2)).clamp_min(0)
        return torch.exp(-distance / tau)

    def coefficients(self, state: Tensor) -> Tensor:
        gram = self.kernel(state, self.probes, self.probes) + self.ridge * self.eye
        chol = torch.linalg.cholesky(gram)
        return torch.cholesky_solve(state, chol)

    def read(self, state: Tensor, query: Tensor) -> Tensor:
        return self.kernel(state, query, self.probes) @ self.coefficients(state)

    def forward(self, state: Tensor, key: Tensor, target: Tensor, active: Tensor | None = None) -> Tensor:
        # Both the reader and propagation kernel use the pre-write state.
        residual = target - self.read(state, key)
        c = torch.log1p(state.square().mean(dim=(-2, -1)).sqrt()).unsqueeze(-1).unsqueeze(-1)
        gain = torch.sigmoid(self.logit_eta + self.eta_state * c)
        delta = gain * (self.kernel(state, self.probes, key) @ residual) / key.shape[-2]
        if active is not None:
            delta = delta * active
        return state + delta

    def config(self) -> dict:
        return {"probes": len(self.probes), "ridge": self.ridge, "conditioned": self.conditioned,
                "log_tau": float(self.log_tau.detach()), "logit_eta": float(self.logit_eta.detach()),
                "tau_state": float(self.tau_state.detach()), "eta_state": float(self.eta_state.detach())}


@torch.no_grad()
def select_probes(candidates: Tensor, count: int, projection: Tensor) -> Tensor:
    """Deterministic farthest-first selection from explicitly train-only keys."""
    if count > len(candidates) or count < 1:
        raise ValueError("invalid probe count")
    projected = F.normalize(candidates @ projection, dim=-1)
    indices, minimum = [], torch.full((len(candidates),), float("inf"), device=candidates.device)
    index = 0
    for _ in range(count):
        indices.append(index)
        distance = (projected - projected[index]).square().sum(-1)
        minimum = torch.minimum(minimum, distance)
        minimum[torch.tensor(indices, device=candidates.device)] = -1
        index = int(minimum.argmax())
    return candidates[indices].clone()
