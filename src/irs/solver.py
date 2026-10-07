"""Multi-start phase optimization using only the channels supplied by the caller."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, nn

from .channels import Channels, channel_rate


@dataclass(frozen=True)
class SolverConfig:
    steps: int = 100
    restarts: int = 3
    learning_rate: float = 0.05
    seed: int = 20261006

    def __post_init__(self) -> None:
        if min(self.steps, self.restarts) <= 0:
            raise ValueError("Solver steps and restarts must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Solver learning rate must be finite and positive")


@torch.enable_grad()
def optimize_phases(channels: Channels, config: SolverConfig = SolverConfig()) -> Tensor:
    """Track the best iterate per restart, including an initial zero-phase restart.

    Inputs are detached. No ground-truth channel or training gradient enters
    this optimizer. It can also provide a full-CSI reference when explicitly
    passed physical channels.
    """
    batch, _, _, elements = channels.cascade.shape
    if batch == 0 or not bool(channels.mask.any(-1).all()):
        raise ValueError("Every solver scene must contain at least one active user")
    if not bool(channels.direct.isfinite().all() and channels.cascade.isfinite().all()):
        raise ValueError("Channels must be finite")
    expanded = Channels(
        channels.direct.detach().repeat_interleave(config.restarts, 0),
        channels.cascade.detach().repeat_interleave(config.restarts, 0),
        channels.mask.repeat_interleave(config.restarts, 0),
    )
    generator = torch.Generator(device=channels.direct.device).manual_seed(config.seed)
    values = (
        2
        * torch.pi
        * torch.rand(
            (batch, config.restarts, elements), generator=generator, device=channels.direct.device
        )
    )
    values[:, 0] = 0
    phases = nn.Parameter(values.flatten(0, 1))
    optimizer = torch.optim.Adam([phases], lr=config.learning_rate)
    with torch.no_grad():
        best = channel_rate(expanded, phases)
        best_phases = phases.detach().clone()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        (-channel_rate(expanded, phases).mean()).backward()
        optimizer.step()
        with torch.no_grad():
            rates = channel_rate(expanded, phases)
            if not bool(rates.isfinite().all()):
                raise RuntimeError("Phase optimization produced non-finite rates")
            improved = rates > best
            best_phases[improved] = phases[improved]
            best = torch.maximum(best, rates)
    winner = best.reshape(batch, config.restarts).argmax(-1)
    return best_phases.reshape(batch, config.restarts, elements)[
        torch.arange(batch, device=winner.device), winner
    ]
