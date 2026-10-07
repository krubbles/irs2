"""The shared channel contract, pilot physics, and coordinated MIMO rate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Existing physical budget: 1 W total downlink, 20 MHz, 290 K, 7 dB noise figure.
NOISE_POWER_W: float = 1.380649e-23 * 20e6 * 290.0 * 10.0**0.7


@dataclass(frozen=True)
class Channels:
    direct: Tensor  # complex64 [batch, user, antenna]
    cascade: Tensor  # complex64 [batch, user, antenna, element]
    mask: Tensor  # bool [batch, user]

    def slice(self, start: int, end: int, device: torch.device | str = "cpu") -> Channels:
        return Channels(
            self.direct[start:end].to(device),
            self.cascade[start:end].to(device),
            self.mask[start:end].to(device),
        )

    def effective(self, phases: Tensor) -> Tensor:
        reflection = torch.polar(torch.ones_like(phases), phases)
        return (self.direct + torch.einsum("bktn,bn->bkt", self.cascade, reflection)) * self.mask[
            ..., None
        ]


def precoder(channel: Tensor, mask: Tensor) -> Tensor:
    """Equal-power RZF, using the small antenna-space solve."""
    counts = mask.sum(-1).clamp_min(1)
    regularization = NOISE_POWER_W * counts + 1e-4 * channel.abs().square().sum((1, 2)) / counts
    hermitian = channel.conj().transpose(-1, -2)
    identity = torch.eye(channel.shape[-1], dtype=channel.dtype, device=channel.device)
    weights: Tensor = torch.linalg.solve(
        hermitian @ channel + regularization[:, None, None] * identity, hermitian
    )
    normalized: Tensor = weights / torch.linalg.vector_norm(weights, dim=1, keepdim=True).clamp_min(
        1e-30
    )
    return normalized


def channel_rate(
    channels: Channels, phases: Tensor, *, estimated: Channels | None = None
) -> Tensor:
    """Mean per-user bit/s/Hz; optionally form the precoder from estimated CSI."""
    actual = channels.effective(phases)
    observed = actual if estimated is None else estimated.effective(phases)
    counts = channels.mask.sum(-1).clamp_min(1)
    powers = (actual @ precoder(observed, channels.mask)).abs().square() / counts[:, None, None]
    desired = powers.diagonal(dim1=1, dim2=2)
    interference = (powers.sum(-1) - desired).clamp_min(0)
    return (torch.log2(1 + desired / (interference + NOISE_POWER_W)) * channels.mask).sum(
        -1
    ) / counts


def received_pilots(
    direct: Tensor, tx_to_irs: Tensor, irs_to_user: Tensor, symbols: Tensor, probes: Tensor
) -> Tensor:
    reflected = torch.einsum("pn,bkn,bnt->bktp", probes, irs_to_user, tx_to_irs)
    return symbols.view(1, 1, 1, -1) * (direct.unsqueeze(-1) + reflected)
