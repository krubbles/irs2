"""Learnable pilots, ridge/learned lifting, convolution, attention and CSI skip fusion.

Preserves the newest estimator's state-dict names for existing checkpoints.
Adapted from Liu et al. (arXiv:2210.12447) and Mashhadi & Gunduz
(arXiv:2006.11796); only observations, the sensing bank and geometry enter inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .channels import Channels, received_pilots
from .data import ChannelBatch

Lift = Literal["learned", "ridge"]


@dataclass(frozen=True)
class ChannelEstimatorConfig:
    pilots: int = 17
    elements: int = 16
    width: int = 32
    blocks: int = 4
    heads: int = 4
    seed: int = 20261006
    lift: Lift = "learned"
    initial_ridge: float = 1e-3

    def __post_init__(self) -> None:
        if min(self.pilots, self.elements, self.width, self.blocks, self.heads) <= 0:
            raise ValueError("dimensions must be positive")
        if self.width % self.heads or not 0 < self.initial_ridge < float("inf"):
            raise ValueError("invalid attention width or ridge")
        if self.lift not in ("learned", "ridge"):
            raise ValueError("lift must be learned or ridge")


def canonical_order(locations: Tensor) -> Tensor:
    """Lexicographic physical order, invariant to input-array permutation.

    Prioritize axes with the largest spread; quantization keeps roundoff in
    nominally coplanar coordinates from splitting a physical grid row.
    Convolution on this flattened order approximates spatial locality.
    """
    b, n, _ = locations.shape
    centered: Tensor = locations - locations.mean(1, keepdim=True)
    axes: Tensor = torch.argsort(centered.square().mean(1), dim=-1, descending=True, stable=True)
    coordinates: Tensor = centered.gather(2, axes[:, None].expand(b, n, 3))
    scale: Tensor = centered.abs().amax((1, 2), keepdim=True).clamp_min(1e-12)
    coordinates = (coordinates / scale * 10000).round()
    order: Tensor = torch.arange(n, device=locations.device)[None].expand(b, n)
    for axis in (2, 1, 0):
        permutation: Tensor = torch.argsort(
            coordinates[..., axis].gather(1, order), dim=-1, stable=True
        )
        order = order.gather(1, permutation)
    return order


def reorder(values: Tensor, order: Tensor, axis: int) -> Tensor:
    """Gather a non-batch axis using an independent order per batch row."""
    axis %= values.ndim
    shape: list[int] = [values.shape[0]] + [1] * (values.ndim - 1)
    shape[axis] = order.shape[1]
    target: list[int] = list(values.shape)
    target[axis] = order.shape[1]
    return values.gather(axis, order.reshape(shape).expand(target))


class AttentionConvBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.conv = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        b, w, t, n = image.shape
        tokens: Tensor = image.flatten(2).transpose(1, 2)
        normalized: Tensor = self.norm(tokens)
        update: Tensor = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        mixed: Tensor = (tokens + update).transpose(1, 2).reshape(b, w, t, n)
        return image + torch.relu(self.conv(mixed))


class NeuralChannelEstimator(nn.Module):
    initial_symbol_phases: Tensor
    initial_probe_phases: Tensor

    def __init__(self, config: ChannelEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        generator: torch.Generator = torch.Generator().manual_seed(config.seed)
        symbol_angles: Tensor = 2 * torch.pi * torch.rand(config.pilots, generator=generator)
        probe_angles: Tensor = (
            2 * torch.pi * torch.rand(config.pilots, config.elements, generator=generator)
        )
        self.symbol_phases = nn.Parameter(symbol_angles)
        self.probe_phases = nn.Parameter(probe_angles)
        self.register_buffer("initial_symbol_phases", symbol_angles.clone())
        self.register_buffer("initial_probe_phases", probe_angles.clone())
        initial: Tensor = self.ridge_matrix(config.initial_ridge).detach()
        self.lift_real = nn.Parameter(initial.real.clone(), requires_grad=config.lift == "learned")
        self.lift_imag = nn.Parameter(initial.imag.clone(), requires_grad=config.lift == "learned")
        # Complex CSI, noise level, geometric IRS/TX coordinates, and power.
        self.input = nn.Conv2d(10, config.width, 3, padding=1)
        self.blocks = nn.ModuleList(
            AttentionConvBlock(config.width, config.heads) for _ in range(config.blocks)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * config.width, 2 * config.width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * config.width, 2, 3, padding=1),
        )
        output: nn.Conv2d = cast(nn.Conv2d, self.fusion[-1])
        nn.init.zeros_(output.weight)
        assert output.bias is not None
        nn.init.zeros_(output.bias)

    def symbols(self) -> Tensor:
        return torch.polar(torch.ones_like(self.symbol_phases), self.symbol_phases)

    def probes(self) -> Tensor:
        return torch.polar(torch.ones_like(self.probe_phases), self.probe_phases)

    def ridge_matrix(self, ridge: float) -> Tensor:
        patterns: Tensor = self.probes()
        # A short pilot bank leaves a large nullspace. Solve this tiny system
        # in double precision so array relabeling does not amplify FP32 error.
        precise: Tensor = patterns.to(torch.complex128)
        centered: Tensor = precise - precise.mean(0, keepdim=True)
        hermitian: Tensor = centered.conj().T
        identity: Tensor = torch.eye(
            self.config.elements, dtype=precise.dtype, device=patterns.device
        )
        matrix: Tensor = torch.linalg.solve(hermitian @ centered + ridge * identity, hermitian).to(
            patterns.dtype
        )
        return matrix

    def observe(
        self,
        examples: ChannelBatch,
        noise: float,
        generator: torch.Generator,
    ) -> Tensor:
        symbols: Tensor = self.symbols()
        probes: Tensor = self.probes()
        fixed_symbols: Tensor = torch.polar(
            torch.ones_like(self.initial_symbol_phases), self.initial_symbol_phases
        )
        fixed_probes: Tensor = torch.polar(
            torch.ones_like(self.initial_probe_phases), self.initial_probe_phases
        )
        received: Tensor = received_pilots(
            examples.direct, examples.tx_to_irs, examples.irs_to_user, symbols, probes
        )
        if noise:
            # Fixed initial-bank power prevents optimizing noise amplitude.
            with torch.no_grad():
                fixed: Tensor = received_pilots(
                    examples.direct,
                    examples.tx_to_irs,
                    examples.irs_to_user,
                    fixed_symbols,
                    fixed_probes,
                )
                deviation: Tensor = fixed.abs().square().mean(-1, keepdim=True).clamp_min(
                    1e-30
                ).sqrt() * (noise / 2**0.5)
            received = received + deviation * torch.complex(
                torch.randn(received.shape, device=received.device, generator=generator),
                torch.randn(received.shape, device=received.device, generator=generator),
            )
        return received

    def linear(self, received: Tensor, mask: Tensor, ridge: float | None = None) -> Channels:
        symbols: Tensor = self.symbols()
        patterns: Tensor = self.probes()
        demodulated: Tensor = received * symbols.conj()
        delta: Tensor = demodulated - demodulated.mean(-1, keepdim=True)
        matrix: Tensor = torch.complex(self.lift_real, self.lift_imag)
        if ridge is not None or self.config.lift == "ridge":
            matrix = self.ridge_matrix(self.config.initial_ridge if ridge is None else ridge)
        cascade: Tensor = delta @ matrix.T
        direct: Tensor = demodulated.mean(-1) - (cascade * patterns.mean(0)).sum(-1)
        return Channels(direct * mask[..., None], cascade * mask[..., None, None], mask)

    def forward(
        self,
        received: Tensor,
        mask: Tensor,
        irs_locations: Tensor,
        transmitter_locations: Tensor,
        noise: float = 0.0,
    ) -> Channels:
        coarse: Channels = self.linear(received, mask)
        b, k, t, n = coarse.cascade.shape
        element_order: Tensor = canonical_order(irs_locations)
        antenna_order: Tensor = canonical_order(transmitter_locations)
        cascade: Tensor = reorder(reorder(coarse.cascade, element_order, 3), antenna_order, 2)
        rms: Tensor = (cascade.abs().square().mean((2, 3), keepdim=True)).clamp_min(1e-30).sqrt()
        anchor_indices: Tensor = cascade.abs().flatten(2).argmax(-1, keepdim=True)
        anchor: Tensor = cascade.flatten(2).gather(2, anchor_indices)[..., None]
        frame: Tensor = torch.where(
            anchor.abs() > 0, anchor.conj() / anchor.abs().clamp_min(1e-30), torch.ones_like(anchor)
        )
        normalized: Tensor = cascade / rms * frame
        pilot_power: Tensor = received.abs().square().mean((2, 3), keepdim=True)
        # Known nominal noise and observed power; no true channel at inference.
        noise_ratio: Tensor = (pilot_power * noise**2 / rms.square().clamp_min(1e-30)).clamp_max(
            1e8
        )
        noise_feature: Tensor = noise_ratio.log1p().expand(b, k, t, n) / 10
        power_feature: Tensor = (rms.log() / 10).clamp(-6, 2).expand(b, k, t, n)
        irs: Tensor = reorder(irs_locations, element_order, 1)
        irs = irs - irs.mean(1, keepdim=True)
        irs = irs / irs.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        tx: Tensor = reorder(transmitter_locations, antenna_order, 1)
        tx = tx - tx.mean(1, keepdim=True)
        tx = tx / tx.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        image: Tensor = torch.cat(
            (
                normalized.real[..., None],
                normalized.imag[..., None],
                noise_feature[..., None],
                power_feature[..., None],
                irs[:, None, None].expand(b, k, t, n, 3),
                tx[:, None, :, None].expand(b, k, t, n, 3),
            ),
            -1,
        )
        image = image.reshape(b * k, t, n, 10).permute(0, 3, 1, 2)
        first: Tensor = torch.relu(self.input(image))
        hidden: Tensor = first
        for block in self.blocks:
            hidden = block(hidden)
        output: Tensor = (
            self.fusion(torch.cat((first, hidden), 1)).permute(0, 2, 3, 1).reshape(b, k, t, n, 2)
        )
        correction: Tensor = torch.complex(output[..., 0], output[..., 1]) * rms * frame.conj()
        refined: Tensor = cascade + correction
        refined = reorder(
            reorder(refined, antenna_order.argsort(-1), 2), element_order.argsort(-1), 3
        )
        demodulated: Tensor = received * self.symbols().conj()
        direct: Tensor = demodulated.mean(-1) - (refined * self.probes().mean(0)).sum(-1)
        return Channels(direct * mask[..., None], refined * mask[..., None, None], mask)


def config_from_values(values: dict[str, object]) -> ChannelEstimatorConfig:
    return ChannelEstimatorConfig(
        pilots=cast(int, values["pilots"]),
        elements=cast(int, values["elements"]),
        width=cast(int, values["width"]),
        blocks=cast(int, values["blocks"]),
        heads=cast(int, values["heads"]),
        seed=cast(int, values["seed"]),
        lift=cast(Lift, values["lift"]),
        initial_ridge=cast(float, values["initial_ridge"]),
    )


def load_estimator(path: Path, device: torch.device | str = "cpu") -> NeuralChannelEstimator:
    """Load an existing neural-estimator checkpoint or a new v2 checkpoint."""
    checkpoint = cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))
    config = config_from_values(cast(dict[str, object], checkpoint["config"]))
    model = NeuralChannelEstimator(config).to(device)
    model.load_state_dict(cast(dict[str, Tensor], checkpoint["model"]))
    model.eval()
    return model
