"""Validated ragged channel pools and deterministic physical-panel sampling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .channels import Channels


@dataclass(frozen=True)
class ChannelBatch:
    direct: Tensor
    tx_to_irs: Tensor
    irs_to_user: Tensor
    user_mask: Tensor
    irs_locations: Tensor
    transmitter_locations: Tensor
    panel_rows: Tensor  # Globally unique physical rows, including across cities.

    @property
    def channels(self) -> Channels:
        return Channels(
            self.direct,
            torch.einsum("bkn,bnt->bktn", self.irs_to_user, self.tx_to_irs),
            self.user_mask,
        )

    def slice(self, start: int, end: int, device: torch.device | str = "cpu") -> ChannelBatch:
        values = [getattr(self, field.name)[start:end].to(device) for field in fields(self)]
        return ChannelBatch(*values)


@dataclass(frozen=True)
class ChannelPool:
    direct: Tensor
    tx_to_irs: Tensor
    irs_to_user: Tensor
    offsets: Tensor
    counts: Tensor
    irs_locations: Tensor
    user_locations: Tensor
    transmitter_locations: Tensor

    @property
    def panel_count(self) -> int:
        return int(self.counts.numel())

    @property
    def elements(self) -> int:
        return int(self.tx_to_irs.shape[1])

    @property
    def antennas(self) -> int:
        return int(self.direct.shape[1])

    def split(self) -> dict[str, Tensor]:
        """Preserve the current 80/10/10 split and seed, using physical rows."""
        rows = torch.randperm(self.panel_count, generator=torch.Generator().manual_seed(20260909))
        holdout = round(0.2 * self.panel_count)
        midpoint = holdout // 2
        if midpoint == 0 or holdout >= self.panel_count:
            raise ValueError(
                "The pool needs enough panels for nonempty train/validation/test splits"
            )
        return {
            "train": rows[holdout:],
            "validation": rows[:midpoint],
            "test": rows[midpoint:holdout],
        }

    def sample(
        self, rows: Tensor, size: int, users: int, antennas: int, generator: torch.Generator
    ) -> ChannelBatch:
        """Match the current conditional panel sampler and aligned augmentations."""
        if size <= 0 or users <= 0 or not 1 <= antennas <= self.antennas:
            raise ValueError("Invalid batch size, user count, or antenna count")
        candidate_counts = self.counts[rows]
        weights = torch.where(
            candidate_counts >= users,
            candidate_counts.double().reciprocal(),
            torch.zeros(rows.numel(), dtype=torch.float64),
        )
        if not bool(weights.sum() > 0):
            raise ValueError(f"No panels in this split support {users} users")
        panels = rows[torch.multinomial(weights, size, replacement=True, generator=generator)]
        counts = self.counts[panels]
        maximum = int(counts.max())
        scores = torch.rand((size, maximum), generator=generator)
        scores.masked_fill_(torch.arange(maximum)[None] >= counts[:, None], float("inf"))
        positions = torch.topk(scores, users, dim=1, largest=False, sorted=True).indices
        links = self.offsets[panels, None] + positions
        elements = torch.stack(
            [torch.randperm(self.elements, generator=generator) for _ in range(size)]
        )
        transmitters = torch.stack(
            [torch.randperm(self.antennas, generator=generator)[:antennas] for _ in range(size)]
        )
        batch_rows = torch.arange(size)[:, None]
        tx_to_irs = self.tx_to_irs[panels].gather(
            2, transmitters[:, None].expand(size, self.elements, antennas)
        )
        tx_to_irs = (
            tx_to_irs.transpose(1, 2)
            .gather(2, elements[:, None].expand(size, antennas, self.elements))
            .transpose(1, 2)
        )
        direct = self.direct[links].gather(2, transmitters[:, None].expand(size, users, antennas))
        irs_to_user = self.irs_to_user[links].gather(
            2, elements[:, None].expand(size, users, self.elements)
        )
        irs_locations = self.irs_locations[panels][batch_rows, elements]
        tx_locations = self.transmitter_locations[panels][batch_rows, transmitters]
        # Preserve the previous shared scene translation/scale; no unused
        # pilot-derived pair features are constructed for the new estimator.
        locations = torch.cat((irs_locations, self.user_locations[links], tx_locations), dim=1)
        centered = locations - locations.mean(1, keepdim=True)
        scale = torch.linalg.vector_norm(centered, dim=-1, keepdim=True).amax(1, keepdim=True)
        normalized = centered / scale.clamp_min(1e-12)
        return ChannelBatch(
            direct,
            tx_to_irs,
            irs_to_user,
            torch.ones((size, users), dtype=torch.bool),
            normalized[:, : self.elements],
            normalized[:, self.elements + users :],
            panels,
        )


def load_pool(path: Path) -> ChannelPool:
    """Read existing and newly generated NPZ pools without ray-tracing imports."""
    names = (
        "direct",
        "tx_to_irs",
        "irs_to_user",
        "panel_user_offsets",
        "panel_user_counts",
        "irs_element_locations_m",
        "user_locations_m",
        "transmitter_locations_m",
    )
    dtypes = (
        np.complex64,
        np.complex64,
        np.complex64,
        np.int64,
        np.int64,
        np.float32,
        np.float32,
        np.float32,
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = set(names) - set(archive.files)
        if missing:
            raise ValueError(f"Channel pool is missing {sorted(missing)}")
        values = [
            torch.from_numpy(np.asarray(archive[name], dtype=dtype))
            for name, dtype in zip(names, dtypes, strict=True)
        ]
    pool = ChannelPool(*values)
    if pool.counts.ndim != 1 or pool.panel_count == 0 or bool((pool.counts <= 0).any()):
        raise ValueError("Every pool panel must have at least one user")
    if pool.direct.ndim != 2 or pool.tx_to_irs.ndim != 3:
        raise ValueError("Expected direct [link,antenna] and TX-to-IRS [panel,element,antenna]")
    panels, elements, antennas = pool.panel_count, pool.elements, pool.antennas
    links = pool.direct.shape[0]
    expected = (
        (links, antennas),
        (panels, elements, antennas),
        (links, elements),
        (panels + 1,),
        (panels,),
        (panels, elements, 3),
        (links, 3),
        (panels, antennas, 3),
    )
    for name, tensor, shape in zip(names, values, expected, strict=True):
        if tensor.shape != shape or not bool(tensor.isfinite().all()):
            raise ValueError(f"Invalid {name}: expected finite values of shape {shape}")
    if elements <= 0 or antennas <= 0:
        raise ValueError("The pool must contain IRS elements and antennas")
    if int(pool.offsets[0]) != 0 or int(pool.offsets[-1]) != links:
        raise ValueError("Panel offsets must span all links")
    if not torch.equal(pool.offsets.diff(), pool.counts):
        raise ValueError("Panel offsets and counts disagree")
    return pool
