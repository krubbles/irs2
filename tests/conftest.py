from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from irs.io import save_npz


@pytest.fixture(autouse=True)
def deterministic_torch() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(12)
    panels, elements, antennas = 20, 16, 4
    counts = np.tile(np.arange(2, 6, dtype=np.int64), 5)
    links = int(counts.sum())

    def complex_values(shape: tuple[int, ...], scale: float) -> npt.NDArray[np.complex64]:
        return np.asarray(
            scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape)), dtype=np.complex64
        )

    offsets = np.concatenate((np.zeros(1, dtype=np.int64), counts.cumsum()))
    centers = rng.normal(size=(panels, 3)).astype(np.float32) * 10
    grid = np.asarray(
        [(0, x, y) for y in np.linspace(-0.31, 0.31, 4) for x in np.linspace(-0.31, 0.31, 4)],
        dtype=np.float32,
    )
    path = tmp_path / "channels.npz"
    save_npz(
        path,
        direct=complex_values((links, antennas), 1e-5),
        tx_to_irs=complex_values((panels, elements, antennas), 0.01),
        irs_to_user=complex_values((links, elements), 1e-4),
        panel_user_counts=counts,
        panel_user_offsets=offsets,
        irs_element_locations_m=centers[:, None] + grid[None],
        transmitter_locations_m=rng.normal(size=(panels, antennas, 3)).astype(np.float32),
        user_locations_m=rng.normal(size=(links, 3)).astype(np.float32),
        scene_names=np.asarray(["a", "b"]),
        panel_scene_indices=np.repeat(np.arange(2), 10),
        link_scene_indices=np.repeat(np.repeat(np.arange(2), 10), counts),
        panel_indices=np.tile(np.arange(10), 2),
        panel_names=np.asarray([f"p{i}" for i in range(panels)]),
        panel_building_names=np.asarray(["building"] * panels),
        panel_centers_m=centers,
        panel_normals=np.tile(np.array([1, 0, 0], dtype=np.float32), (panels, 1)),
        transmitter_site_indices=np.zeros(panels, dtype=np.int64),
        user_indices=np.arange(links),
        user_source_indices=np.arange(links),
        metadata_json=np.asarray(json.dumps({"format_version": 1})),
    )
    return path
