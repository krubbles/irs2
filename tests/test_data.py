from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from irs.data import load_pool
from irs.generation.gain import cache_gains, filter_by_gain
from irs.io import file_hash, save_npz, validate_manifest
from irs.solver import SolverConfig


def test_pool_split_and_sampling_are_aligned(dataset: Path) -> None:
    pool = load_pool(dataset)
    split = pool.split()
    all_rows = torch.cat(list(split.values()))
    assert sorted(all_rows.tolist()) == list(range(pool.panel_count))
    assert [rows.numel() for rows in split.values()] == [16, 2, 2]
    batch = pool.sample(split["train"], 8, 2, 3, torch.Generator().manual_seed(9))
    for i, row in enumerate(batch.panel_rows):
        start, end = int(pool.offsets[row]), int(pool.offsets[row + 1])
        for user in batch.irs_to_user[i]:
            # Element permutations preserve each physical link's multiset.
            values = user.abs().sort().values
            possible = pool.irs_to_user[start:end].abs().sort(-1).values
            assert bool(torch.isclose(possible, values[None]).all(-1).any())
    assert bool(torch.isin(batch.panel_rows, split["train"]).all())
    with pytest.raises(ValueError, match="No panels"):
        pool.sample(split["train"], 2, 100, 2, torch.Generator())


def test_reject_inconsistent_serialized_offsets(dataset: Path, tmp_path: Path) -> None:
    with np.load(dataset, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["panel_user_offsets"][2] += 1
    invalid = tmp_path / "invalid.npz"
    save_npz(invalid, **arrays)
    with pytest.raises(ValueError, match="offsets and counts"):
        load_pool(invalid)


def test_gain_filter_handles_shuffled_cache_links_and_retains_ragged_structure(
    dataset: Path, tmp_path: Path
) -> None:
    pool = load_pool(dataset)
    size = pool.direct.shape[0]
    order = np.arange(size)[::-1]
    gains = np.arange(size, dtype=np.float32) / size
    cache = tmp_path / "gains.npz"
    save_npz(
        cache,
        link_index=order,
        identity_rate=np.ones(size, dtype=np.float32),
        full_csi_rate=1 + gains[order],
        absolute_gain=gains[order],
        gain_percent=100 * gains[order],
        transmitter_count=np.ones(size, dtype=np.int64),
        split_code=np.zeros(size, dtype=np.uint8),
    )
    output = tmp_path / "filtered.npz"
    source_hash = file_hash(dataset)
    filter_by_gain(dataset, cache, output, threshold=0.5)
    filtered = load_pool(output)
    assert filtered.direct.shape[0] == int((gains >= 0.5).sum())
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["original_link_indices"], np.flatnonzero(gains >= 0.5))
        assert np.all(saved["full_csi_absolute_gain"] >= 0.5)
    assert file_hash(dataset) == source_hash
    with pytest.raises(ValueError, match="No links"):
        filter_by_gain(dataset, cache, tmp_path / "empty.npz", threshold=2)


def test_gain_cache_resumes_and_rejects_changed_settings(dataset: Path, tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    config = SolverConfig(steps=2, restarts=1)
    path = cache_gains(dataset, directory, torch.device("cpu"), config, batch_size=32)
    first = file_hash(directory / "batches/000000.npz")
    cache_gains(dataset, directory, torch.device("cpu"), config, batch_size=32)
    assert file_hash(directory / "batches/000000.npz") == first
    with np.load(path, allow_pickle=False) as arrays:
        assert sorted(arrays["link_index"].tolist()) == list(
            range(load_pool(dataset).direct.shape[0])
        )
        assert np.all(arrays["absolute_gain"] >= 0)
    with pytest.raises(ValueError, match="manifest differs"):
        cache_gains(dataset, directory, torch.device("cpu"), config, batch_size=16)


def test_manifest_normalizes_json_sequences(tmp_path: Path) -> None:
    validate_manifest(tmp_path, {"shape": (4, 4)})
    validate_manifest(tmp_path, {"shape": (4, 4)})
    with pytest.raises(ValueError, match="manifest differs"):
        validate_manifest(tmp_path, {"shape": (8, 8)})
