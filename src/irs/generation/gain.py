"""Resumable single-user gain caching and the current 0.1-bit/s/Hz link filter."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

from ..channels import Channels, channel_rate
from ..data import load_pool
from ..io import file_hash, save_npz, validate_manifest, write_json
from ..solver import SolverConfig, optimize_phases

IndexArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


PANEL_FIELDS: tuple[str, ...] = (
    "panel_scene_indices",
    "panel_indices",
    "panel_names",
    "panel_building_names",
    "panel_centers_m",
    "panel_normals",
    "transmitter_site_indices",
    "transmitter_locations_m",
    "irs_element_locations_m",
    "tx_to_irs",
)


LINK_FIELDS: tuple[str, ...] = (
    "link_scene_indices",
    "user_indices",
    "user_source_indices",
    "user_locations_m",
    "direct",
    "irs_to_user",
)


def cache_gains(
    dataset: Path,
    directory: Path,
    device: torch.device,
    solver: SolverConfig = SolverConfig(steps=250, restarts=8),
    batch_size: int = 128,
    seed: int = 20260827,
) -> Path:
    """Score every physical link once, with random 1..4 antenna subsets per batch.

    The original generator's validation-first work order and sampling seeds
    are retained. No old training configuration is needed to run this stage.
    """
    if batch_size <= 0:
        raise ValueError("Gain cache batch size must be positive")
    pool = load_pool(dataset)
    if pool.antennas != 4:
        raise ValueError("The current gain filter expects four physical TX antennas")
    configuration: dict[str, object] = {
        "dataset_sha256": file_hash(dataset),
        "solver": asdict(solver),
        "batch_size": batch_size,
        "seed": seed,
        "split_seed": 20260909,
        "sampling": "K=1; one random 1..4 antenna count per batch; uniform antenna subset per link",
    }
    validate_manifest(directory, configuration)
    rows = torch.randperm(pool.panel_count, generator=torch.Generator().manual_seed(20260909))
    holdout = max(1, round(pool.panel_count * 0.2))
    link_panels = torch.repeat_interleave(torch.arange(pool.panel_count), pool.counts)
    pieces: list[dict[str, npt.NDArray[np.generic]]] = []
    index = 0
    for code, panels in ((1, rows[:holdout]), (0, rows[holdout:])):
        links = torch.nonzero(torch.isin(link_panels, panels), as_tuple=False).squeeze(1)
        for start in range(0, links.numel(), batch_size):
            selected = links[start : start + batch_size]
            physical_rows = link_panels[selected]
            path = directory / "batches" / f"{index:06d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if not np.array_equal(saved["link_index"], selected.numpy()):
                        raise ValueError(f"Cached gain batch differs: {path}")
                    pieces.append({name: np.asarray(saved[name]) for name in saved.files})
            else:
                generator = torch.Generator().manual_seed(seed + 101 + index)
                antennas = int(torch.randint(1, 5, (1,), generator=generator))
                subset = torch.stack(
                    [
                        torch.randperm(4, generator=generator)[:antennas]
                        for _ in range(selected.numel())
                    ]
                )
                direct = pool.direct[selected].gather(1, subset)[:, None]
                tx = pool.tx_to_irs[physical_rows].gather(
                    2, subset[:, None].expand(-1, pool.elements, -1)
                )
                cascade = torch.einsum("bn,bnt->btn", pool.irs_to_user[selected], tx)[:, None]
                channels = Channels(
                    direct.to(device),
                    cascade.to(device),
                    torch.ones((selected.numel(), 1), dtype=torch.bool, device=device),
                )
                phases = optimize_phases(channels, replace(solver, seed=seed + 1_000_101 + index))
                with torch.no_grad():
                    identity = channel_rate(channels, torch.zeros_like(phases)).cpu()
                    reference = channel_rate(channels, phases).cpu()
                gains = reference - identity
                arrays: dict[str, npt.NDArray[np.generic]] = {
                    "link_index": selected.numpy(),
                    "identity_rate": identity.numpy(),
                    "full_csi_rate": reference.numpy(),
                    "absolute_gain": gains.numpy(),
                    "gain_percent": (100 * gains / identity.clamp_min(1e-12)).numpy(),
                    "transmitter_count": np.full(selected.numel(), antennas, dtype=np.int64),
                    "split_code": np.full(selected.numel(), code, dtype=np.uint8),
                }
                save_npz(path, **arrays)
                pieces.append(arrays)
            index += 1
            write_json(
                directory / "progress.json",
                {
                    "completed_batches": index,
                    "completed_links": sum(p["link_index"].size for p in pieces),
                    "total_links": int(pool.direct.shape[0]),
                },
            )
            print(f"Gain batch {index} saved/cached", flush=True)
    combined = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
    combined["metadata_json"] = np.asarray(json.dumps(configuration))
    output = directory / "samples.npz"
    save_npz(output, **combined)
    return output


def filter_by_gain(dataset: Path, gain_cache: Path, output: Path, threshold: float = 0.1) -> None:
    if output.resolve() == dataset.resolve():
        raise ValueError("Filtered output must differ from the source dataset")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum gain must be finite and nonnegative")
    load_pool(dataset)
    with np.load(dataset, allow_pickle=False) as source:
        source_arrays: dict[str, npt.NDArray[np.generic]] = {
            name: np.asarray(source[name]) for name in source.files
        }
    counts: IndexArray = np.asarray(source_arrays["panel_user_counts"], dtype=np.int64)
    offsets: IndexArray = np.asarray(source_arrays["panel_user_offsets"], dtype=np.int64)
    panel_count: int = int(counts.shape[0])
    link_count: int = int(offsets[-1])
    if offsets.shape != (panel_count + 1,):
        raise ValueError("source panel offsets have an unexpected shape")

    with np.load(gain_cache, allow_pickle=False) as cache:
        if "metadata_json" in cache:
            provenance = cast(dict[str, object], json.loads(str(cache["metadata_json"].item())))
            if provenance.get("dataset_sha256") != file_hash(dataset):
                raise ValueError("Gain cache belongs to a different source dataset")
        cache_links: IndexArray = np.asarray(cache["link_index"], dtype=np.int64)
        if cache_links.shape != (link_count,):
            raise ValueError("gain cache does not cover the source link count")
        if not np.array_equal(np.sort(cache_links), np.arange(link_count)):
            raise ValueError("gain cache link indices are not a complete permutation")
        for name in (
            "identity_rate",
            "full_csi_rate",
            "absolute_gain",
            "gain_percent",
            "transmitter_count",
            "split_code",
        ):
            if cache[name].shape != (link_count,) or not np.isfinite(cache[name]).all():
                raise ValueError(f"Invalid gain-cache field: {name}")
        identity_by_link: FloatArray = np.empty(link_count, dtype=np.float32)
        full_csi_by_link: FloatArray = np.empty(link_count, dtype=np.float32)
        gain_by_link: FloatArray = np.empty(link_count, dtype=np.float32)
        percent_by_link: FloatArray = np.empty(link_count, dtype=np.float32)
        transmitter_count_by_link: IndexArray = np.empty(link_count, dtype=np.int64)
        split_code_by_link: npt.NDArray[np.uint8] = np.empty(link_count, dtype=np.uint8)
        identity_by_link[cache_links] = np.asarray(cache["identity_rate"], dtype=np.float32)
        full_csi_by_link[cache_links] = np.asarray(cache["full_csi_rate"], dtype=np.float32)
        gain_by_link[cache_links] = np.asarray(cache["absolute_gain"], dtype=np.float32)
        percent_by_link[cache_links] = np.asarray(cache["gain_percent"], dtype=np.float32)
        transmitter_count_by_link[cache_links] = np.asarray(
            cache["transmitter_count"], dtype=np.int64
        )
        split_code_by_link[cache_links] = np.asarray(cache["split_code"], dtype=np.uint8)

    retained_link_mask: BoolArray = np.asarray(
        gain_by_link >= np.float32(threshold), dtype=np.bool_
    )
    cumulative: IndexArray = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(retained_link_mask, dtype=np.int64))
    )
    retained_counts_all: IndexArray = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
    retained_panel_mask: BoolArray = retained_counts_all > 0
    retained_panel_rows: IndexArray = np.flatnonzero(retained_panel_mask).astype(np.int64)
    retained_link_indices: IndexArray = np.flatnonzero(retained_link_mask).astype(np.int64)
    retained_counts: IndexArray = retained_counts_all[retained_panel_mask]
    if retained_link_indices.size == 0:
        raise ValueError("No links meet the gain threshold")
    retained_offsets: IndexArray = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(retained_counts, dtype=np.int64))
    )

    output_arrays: dict[str, npt.NDArray[np.generic]] = {
        "scene_names": source_arrays["scene_names"],
        "panel_user_counts": retained_counts,
        "panel_user_offsets": retained_offsets,
        "original_panel_rows": retained_panel_rows,
        "original_link_indices": retained_link_indices,
    }
    for name in PANEL_FIELDS:
        output_arrays[name] = source_arrays[name][retained_panel_rows]
    for name in LINK_FIELDS:
        output_arrays[name] = source_arrays[name][retained_link_indices]
    output_arrays.update(
        {
            "full_csi_identity_rate": identity_by_link[retained_link_indices],
            "full_csi_rate": full_csi_by_link[retained_link_indices],
            "full_csi_absolute_gain": gain_by_link[retained_link_indices],
            "full_csi_gain_percent": percent_by_link[retained_link_indices],
            "full_csi_transmitter_count": transmitter_count_by_link[retained_link_indices],
            "full_csi_split_code": split_code_by_link[retained_link_indices],
        }
    )
    raw_metadata: object = json.loads(str(source_arrays["metadata_json"].item()))
    metadata: dict[str, object] = (
        cast(dict[str, object], raw_metadata)
        if isinstance(raw_metadata, dict)
        else {"source_metadata": raw_metadata}
    )
    metadata["full_csi_gain_filter"] = {
        "minimum_absolute_gain_bit_s_hz_per_user": threshold,
        "comparison": "greater than or equal",
        "source_dataset": str(dataset.resolve()),
        "source_dataset_sha256": file_hash(dataset),
        "gain_cache": str(gain_cache.resolve()),
        "gain_cache_sha256": file_hash(gain_cache),
        "source_panel_count": panel_count,
        "retained_panel_count": int(retained_panel_rows.shape[0]),
        "source_link_count": link_count,
        "retained_link_count": int(retained_link_indices.shape[0]),
        "removed_link_count": int(link_count - retained_link_indices.shape[0]),
        "note": (
            "Gain was measured in the exhaustive K=1 cache using its cached "
            "random 1..4 transmitter count and subset for each panel-user link."
        ),
    }
    output_arrays["metadata_json"] = np.asarray(json.dumps(metadata, indent=2, sort_keys=True))
    save_npz(output, **output_arrays)

    # Reload and verify the serialized ragged structure rather than only memory.
    with np.load(output, allow_pickle=False) as saved:
        saved_counts: IndexArray = np.asarray(saved["panel_user_counts"], dtype=np.int64)
        saved_offsets: IndexArray = np.asarray(saved["panel_user_offsets"], dtype=np.int64)
        saved_gains: FloatArray = np.asarray(saved["full_csi_absolute_gain"], dtype=np.float32)
        if not np.array_equal(saved_offsets[1:] - saved_offsets[:-1], saved_counts):
            raise RuntimeError("saved counts and offsets disagree")
        if int(saved_offsets[-1]) != int(saved_gains.shape[0]):
            raise RuntimeError("saved offsets do not cover all retained links")
        if np.any(saved_gains < np.float32(threshold)):
            raise RuntimeError("saved dataset contains a below-threshold gain")
        if any(int(value) <= 0 for value in saved_counts):
            raise RuntimeError("saved dataset contains an empty panel")

    summary: dict[str, object] = {
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "minimum_gain_bit_s_hz_per_user": threshold,
        "source_panel_count": panel_count,
        "retained_panel_count": int(retained_panel_rows.shape[0]),
        "removed_panel_count": int(panel_count - retained_panel_rows.shape[0]),
        "source_link_count": link_count,
        "retained_link_count": int(retained_link_indices.shape[0]),
        "removed_link_count": int(link_count - retained_link_indices.shape[0]),
        "retained_link_fraction": float(retained_link_indices.shape[0] / link_count),
        "minimum_retained_gain": float(gain_by_link[retained_link_indices].min()),
        "maximum_retained_gain": float(gain_by_link[retained_link_indices].max()),
    }
    write_json(output.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
