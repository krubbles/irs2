"""Resume exact complex channel tracing, then combine ragged per-city pools."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter, time
from typing import cast

import deepmimo as dm
import numpy as np
import numpy.typing as npt

from ..io import file_hash, read_json, save_npy, validate_manifest, write_json
from .cell import TangCellModel
from .config import (
    IRS_CELL_SIZE_M,
    IRS_COLUMNS,
    IRS_PATTERN_EXPONENT,
    IRS_REFLECTION_AMPLITUDE,
    IRS_ROWS,
    MIMO_ANTENNA_COORDINATES,
    RAY_TRACING_SEED,
    MaterializationSettings,
)
from .geometry import irs_element_positions, transmitter_element_positions
from .scene import create_sionna_full_scene_xml
from .tracing import solve_complex_links

FloatArray = npt.NDArray[np.float32]
ComplexArray = npt.NDArray[np.complex64]
IndexArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
NameArray = npt.NDArray[np.str_]


def _write_progress(
    checkpoint_directory: Path,
    *,
    scenario_name: str,
    stage: str,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    write_json(
        checkpoint_directory / "progress.json",
        {
            "scenario": scenario_name,
            "stage": stage,
            "completed": completed,
            "total": total,
            "fraction": 1.0 if total == 0 else completed / total,
            "elapsed_seconds": perf_counter() - started_at,
            "updated_at_unix": time(),
        },
    )


def _load_scene_geometry(scenario_name: str) -> dm.MacroDataset:
    loaded: dm.Dataset | dm.MacroDataset = dm.load(
        scenario_name,
        max_paths=1,
        tx_sets={0: "all"},
        rx_sets={1: [0]},
        matrices=["tx_pos"],
    )
    if not isinstance(loaded, dm.MacroDataset):
        raise TypeError(f"Expected a MacroDataset for {scenario_name}.")
    return loaded


def _load_or_trace_direct_channels(
    *,
    checkpoint_directory: Path,
    scene_xml: str,
    frequency_hz: float,
    transmitter_positions: FloatArray,
    user_positions: FloatArray,
    settings: MaterializationSettings,
    cell_model: TangCellModel,
) -> ComplexArray:
    cache_path: Path = checkpoint_directory / "direct_channels.npy"
    site_count: int = transmitter_positions.shape[0]
    antenna_count: int = transmitter_positions.shape[1]
    expected_shape: tuple[int, int, int] = (
        user_positions.shape[0],
        site_count,
        antenna_count,
    )
    if cache_path.exists():
        cached: ComplexArray = np.asarray(np.load(cache_path), dtype=np.complex64)
        if cached.shape != expected_shape:
            raise ValueError(f"Invalid direct-channel cache {cache_path}.")
        return cached
    source_positions: FloatArray = transmitter_positions.reshape(-1, 3)
    source_names: NameArray = np.asarray(
        [
            f"direct_site_{site}_antenna_{antenna}"
            for site in range(site_count)
            for antenna in range(antenna_count)
        ],
        dtype=np.str_,
    )
    target_names: NameArray = np.asarray(
        [f"direct_user_{index}" for index in range(user_positions.shape[0])],
        dtype=np.str_,
    )
    result: ComplexArray = solve_complex_links(
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        source_names=source_names,
        source_positions=source_positions,
        target_names=target_names,
        target_positions=user_positions,
        samples_per_source=settings.samples_per_source,
        cell_model=cell_model,
    ).reshape(user_positions.shape[0], site_count, antenna_count)
    save_npy(cache_path, result)
    return result


def _materialize_scene(
    scenario_name: str,
    *,
    filter_path: Path,
    output_root: Path,
    settings: MaterializationSettings,
) -> dict[str, object]:
    output_path: Path = output_root / "scenes" / f"{scenario_name}.npz"
    summary_path: Path = output_path.with_suffix(".json")
    if output_path.exists() and summary_path.exists():
        summary = read_json(summary_path)
        if (
            summary.get("settings") != asdict(settings)
            or summary.get("source_filter_sha256") != sha256(filter_path.read_bytes()).hexdigest()
        ):
            raise ValueError(
                f"Completed materialization settings or source differ for {scenario_name}"
            )
        print(f"MATERIALIZED_SCENE_CACHED={scenario_name}")
        return summary
    started_at: float = perf_counter()
    with np.load(filter_path, allow_pickle=False) as filtered:
        all_qualifying: BoolArray = np.asarray(filtered["qualifying_mask"], dtype=np.bool_)
        retained_rows: IndexArray = np.asarray(
            np.flatnonzero(all_qualifying.any(axis=1)),
            dtype=np.int64,
        )
        panel_indices: IndexArray = np.asarray(
            filtered["panel_indices"][retained_rows], dtype=np.int64
        )
        panel_names: NameArray = np.asarray(filtered["panel_names"][retained_rows], dtype=np.str_)
        building_names: NameArray = np.asarray(
            filtered["panel_building_names"][retained_rows], dtype=np.str_
        )
        panel_centers: FloatArray = np.asarray(
            filtered["panel_centers_m"][retained_rows], dtype=np.float32
        )
        panel_normals: FloatArray = np.asarray(
            filtered["panel_normals"][retained_rows], dtype=np.float32
        )
        best_sites: IndexArray = np.asarray(
            filtered["best_panel_transmitter_index"][retained_rows], dtype=np.int64
        )
        qualifying: BoolArray = np.asarray(all_qualifying[retained_rows], dtype=np.bool_)
        user_positions: FloatArray = np.asarray(filtered["user_positions_m"], dtype=np.float32)
        user_source_indices: IndexArray = np.asarray(
            filtered["user_source_indices"], dtype=np.int64
        )
        transmitter_centers: FloatArray = np.asarray(
            filtered["transmitter_positions_m"], dtype=np.float32
        )
    panel_count: int = panel_indices.shape[0]
    user_counts: IndexArray = np.asarray(qualifying.sum(axis=1, dtype=np.int64), dtype=np.int64)
    offsets: IndexArray = np.zeros(panel_count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(user_counts, dtype=np.int64)
    link_count: int = int(offsets[-1])
    scenario: dm.MacroDataset = _load_scene_geometry(scenario_name)
    frequency_hz: float = float(scenario[0].rt_params.frequency)
    wavelength_m: float = 299_792_458.0 / frequency_hz
    metadata: dict[str, object] = {
        "format": "IRS2 multi-city filtered tiled-panel optimizer channel pool",
        "format_version": 1,
        "scenario": scenario_name,
        "source_filter_dataset": str(filter_path.resolve()),
        "source_filter_sha256": sha256(filter_path.read_bytes()).hexdigest(),
        "panel_count": panel_count,
        "link_count": link_count,
        "settings": asdict(settings),
        "ray_tracing_seed": RAY_TRACING_SEED,
        "irs_shape": [IRS_ROWS, IRS_COLUMNS],
        "mimo_shape": [2, 2],
        "mimo_antenna_coordinates": MIMO_ANTENNA_COORDINATES,
        "user_pool": "all qualifying users per panel; ragged via panel_user_offsets",
        "channel_aggregation": "coherent complex path sum",
        "direct_channels": "ray traced at the same four physical TX elements",
        "checkpointing": "every 32 TX-to-IRS panels and every 8 IRS-to-user panels",
    }
    checkpoint_directory: Path = output_root / "checkpoints" / scenario_name
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    validate_manifest(checkpoint_directory, metadata)
    _write_progress(
        checkpoint_directory,
        scenario_name=scenario_name,
        stage="planned",
        completed=0,
        total=panel_count,
        started_at=started_at,
    )
    mesh_directory: Path = output_root / "meshes" / scenario_name
    scene_xml: str = create_sionna_full_scene_xml(
        scenario.scene, output_directory=mesh_directory.resolve()
    )
    cell_model = TangCellModel(
        width_m=IRS_CELL_SIZE_M,
        height_m=IRS_CELL_SIZE_M,
        reflection_amplitude=IRS_REFLECTION_AMPLITUDE,
        pattern_exponent=IRS_PATTERN_EXPONENT,
    )
    element_positions: FloatArray = irs_element_positions(panel_centers, panel_normals)
    transmitter_positions: FloatArray = transmitter_element_positions(transmitter_centers)
    direct_by_user_site: ComplexArray = _load_or_trace_direct_channels(
        checkpoint_directory=checkpoint_directory,
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        transmitter_positions=transmitter_positions,
        user_positions=user_positions,
        settings=settings,
        cell_model=cell_model,
    )
    _write_progress(
        checkpoint_directory,
        scenario_name=scenario_name,
        stage="direct_channels",
        completed=1,
        total=1,
        started_at=started_at,
    )
    tx_to_irs: ComplexArray = _materialize_tx_to_irs(
        checkpoint_directory=checkpoint_directory,
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        element_positions=element_positions,
        panel_normals=panel_normals,
        best_sites=best_sites,
        transmitter_positions=transmitter_positions,
        settings=settings,
        cell_model=cell_model,
    )
    _write_progress(
        checkpoint_directory,
        scenario_name=scenario_name,
        stage="tx_to_irs",
        completed=panel_count,
        total=panel_count,
        started_at=started_at,
    )
    for start in range(0, panel_count, settings.panel_batch_size):
        stop = min(start + settings.panel_batch_size, panel_count)
        batch_path: Path = _batch_path(checkpoint_directory, start, stop)
        if batch_path.exists():
            print(f"IRS_USER_BATCH_CACHED={scenario_name}:{start}:{stop}")
            continue
        source_positions: FloatArray = element_positions[start:stop].reshape(-1, 3)
        source_normals: FloatArray = np.repeat(
            panel_normals[start:stop], IRS_ROWS * IRS_COLUMNS, axis=0
        ).astype(np.float32, copy=False)
        source_names: NameArray = np.asarray(
            [
                f"irsuser_panel_{panel}_element_{element}"
                for panel in range(start, stop)
                for element in range(IRS_ROWS * IRS_COLUMNS)
            ],
            dtype=np.str_,
        )
        batch_user_mask: BoolArray = np.asarray(qualifying[start:stop].any(axis=0), dtype=np.bool_)
        batch_users: IndexArray = np.asarray(np.flatnonzero(batch_user_mask), dtype=np.int64)
        target_names: NameArray = np.asarray(
            [f"pool_user_{int(user)}" for user in batch_users], dtype=np.str_
        )
        channel: ComplexArray = solve_complex_links(
            scene_xml=scene_xml,
            frequency_hz=frequency_hz,
            source_names=source_names,
            source_positions=source_positions,
            target_names=target_names,
            target_positions=user_positions[batch_users],
            samples_per_source=settings.samples_per_source,
            source_irs_normals=source_normals,
            cell_model=cell_model,
            amplitude_scale=cell_model.cascaded_amplitude_scale(wavelength_m),
        ).reshape(batch_users.shape[0], stop - start, IRS_ROWS * IRS_COLUMNS)
        local_links: list[ComplexArray] = []
        local_user_indices: list[IndexArray] = []
        for local_panel in range(stop - start):
            users: IndexArray = np.asarray(
                np.flatnonzero(qualifying[start + local_panel]), dtype=np.int64
            )
            positions: IndexArray = np.asarray(np.searchsorted(batch_users, users), dtype=np.int64)
            local_links.append(np.asarray(channel[positions, local_panel], dtype=np.complex64))
            local_user_indices.append(users)
        temporary: Path = batch_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            user_indices=np.concatenate(local_user_indices).astype(np.int64),
            irs_to_user=np.concatenate(local_links, axis=0).astype(np.complex64),
            user_counts=user_counts[start:stop],
        )
        temporary.replace(batch_path)
        _write_progress(
            checkpoint_directory,
            scenario_name=scenario_name,
            stage="irs_to_user",
            completed=stop,
            total=panel_count,
            started_at=started_at,
        )
        print(f"IRS_USER_BATCH_SAVED={scenario_name}:{start}:{stop}")
    flat_users: IndexArray = np.zeros(link_count, dtype=np.int64)
    irs_to_user: ComplexArray = np.zeros((link_count, IRS_ROWS * IRS_COLUMNS), dtype=np.complex64)
    for start in range(0, panel_count, settings.panel_batch_size):
        stop = min(start + settings.panel_batch_size, panel_count)
        begin_link: int = int(offsets[start])
        end_link: int = int(offsets[stop])
        with np.load(_batch_path(checkpoint_directory, start, stop)) as batch:
            flat_users[begin_link:end_link] = np.asarray(batch["user_indices"], dtype=np.int64)
            irs_to_user[begin_link:end_link] = np.asarray(batch["irs_to_user"], dtype=np.complex64)
    direct: ComplexArray = np.zeros((link_count, len(MIMO_ANTENNA_COORDINATES)), dtype=np.complex64)
    user_locations: FloatArray = np.zeros((link_count, 3), dtype=np.float32)
    for panel in range(panel_count):
        begin: int = int(offsets[panel])
        end: int = int(offsets[panel + 1])
        users = flat_users[begin:end]
        site: int = int(best_sites[panel])
        direct[begin:end] = direct_by_user_site[users, site]
        user_locations[begin:end] = user_positions[users]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output: Path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_output,
        panel_indices=panel_indices,
        panel_names=panel_names,
        panel_building_names=building_names,
        panel_centers_m=panel_centers,
        panel_normals=panel_normals,
        panel_user_offsets=offsets,
        panel_user_counts=user_counts,
        user_indices=flat_users,
        user_source_indices=user_source_indices[flat_users],
        user_locations_m=user_locations,
        transmitter_site_indices=best_sites,
        transmitter_locations_m=transmitter_positions[best_sites],
        irs_element_locations_m=element_positions,
        direct=direct,
        tx_to_irs=tx_to_irs,
        irs_to_user=irs_to_user,
        metadata_json=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    )
    temporary_output.replace(output_path)
    summary = {
        **metadata,
        "output_path": str(output_path.resolve()),
        "elapsed_seconds": perf_counter() - started_at,
    }
    write_json(summary_path, summary)
    _write_progress(
        checkpoint_directory,
        scenario_name=scenario_name,
        stage="complete",
        completed=panel_count,
        total=panel_count,
        started_at=started_at,
    )
    print(f"MATERIALIZED_SCENE_SAVED={scenario_name}:{output_path.resolve()}")
    return summary


def _combine_scene_pools(
    scenario_names: tuple[str, ...],
    *,
    output_root: Path,
    combined_output: Path,
    settings: MaterializationSettings,
) -> None:
    pools: list[dict[str, npt.NDArray[np.generic]]] = []
    for scenario_name in scenario_names:
        path: Path = output_root / "scenes" / f"{scenario_name}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            pools.append({name: np.asarray(arrays[name]) for name in arrays.files})
    panel_counts: IndexArray = np.asarray(
        [pool["panel_indices"].shape[0] for pool in pools], dtype=np.int64
    )
    link_counts: IndexArray = np.asarray(
        [pool["direct"].shape[0] for pool in pools], dtype=np.int64
    )
    all_user_counts: IndexArray = np.concatenate(
        [np.asarray(pool["panel_user_counts"], dtype=np.int64) for pool in pools]
    )
    global_offsets: IndexArray = np.zeros(all_user_counts.shape[0] + 1, dtype=np.int64)
    global_offsets[1:] = np.cumsum(all_user_counts, dtype=np.int64)
    metadata: dict[str, object] = {
        "format": "IRS2 combined multi-city tiled-panel optimizer channel pool",
        "format_version": 1,
        "scenario_names": list(scenario_names),
        "scene_count": len(scenario_names),
        "panel_count": int(np.sum(panel_counts)),
        "link_count": int(np.sum(link_counts)),
        "per_scene_panel_counts": panel_counts.tolist(),
        "per_scene_link_counts": link_counts.tolist(),
        "settings": asdict(settings),
        "irs_shape": [IRS_ROWS, IRS_COLUMNS],
        "mimo_shape": [2, 2],
        "source_scene_sha256": {
            name: file_hash(output_root / "scenes" / f"{name}.npz") for name in scenario_names
        },
    }
    summary_path = combined_output.with_suffix(".json")
    if combined_output.exists() and summary_path.exists() and read_json(summary_path) == metadata:
        print(f"COMBINED_POOL_CACHED={combined_output.resolve()}")
        return
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = combined_output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        scene_names=np.asarray(scenario_names, dtype=np.str_),
        panel_scene_indices=np.repeat(np.arange(len(pools), dtype=np.int64), panel_counts),
        link_scene_indices=np.repeat(np.arange(len(pools), dtype=np.int64), link_counts),
        panel_indices=np.concatenate(
            [np.asarray(pool["panel_indices"], dtype=np.int64) for pool in pools]
        ),
        panel_names=np.concatenate(
            [np.asarray(pool["panel_names"], dtype=np.str_) for pool in pools]
        ),
        panel_building_names=np.concatenate(
            [np.asarray(pool["panel_building_names"], dtype=np.str_) for pool in pools]
        ),
        panel_centers_m=np.concatenate(
            [np.asarray(pool["panel_centers_m"], dtype=np.float32) for pool in pools]
        ),
        panel_normals=np.concatenate(
            [np.asarray(pool["panel_normals"], dtype=np.float32) for pool in pools]
        ),
        panel_user_offsets=global_offsets,
        panel_user_counts=all_user_counts,
        user_indices=np.concatenate(
            [np.asarray(pool["user_indices"], dtype=np.int64) for pool in pools]
        ),
        user_source_indices=np.concatenate(
            [np.asarray(pool["user_source_indices"], dtype=np.int64) for pool in pools]
        ),
        user_locations_m=np.concatenate(
            [np.asarray(pool["user_locations_m"], dtype=np.float32) for pool in pools]
        ),
        transmitter_site_indices=np.concatenate(
            [np.asarray(pool["transmitter_site_indices"], dtype=np.int64) for pool in pools]
        ),
        transmitter_locations_m=np.concatenate(
            [np.asarray(pool["transmitter_locations_m"], dtype=np.float32) for pool in pools]
        ),
        irs_element_locations_m=np.concatenate(
            [np.asarray(pool["irs_element_locations_m"], dtype=np.float32) for pool in pools]
        ),
        direct=np.concatenate([np.asarray(pool["direct"], dtype=np.complex64) for pool in pools]),
        tx_to_irs=np.concatenate(
            [np.asarray(pool["tx_to_irs"], dtype=np.complex64) for pool in pools]
        ),
        irs_to_user=np.concatenate(
            [np.asarray(pool["irs_to_user"], dtype=np.complex64) for pool in pools]
        ),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    )
    temporary.replace(combined_output)
    write_json(combined_output.with_suffix(".json"), metadata)
    print(f"COMBINED_POOL_SAVED={combined_output.resolve()}")


def materialize_collection(
    *,
    filter_root: Path,
    output_root: Path,
    combined_output: Path,
    settings: MaterializationSettings,
) -> None:
    """Materialize every indexed scene, then combine compatible arrays."""
    index_path: Path = filter_root / "dataset_index.json"
    index: dict[str, object] = json.loads(index_path.read_text(encoding="utf-8"))
    raw_names: object = index.get("selected_scenarios")
    if not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
        raise ValueError("Filter dataset index has invalid selected_scenarios.")
    scenario_names: tuple[str, ...] = tuple(str(name) for name in raw_names)
    if not scenario_names or index.get("completed_scene_count") != len(scenario_names):
        raise ValueError(
            "Filter collection is incomplete; finish power filtering before materializing"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    started_at: float = perf_counter()
    for index_value, scenario_name in enumerate(scenario_names, start=1):
        print(f"MATERIALIZATION_PROGRESS={index_value}/{len(scenario_names)}:{scenario_name}")
        summary: dict[str, object] = _materialize_scene(
            scenario_name,
            filter_path=filter_root / "scenes" / f"{scenario_name}.npz",
            output_root=output_root,
            settings=settings,
        )
        summaries.append(summary)
        write_json(
            output_root / "dataset_index.json",
            {
                "format": "IRS2 multi-city optimizer channel collection",
                "format_version": 1,
                "selected_scenarios": list(scenario_names),
                "completed_scene_count": len(summaries),
                "scene_count": len(scenario_names),
                "panel_count": sum(cast(int, item["panel_count"]) for item in summaries),
                "link_count": sum(cast(int, item["link_count"]) for item in summaries),
                "settings": asdict(settings),
                "elapsed_seconds": perf_counter() - started_at,
                "updated_at_unix": time(),
                "scenes": summaries,
            },
        )
    _combine_scene_pools(
        scenario_names,
        output_root=output_root,
        combined_output=combined_output,
        settings=settings,
    )
    print(f"MATERIALIZATION_COMPLETE={len(scenario_names)}")
    print(f"ELAPSED_SECONDS={perf_counter() - started_at:.1f}")


def _materialize_tx_to_irs(
    *,
    checkpoint_directory: Path,
    scene_xml: str,
    frequency_hz: float,
    element_positions: FloatArray,
    panel_normals: FloatArray,
    best_sites: IndexArray,
    transmitter_positions: FloatArray,
    settings: MaterializationSettings,
    cell_model: TangCellModel,
) -> ComplexArray:
    cache_path: Path = checkpoint_directory / "tx_to_irs.npy"
    panel_count: int = element_positions.shape[0]
    expected_shape: tuple[int, int, int] = (
        panel_count,
        IRS_ROWS * IRS_COLUMNS,
        len(MIMO_ANTENNA_COORDINATES),
    )
    if cache_path.exists():
        cached: ComplexArray = np.asarray(np.load(cache_path), dtype=np.complex64)
        if cached.shape != expected_shape:
            raise ValueError("Invalid cached TX-to-IRS shape.")
        return cached
    all_sources: FloatArray = transmitter_positions.reshape(-1, 3)
    site_count: int = int(transmitter_positions.shape[0])
    antenna_count: int = int(transmitter_positions.shape[1])
    source_names: NameArray = np.asarray(
        [
            f"site_{site}_antenna_{antenna}"
            for site in range(site_count)
            for antenna in range(antenna_count)
        ],
        dtype=np.str_,
    )
    result: ComplexArray = np.zeros(expected_shape, dtype=np.complex64)
    for start in range(0, panel_count, settings.tx_target_panel_batch_size):
        stop: int = min(start + settings.tx_target_panel_batch_size, panel_count)
        batch_path: Path = checkpoint_directory / f"tx_irs_{start:05d}_{stop:05d}.npy"
        expected_batch_shape: tuple[int, int, int] = (
            stop - start,
            IRS_ROWS * IRS_COLUMNS,
            len(MIMO_ANTENNA_COORDINATES),
        )
        if batch_path.exists():
            cached_batch: ComplexArray = np.asarray(np.load(batch_path), dtype=np.complex64)
            if cached_batch.shape != expected_batch_shape:
                raise ValueError(f"Invalid cached TX-to-IRS batch {batch_path}.")
            result[start:stop] = cached_batch
            print(f"TX_IRS_BATCH_CACHED={start}:{stop}")
            continue
        target_positions: FloatArray = element_positions[start:stop].reshape(-1, 3)
        target_normals: FloatArray = np.repeat(
            panel_normals[start:stop], IRS_ROWS * IRS_COLUMNS, axis=0
        ).astype(np.float32, copy=False)
        target_names: NameArray = np.asarray(
            [f"txirs_panel_{p}_element_{e}" for p in range(start, stop) for e in range(16)],
            dtype=np.str_,
        )
        channel: ComplexArray = solve_complex_links(
            scene_xml=scene_xml,
            frequency_hz=frequency_hz,
            source_names=source_names,
            source_positions=all_sources,
            target_names=target_names,
            target_positions=target_positions,
            samples_per_source=settings.samples_per_source,
            target_irs_normals=target_normals,
            cell_model=cell_model,
        ).reshape(stop - start, 16, site_count, antenna_count)
        for local_panel, site in enumerate(best_sites[start:stop]):
            result[start + local_panel] = channel[local_panel, :, int(site)]
        save_npy(batch_path, result[start:stop])
        print(f"TX_IRS_PROGRESS={stop}/{panel_count}")
    save_npy(cache_path, result)
    return result


def _batch_path(directory: Path, start: int, stop: int) -> Path:
    return directory / f"irs_user_{start:05d}_{stop:05d}.npz"
