"""Deterministic multi-city candidate planning and resumable power filtering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter, time
from typing import cast

import deepmimo as dm
import numpy as np
import numpy.typing as npt

from ..io import read_json, save_npy, validate_manifest, write_json
from .cell import TangCellModel
from .config import (
    CANDIDATE_COLUMNS,
    CANDIDATE_PANEL_GAP_M,
    CANDIDATE_ROWS,
    CANDIDATE_SPACING_M,
    CANDIDATE_WALL_MARGIN_M,
    CANDIDATE_WALL_OFFSET_M,
    IRS_PATTERN_EXPONENT,
    IRS_REFLECTION_AMPLITUDE,
    MAXIMUM_IRS_PATH_LOSS_DB,
    MAXIMUM_USER_HEIGHT_M,
    PHYSICAL_CELL_SIZE_M,
    PHYSICAL_IRS_COLUMNS,
    PHYSICAL_IRS_ROWS,
    RAY_TRACING_SEED,
    RELATIVE_THRESHOLD_DB,
    USER_EDGE_BUFFER_M,
    MultiCitySettings,
)
from .geometry import CandidateGeometry, IRSPanel, create_tiled_wall_irs_panels
from .scene import create_sionna_full_scene_xml
from .tracing import (
    PanelPowerFilterResult,
    RayTracingSettings,
    compare_panel_to_direct_power,
    solve_total_link_power,
)

FloatArray = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
NameArray = npt.NDArray[np.str_]


@dataclass(frozen=True)
class SceneEndpoints:
    """Aligned transmitter and sampled-user locations for one city."""

    transmitter_names: NameArray
    transmitter_positions_m: FloatArray
    user_names: NameArray
    user_positions_m: FloatArray
    user_source_indices: IndexArray


@dataclass(frozen=True)
class ScenePlan:
    """Loaded geometry, endpoints, and deterministic panel selection."""

    scenario: dm.MacroDataset
    geometry: CandidateGeometry
    endpoints: SceneEndpoints
    total_tiled_panel_count: int
    selection_seed: int
    user_seed: int


def _stable_seed(base_seed: int, scenario_name: str, purpose: str) -> int:
    payload: bytes = f"{base_seed}:{scenario_name}:{purpose}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little", signed=False)


def _load_scenario(scenario_name: str) -> dm.MacroDataset:
    scenario_directory: Path = Path("deepmimo_scenarios") / scenario_name
    if not scenario_directory.exists():
        downloaded_path: str | None = dm.download(scenario_name)
        if downloaded_path is None and not scenario_directory.exists():
            raise RuntimeError(f"Could not download {scenario_name!r}.")
    loaded: dm.Dataset | dm.MacroDataset = dm.load(
        scenario_name,
        max_paths=1,
        tx_sets={0: "all"},
        rx_sets={1: "all"},
        matrices=["rx_pos", "tx_pos"],
    )
    if not isinstance(loaded, dm.MacroDataset):
        raise TypeError(f"Expected a multi-transmitter MacroDataset for {scenario_name}.")
    return loaded


def _sample_endpoints(
    scenario_name: str,
    scenario: dm.MacroDataset,
    *,
    user_count: int,
    seed: int,
) -> SceneEndpoints:
    receiver_positions: FloatArray = np.asarray(
        scenario[0].rx_pos,
        dtype=np.float32,
    )
    transmitter_positions: FloatArray = np.asarray(
        np.vstack([np.asarray(dataset.tx_pos, dtype=np.float32)[0] for dataset in scenario]),
        dtype=np.float32,
    )
    bounds: dm.BoundingBox = scenario.scene.bounding_box
    eligible_mask: BoolArray = np.asarray(
        (receiver_positions[:, 0] >= float(bounds.x_min) + USER_EDGE_BUFFER_M)
        & (receiver_positions[:, 0] <= float(bounds.x_max) - USER_EDGE_BUFFER_M)
        & (receiver_positions[:, 1] >= float(bounds.y_min) + USER_EDGE_BUFFER_M)
        & (receiver_positions[:, 1] <= float(bounds.y_max) - USER_EDGE_BUFFER_M)
        & (receiver_positions[:, 2] < MAXIMUM_USER_HEIGHT_M),
        dtype=np.bool_,
    )
    eligible_indices: IndexArray = np.asarray(
        np.flatnonzero(eligible_mask),
        dtype=np.int64,
    )
    if eligible_indices.shape[0] < user_count:
        raise ValueError(
            f"{scenario_name} has only {eligible_indices.shape[0]} eligible users; "
            f"{user_count} were requested."
        )
    selected_indices: IndexArray = np.asarray(
        np.random.default_rng(seed).choice(
            eligible_indices,
            size=user_count,
            replace=False,
        ),
        dtype=np.int64,
    )
    return SceneEndpoints(
        transmitter_names=np.asarray(
            [f"filter_tx_{index}" for index in range(transmitter_positions.shape[0])],
            dtype=np.str_,
        ),
        transmitter_positions_m=transmitter_positions,
        user_names=np.asarray(
            [f"filter_user_{index}" for index in range(user_count)],
            dtype=np.str_,
        ),
        user_positions_m=np.asarray(
            receiver_positions[selected_indices],
            dtype=np.float32,
        ),
        user_source_indices=selected_indices,
    )


def _select_candidate_geometry(
    scenario_name: str,
    scenario: dm.MacroDataset,
    *,
    maximum_panels: int,
    seed: int,
) -> tuple[CandidateGeometry, int]:
    all_panels: tuple[IRSPanel, ...] = create_tiled_wall_irs_panels(
        scenario.scene,
        rows=CANDIDATE_ROWS,
        columns=CANDIDATE_COLUMNS,
        spacing_m=CANDIDATE_SPACING_M,
        panel_gap_m=CANDIDATE_PANEL_GAP_M,
        mounting_margin_m=CANDIDATE_WALL_MARGIN_M,
        wall_offset_m=CANDIDATE_WALL_OFFSET_M,
    )
    total_count: int = len(all_panels)
    selected_original_indices: IndexArray
    if total_count > maximum_panels:
        selected_original_indices = np.sort(
            np.asarray(
                np.random.default_rng(seed).choice(
                    total_count,
                    size=maximum_panels,
                    replace=False,
                ),
                dtype=np.int64,
            )
        )
    else:
        selected_original_indices = np.arange(total_count, dtype=np.int64)
    selected_panels: tuple[IRSPanel, ...] = tuple(
        all_panels[int(index)] for index in selected_original_indices
    )
    geometry = CandidateGeometry(
        indices=selected_original_indices,
        names=np.asarray([panel.name for panel in selected_panels], dtype=np.str_),
        building_names=np.asarray(
            [panel.building_name for panel in selected_panels],
            dtype=np.str_,
        ),
        centers_m=np.asarray(
            [
                np.mean(np.asarray(panel.positions, dtype=np.float32), axis=0)
                for panel in selected_panels
            ],
            dtype=np.float32,
        ),
        normals=np.asarray([panel.normal for panel in selected_panels], dtype=np.float32),
    )
    print(
        f"SCENE_PLAN={scenario_name} TOTAL_PANELS={total_count} "
        f"SELECTED_PANELS={geometry.indices.shape[0]}"
    )
    return geometry, total_count


def _build_scene_plan(
    scenario_name: str,
    settings: MultiCitySettings,
) -> ScenePlan:
    scenario: dm.MacroDataset = _load_scenario(scenario_name)
    panel_seed: int = _stable_seed(settings.selection_seed, scenario_name, "panels")
    user_seed: int = _stable_seed(settings.selection_seed, scenario_name, "users")
    geometry, total_count = _select_candidate_geometry(
        scenario_name,
        scenario,
        maximum_panels=settings.maximum_panels_per_scene,
        seed=panel_seed,
    )
    endpoints: SceneEndpoints = _sample_endpoints(
        scenario_name,
        scenario,
        user_count=settings.user_count,
        seed=user_seed,
    )
    return ScenePlan(
        scenario=scenario,
        geometry=geometry,
        endpoints=endpoints,
        total_tiled_panel_count=total_count,
        selection_seed=panel_seed,
        user_seed=user_seed,
    )


def _manifest(
    scenario_name: str,
    plan: ScenePlan,
    settings: MultiCitySettings,
) -> dict[str, object]:
    frequency_hz: float = float(plan.scenario[0].rt_params.frequency)
    wavelength_m: float = 299_792_458.0 / frequency_hz
    cell_model = TangCellModel(
        width_m=PHYSICAL_CELL_SIZE_M,
        height_m=PHYSICAL_CELL_SIZE_M,
        reflection_amplitude=IRS_REFLECTION_AMPLITUDE,
        pattern_exponent=IRS_PATTERN_EXPONENT,
    )
    selected_indices_hash: str = sha256(
        plan.geometry.indices.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return {
        "format": "IRS2 multi-city tiled candidate panel filter dataset",
        "format_version": 1,
        "scenario": scenario_name,
        "frequency_hz": frequency_hz,
        "total_tiled_panel_count": plan.total_tiled_panel_count,
        "selected_panel_count": int(plan.geometry.indices.shape[0]),
        "maximum_panels_per_scene": settings.maximum_panels_per_scene,
        "panel_selection": (
            "uniform random subset without replacement, sorted by original index"
            if plan.total_tiled_panel_count > settings.maximum_panels_per_scene
            else "all tiled panels"
        ),
        "panel_selection_seed": plan.selection_seed,
        "selected_panel_indices_sha256": selected_indices_hash,
        "user_count": settings.user_count,
        "user_selection_seed": plan.user_seed,
        "user_selection": {
            "maximum_height_m": MAXIMUM_USER_HEIGHT_M,
            "scene_edge_buffer_m": USER_EDGE_BUFFER_M,
            "method": "uniform random subset without replacement",
        },
        "relative_threshold_db": RELATIVE_THRESHOLD_DB,
        "maximum_irs_path_loss_db": MAXIMUM_IRS_PATH_LOSS_DB,
        "ray_tracing_seed": RAY_TRACING_SEED,
        "ray_tracing": {
            "samples_per_source": settings.samples_per_source,
            "panel_source_batch_size": settings.panel_source_batch_size,
            "panel_target_batch_size": settings.panel_target_batch_size,
        },
        "candidate_tiling": {
            "rows": CANDIDATE_ROWS,
            "columns": CANDIDATE_COLUMNS,
            "spacing_m": CANDIDATE_SPACING_M,
            "panel_gap_m": CANDIDATE_PANEL_GAP_M,
            "wall_margin_m": CANDIDATE_WALL_MARGIN_M,
            "wall_offset_m": CANDIDATE_WALL_OFFSET_M,
            "purpose": "candidate centers and normals only",
        },
        "physical_panel": {
            "rows": PHYSICAL_IRS_ROWS,
            "columns": PHYSICAL_IRS_COLUMNS,
            "element_count": PHYSICAL_IRS_ROWS * PHYSICAL_IRS_COLUMNS,
            **cell_model.metadata(wavelength_m),
        },
        "aggregation": "coherent complex path sum before conversion to power",
        "checkpointing": {
            "direct_power": "after the one-time trace",
            "transmitter_to_panel_power": "after every target batch",
            "panel_to_user_filter": "after every source batch",
            "progress_json": "after every completed batch",
        },
    }


def _write_scene_progress(
    checkpoint_directory: Path,
    *,
    scenario_name: str,
    stage: str,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    elapsed_seconds: float = perf_counter() - started_at
    write_json(
        checkpoint_directory / "progress.json",
        {
            "scenario": scenario_name,
            "stage": stage,
            "completed": completed,
            "total": total,
            "fraction": 1.0 if total == 0 else completed / total,
            "elapsed_seconds": elapsed_seconds,
            "updated_at_unix": time(),
        },
    )


def _load_or_trace_direct_power(
    *,
    scenario_name: str,
    checkpoint_directory: Path,
    scene_xml: str,
    frequency_hz: float,
    endpoints: SceneEndpoints,
    ray_settings: RayTracingSettings,
    started_at: float,
) -> FloatArray:
    cache_path: Path = checkpoint_directory / "direct_power.npy"
    expected_shape: tuple[int, int] = (
        endpoints.user_positions_m.shape[0],
        endpoints.transmitter_positions_m.shape[0],
    )
    if cache_path.exists():
        cached: FloatArray = np.asarray(np.load(cache_path), dtype=np.float32)
        if cached.shape != expected_shape:
            raise ValueError(f"Invalid direct-power cache shape {cached.shape}.")
        print(f"DIRECT_POWER_CACHED={scenario_name}")
        return cached
    result: FloatArray = solve_total_link_power(
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        source_names=endpoints.transmitter_names,
        source_positions=endpoints.transmitter_positions_m,
        target_names=endpoints.user_names,
        target_positions=endpoints.user_positions_m,
        settings=ray_settings,
    )
    save_npy(cache_path, result)
    _write_scene_progress(
        checkpoint_directory,
        scenario_name=scenario_name,
        stage="direct_power",
        completed=1,
        total=1,
        started_at=started_at,
    )
    print(f"DIRECT_POWER_SAVED={scenario_name}")
    return result


def _load_or_trace_transmitter_to_panels(
    *,
    scenario_name: str,
    checkpoint_directory: Path,
    geometry: CandidateGeometry,
    scene_xml: str,
    frequency_hz: float,
    endpoints: SceneEndpoints,
    ray_settings: RayTracingSettings,
    cell_model: TangCellModel,
    target_batch_size: int,
    started_at: float,
) -> FloatArray:
    cache_path: Path = checkpoint_directory / "transmitter_to_panel_power.npy"
    panel_count: int = geometry.indices.shape[0]
    transmitter_count: int = endpoints.transmitter_positions_m.shape[0]
    expected_shape: tuple[int, int] = (panel_count, transmitter_count)
    if cache_path.exists():
        cached: FloatArray = np.asarray(np.load(cache_path), dtype=np.float32)
        if cached.shape != expected_shape:
            raise ValueError(f"Invalid TX-to-panel cache shape {cached.shape}.")
        print(f"TX_TO_PANEL_CACHED={scenario_name}")
        return cached
    batch_directory: Path = checkpoint_directory / "transmitter_to_panel_batches"
    batch_directory.mkdir(parents=True, exist_ok=True)
    result: FloatArray = np.zeros(expected_shape, dtype=np.float32)
    for start in range(0, panel_count, target_batch_size):
        stop: int = min(start + target_batch_size, panel_count)
        batch_path: Path = batch_directory / f"targets_{start:05d}_{stop:05d}.npy"
        expected_batch_shape: tuple[int, int] = (stop - start, transmitter_count)
        if batch_path.exists():
            cached_batch: FloatArray = np.asarray(
                np.load(batch_path),
                dtype=np.float32,
            )
            if cached_batch.shape != expected_batch_shape:
                raise ValueError(f"Invalid TX-to-panel batch {batch_path}.")
            result[start:stop] = cached_batch
            print(f"TX_TO_PANEL_BATCH_CACHED={scenario_name}:{start}:{stop}")
            continue
        target_names: NameArray = np.asarray(
            [f"candidate_panel_{index:05d}" for index in range(start, stop)],
            dtype=np.str_,
        )
        batch_result: FloatArray = solve_total_link_power(
            scene_xml=scene_xml,
            frequency_hz=frequency_hz,
            source_names=endpoints.transmitter_names,
            source_positions=endpoints.transmitter_positions_m,
            target_names=target_names,
            target_positions=geometry.centers_m[start:stop],
            settings=ray_settings,
            cell_model=cell_model,
            target_irs_normals=geometry.normals[start:stop],
        )
        result[start:stop] = batch_result
        save_npy(batch_path, batch_result)
        _write_scene_progress(
            checkpoint_directory,
            scenario_name=scenario_name,
            stage="transmitter_to_panel_power",
            completed=stop,
            total=panel_count,
            started_at=started_at,
        )
        print(f"TX_TO_PANEL_BATCH_SAVED={scenario_name}:{start}:{stop}")
    save_npy(cache_path, result)
    return result


def _assemble_scene_dataset(
    *,
    scenario_name: str,
    output_path: Path,
    checkpoint_directory: Path,
    manifest: dict[str, object],
    geometry: CandidateGeometry,
    endpoints: SceneEndpoints,
    direct_power: FloatArray,
    transmitter_to_panel_power: FloatArray,
    panel_source_batch_size: int,
) -> None:
    panel_count: int = geometry.indices.shape[0]
    user_count: int = endpoints.user_positions_m.shape[0]
    cascaded_power: FloatArray = np.zeros((panel_count, user_count), dtype=np.float32)
    margin_db: FloatArray = np.full(
        (panel_count, user_count),
        -np.inf,
        dtype=np.float32,
    )
    path_loss_db: FloatArray = np.full(
        (panel_count, user_count),
        np.inf,
        dtype=np.float32,
    )
    qualifying_mask: BoolArray = np.zeros((panel_count, user_count), dtype=np.bool_)
    best_transmitter_index: IndexArray = np.zeros(panel_count, dtype=np.int64)
    for start in range(0, panel_count, panel_source_batch_size):
        stop: int = min(start + panel_source_batch_size, panel_count)
        batch_path: Path = _batch_path(checkpoint_directory, start, stop)
        if not batch_path.exists():
            raise FileNotFoundError(f"Missing panel checkpoint {batch_path}.")
        with np.load(batch_path, allow_pickle=False) as batch:
            cascaded_power[start:stop] = np.asarray(batch["cascaded_irs_power"], dtype=np.float32)
            margin_db[start:stop] = np.asarray(batch["irs_direct_margin_db"], dtype=np.float32)
            path_loss_db[start:stop] = np.asarray(batch["irs_path_loss_db"], dtype=np.float32)
            qualifying_mask[start:stop] = np.asarray(batch["qualifying_mask"], dtype=np.bool_)
            best_transmitter_index[start:stop] = np.asarray(
                batch["best_transmitter_index"], dtype=np.int64
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        panel_indices=geometry.indices,
        panel_names=geometry.names,
        panel_building_names=geometry.building_names,
        panel_centers_m=geometry.centers_m,
        panel_normals=geometry.normals,
        user_names=endpoints.user_names,
        user_positions_m=endpoints.user_positions_m,
        user_source_indices=endpoints.user_source_indices,
        transmitter_names=endpoints.transmitter_names,
        transmitter_positions_m=endpoints.transmitter_positions_m,
        direct_power=direct_power,
        direct_best_power=np.max(direct_power, axis=1).astype(np.float32),
        transmitter_to_panel_power=transmitter_to_panel_power,
        best_panel_transmitter_index=best_transmitter_index,
        cascaded_irs_power=cascaded_power,
        irs_direct_margin_db=margin_db,
        irs_path_loss_db=path_loss_db,
        qualifying_mask=qualifying_mask,
        qualifying_user_count=np.sum(qualifying_mask, axis=1, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(manifest, indent=2, sort_keys=True)),
    )
    temporary_path.replace(output_path)
    qualifying_counts: IndexArray = np.asarray(
        np.sum(qualifying_mask, axis=1, dtype=np.int64),
        dtype=np.int64,
    )
    summary: dict[str, object] = {
        **manifest,
        "output_path": str(output_path.resolve()),
        "panels_with_qualifying_users": int(np.count_nonzero(qualifying_counts)),
        "total_qualifying_panel_user_links": int(np.sum(qualifying_counts)),
        "maximum_qualifying_users_for_one_panel": int(np.max(qualifying_counts)),
    }
    write_json(output_path.with_suffix(".json"), summary)
    print(f"SCENE_DATASET_SAVED={scenario_name}:{output_path.resolve()}")


def _build_scene(
    scenario_name: str,
    *,
    output_root: Path,
    settings: MultiCitySettings,
    plan_only: bool,
) -> dict[str, object]:
    started_at: float = perf_counter()
    output_path: Path = output_root / "scenes" / f"{scenario_name}.npz"
    summary_path: Path = output_path.with_suffix(".json")
    if output_path.exists() and summary_path.exists() and not plan_only:
        summary = read_json(summary_path)
        expected: dict[str, object] = {
            "user_count": settings.user_count,
            "maximum_panels_per_scene": settings.maximum_panels_per_scene,
            "panel_selection_seed": _stable_seed(settings.selection_seed, scenario_name, "panels"),
            "user_selection_seed": _stable_seed(settings.selection_seed, scenario_name, "users"),
            "ray_tracing": {
                "samples_per_source": settings.samples_per_source,
                "panel_source_batch_size": settings.panel_source_batch_size,
                "panel_target_batch_size": settings.panel_target_batch_size,
            },
            "ray_tracing_seed": RAY_TRACING_SEED,
            "relative_threshold_db": RELATIVE_THRESHOLD_DB,
            "maximum_irs_path_loss_db": MAXIMUM_IRS_PATH_LOSS_DB,
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Completed filter settings differ for {scenario_name}")
        print(f"SCENE_ALREADY_COMPLETE={scenario_name}")
        return summary
    plan: ScenePlan = _build_scene_plan(scenario_name, settings)
    manifest: dict[str, object] = _manifest(scenario_name, plan, settings)
    if plan_only:
        return manifest
    checkpoint_directory: Path = output_root / "checkpoints" / scenario_name
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    validate_manifest(checkpoint_directory, manifest)
    _write_scene_progress(
        checkpoint_directory,
        scenario_name=scenario_name,
        stage="planned",
        completed=0,
        total=int(plan.geometry.indices.shape[0]),
        started_at=started_at,
    )
    frequency_hz: float = float(plan.scenario[0].rt_params.frequency)
    wavelength_m: float = 299_792_458.0 / frequency_hz
    mesh_directory: Path = output_root / "meshes" / scenario_name
    scene_xml: str = create_sionna_full_scene_xml(
        plan.scenario.scene,
        output_directory=mesh_directory.resolve(),
    )
    ray_settings = RayTracingSettings(
        max_depth=3,
        samples_per_src=settings.samples_per_source,
        seed=RAY_TRACING_SEED,
    )
    cell_model = TangCellModel(
        width_m=PHYSICAL_CELL_SIZE_M,
        height_m=PHYSICAL_CELL_SIZE_M,
        reflection_amplitude=IRS_REFLECTION_AMPLITUDE,
        pattern_exponent=IRS_PATTERN_EXPONENT,
    )
    direct_power: FloatArray = _load_or_trace_direct_power(
        scenario_name=scenario_name,
        checkpoint_directory=checkpoint_directory,
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        endpoints=plan.endpoints,
        ray_settings=ray_settings,
        started_at=started_at,
    )
    transmitter_to_panel_power: FloatArray = _load_or_trace_transmitter_to_panels(
        scenario_name=scenario_name,
        checkpoint_directory=checkpoint_directory,
        geometry=plan.geometry,
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        endpoints=plan.endpoints,
        ray_settings=ray_settings,
        cell_model=cell_model,
        target_batch_size=settings.panel_target_batch_size,
        started_at=started_at,
    )
    panel_count: int = plan.geometry.indices.shape[0]
    for start in range(0, panel_count, settings.panel_source_batch_size):
        stop: int = min(start + settings.panel_source_batch_size, panel_count)
        _calculate_panel_batch(
            start=start,
            stop=stop,
            checkpoint_directory=checkpoint_directory,
            geometry=plan.geometry,
            scene_xml=scene_xml,
            frequency_hz=frequency_hz,
            user_names=plan.endpoints.user_names,
            user_positions=plan.endpoints.user_positions_m,
            direct_power=direct_power,
            transmitter_to_panel_power=transmitter_to_panel_power,
            ray_settings=ray_settings,
            cell_model=cell_model,
            wavelength_m=wavelength_m,
        )
        _write_scene_progress(
            checkpoint_directory,
            scenario_name=scenario_name,
            stage="panel_to_user_filter",
            completed=stop,
            total=panel_count,
            started_at=started_at,
        )
    _assemble_scene_dataset(
        scenario_name=scenario_name,
        output_path=output_path,
        checkpoint_directory=checkpoint_directory,
        manifest=manifest,
        geometry=plan.geometry,
        endpoints=plan.endpoints,
        direct_power=direct_power,
        transmitter_to_panel_power=transmitter_to_panel_power,
        panel_source_batch_size=settings.panel_source_batch_size,
    )
    _write_scene_progress(
        checkpoint_directory,
        scenario_name=scenario_name,
        stage="complete",
        completed=panel_count,
        total=panel_count,
        started_at=started_at,
    )
    return read_json(summary_path)


def build_multicity_dataset(
    scenario_names: tuple[str, ...],
    *,
    output_root: Path,
    settings: MultiCitySettings,
    plan_only: bool,
) -> None:
    """Build or resume all requested scene datasets and their shared index."""
    if not scenario_names:
        raise ValueError("At least one scenario is required.")
    if settings.samples_per_source <= 0:
        raise ValueError("samples_per_source must be positive.")
    if settings.panel_source_batch_size <= 0:
        raise ValueError("panel_source_batch_size must be positive.")
    if settings.panel_target_batch_size <= 0:
        raise ValueError("panel_target_batch_size must be positive.")
    if settings.user_count <= 0:
        raise ValueError("user_count must be positive.")
    if settings.maximum_panels_per_scene <= 0:
        raise ValueError("maximum_panels_per_scene must be positive.")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    run_started_at: float = perf_counter()
    for scene_index, scenario_name in enumerate(scenario_names, start=1):
        print(f"MULTICITY_PROGRESS={scene_index}/{len(scenario_names)}:{scenario_name}")
        summary: dict[str, object] = _build_scene(
            scenario_name,
            output_root=output_root,
            settings=settings,
            plan_only=plan_only,
        )
        summaries.append(summary)
        write_json(
            output_root / ("plan.json" if plan_only else "dataset_index.json"),
            {
                "format": "IRS2 multi-city tiled panel filter collection",
                "format_version": 1,
                "plan_only": plan_only,
                "selected_scenarios": list(scenario_names),
                "settings": asdict(settings),
                "completed_scene_count": len(summaries),
                "scene_count": len(scenario_names),
                "selected_panel_count": sum(
                    cast(int, summary["selected_panel_count"]) for summary in summaries
                ),
                "total_tiled_panel_count": sum(
                    cast(int, summary["total_tiled_panel_count"]) for summary in summaries
                ),
                "elapsed_seconds": perf_counter() - run_started_at,
                "updated_at_unix": time(),
                "scenes": summaries,
            },
        )
    print(f"MULTICITY_COMPLETE={len(summaries)}")
    print(f"ELAPSED_SECONDS={perf_counter() - run_started_at:.1f}")


def _batch_path(checkpoint_directory: Path, start: int, stop: int) -> Path:
    return checkpoint_directory / f"panels_{start:05d}_{stop:05d}.npz"


def _calculate_panel_batch(
    *,
    start: int,
    stop: int,
    checkpoint_directory: Path,
    geometry: CandidateGeometry,
    scene_xml: str,
    frequency_hz: float,
    user_names: NameArray,
    user_positions: FloatArray,
    direct_power: FloatArray,
    transmitter_to_panel_power: FloatArray,
    ray_settings: RayTracingSettings,
    cell_model: TangCellModel,
    wavelength_m: float,
) -> None:
    output_path: Path = _batch_path(checkpoint_directory, start, stop)
    if output_path.exists():
        print(f"PANEL_BATCH_CACHED={start}:{stop}")
        return
    panel_names: NameArray = np.asarray(
        [f"candidate_panel_{index:05d}" for index in range(start, stop)],
        dtype=np.str_,
    )
    panel_to_user_power: FloatArray = solve_total_link_power(
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        source_names=panel_names,
        source_positions=geometry.centers_m[start:stop],
        target_names=user_names,
        target_positions=user_positions,
        settings=ray_settings,
        cell_model=cell_model,
        source_irs_normals=geometry.normals[start:stop],
        channel_amplitude_scale=cell_model.cascaded_amplitude_scale(wavelength_m),
    )
    batch_count: int = stop - start
    user_count: int = user_positions.shape[0]
    cascaded_power: FloatArray = np.zeros((batch_count, user_count), dtype=np.float32)
    margin_db: FloatArray = np.full((batch_count, user_count), -np.inf, dtype=np.float32)
    irs_path_loss_db: FloatArray = np.full((batch_count, user_count), np.inf, dtype=np.float32)
    qualifying_mask: BoolArray = np.zeros((batch_count, user_count), dtype=np.bool_)
    best_transmitter_index: IndexArray = np.zeros(batch_count, dtype=np.int64)
    panel_element_count: int = PHYSICAL_IRS_ROWS * PHYSICAL_IRS_COLUMNS
    for offset in range(batch_count):
        result: PanelPowerFilterResult = compare_panel_to_direct_power(
            direct_power=direct_power,
            transmitter_to_panel_power=transmitter_to_panel_power[start + offset],
            panel_to_receiver_power=panel_to_user_power[:, offset],
            panel_node_count=panel_element_count,
            threshold_db=RELATIVE_THRESHOLD_DB,
        )
        cascaded_power[offset] = result.cascaded_best_power
        margin_db[offset] = result.cascaded_margin_db
        positive: BoolArray = np.asarray(result.cascaded_best_power > 0.0, dtype=np.bool_)
        irs_path_loss_db[offset, positive] = np.asarray(
            -10.0 * np.log10(result.cascaded_best_power[positive]),
            dtype=np.float32,
        )
        qualifying_mask[offset] = np.asarray(
            result.keep_mask & (irs_path_loss_db[offset] <= MAXIMUM_IRS_PATH_LOSS_DB),
            dtype=np.bool_,
        )
        best_transmitter_index[offset] = result.best_panel_transmitter_index
    temporary_path: Path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        start=np.asarray(start, dtype=np.int64),
        stop=np.asarray(stop, dtype=np.int64),
        cascaded_irs_power=cascaded_power,
        irs_direct_margin_db=margin_db,
        irs_path_loss_db=irs_path_loss_db,
        qualifying_mask=qualifying_mask,
        best_transmitter_index=best_transmitter_index,
    )
    temporary_path.replace(output_path)
    print(f"PANEL_BATCH_SAVED={start}:{stop} QUALIFYING_LINKS={int(np.sum(qualifying_mask))}")
