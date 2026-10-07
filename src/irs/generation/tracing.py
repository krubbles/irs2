"""One coherent Sionna tracing path shared by power filtering and materialization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import mitsuba as mi
import numpy as np
import numpy.typing as npt
from sionna.rt import (
    Paths,
    PathSolver,
    PlanarArray,
    Receiver,
    Scene,
    Transmitter,
    load_scene_from_string,
)

from .cell import TangCellModel, arrival_amplitude_weights, departure_amplitude_weights
from .config import RAY_TRACING_SEED

FloatArray = npt.NDArray[np.float32]
ComplexArray = npt.NDArray[np.complex64]
BoolArray = npt.NDArray[np.bool_]
NameArray = npt.NDArray[np.str_]


@dataclass(frozen=True)
class RayTracingSettings:
    max_depth: int = 3
    samples_per_src: int = 50_000
    seed: int = RAY_TRACING_SEED


@dataclass(frozen=True)
class PanelPowerFilterResult:
    cascaded_best_power: FloatArray
    cascaded_margin_db: FloatArray
    keep_mask: BoolArray
    best_panel_transmitter_index: int


def solve_complex_links(
    *,
    scene_xml: str,
    frequency_hz: float,
    source_names: NameArray,
    source_positions: FloatArray,
    target_names: NameArray,
    target_positions: FloatArray,
    samples_per_source: int,
    cell_model: TangCellModel | None = None,
    source_irs_normals: FloatArray | None = None,
    target_irs_normals: FloatArray | None = None,
    amplitude_scale: float = 1.0,
    max_depth: int = 3,
    seed: int = RAY_TRACING_SEED,
) -> ComplexArray:
    if frequency_hz <= 0 or samples_per_source <= 0 or amplitude_scale < 0:
        raise ValueError("Invalid tracing frequency, sample count, or amplitude")
    if source_irs_normals is not None and target_irs_normals is not None:
        raise ValueError("Only one endpoint can represent the IRS")
    for names, positions, normals in (
        (source_names, source_positions, source_irs_normals),
        (target_names, target_positions, target_irs_normals),
    ):
        if positions.shape != (names.size, 3):
            raise ValueError("Endpoint positions must be [count,3]")
        if normals is not None and (normals.shape != positions.shape or cell_model is None):
            raise ValueError("IRS normals must align with endpoints and need a cell model")
    scene: Scene = load_scene_from_string(scene_xml, merge_shapes=False)
    scene.frequency = frequency_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    for name, position in zip(target_names, target_positions, strict=True):
        scene.add(Receiver(name=str(name), position=mi.Point3f(*map(float, position))))
    for name, position in zip(source_names, source_positions, strict=True):
        scene.add(Transmitter(name=str(name), position=mi.Point3f(*map(float, position))))
    started = perf_counter()
    paths: Paths = PathSolver()(
        scene,
        max_depth=max_depth,
        max_num_paths_per_src=100_000,
        samples_per_src=samples_per_source,
        synthetic_array=True,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        seed=seed,
    )
    coefficients, _ = paths.cir(normalize_delays=False, out_type="numpy")
    values: ComplexArray = np.asarray(coefficients[:, 0, :, 0, :, 0], dtype=np.complex64)
    if target_irs_normals is not None and cell_model is not None:
        values *= arrival_amplitude_weights(
            np.asarray(paths.theta_r.numpy(), dtype=np.float32),
            np.asarray(paths.phi_r.numpy(), dtype=np.float32),
            target_irs_normals,
            cell_model,
        )
    elif source_irs_normals is not None and cell_model is not None:
        values *= departure_amplitude_weights(
            np.asarray(paths.theta_t.numpy(), dtype=np.float32),
            np.asarray(paths.phi_t.numpy(), dtype=np.float32),
            source_irs_normals,
            cell_model,
        )
    channel: ComplexArray = np.asarray(
        amplitude_scale * values.sum(-1, dtype=np.complex64), dtype=np.complex64
    )
    print(
        f"Traced {source_names.size} sources -> {target_names.size} targets in {perf_counter() - started:.1f}s",
        flush=True,
    )
    return channel


def solve_total_link_power(
    *,
    scene_xml: str,
    frequency_hz: float,
    source_names: NameArray,
    source_positions: FloatArray,
    target_names: NameArray,
    target_positions: FloatArray,
    settings: RayTracingSettings,
    cell_model: TangCellModel | None = None,
    source_irs_normals: FloatArray | None = None,
    target_irs_normals: FloatArray | None = None,
    channel_amplitude_scale: float = 1.0,
) -> FloatArray:
    channel = solve_complex_links(
        scene_xml=scene_xml,
        frequency_hz=frequency_hz,
        source_names=source_names,
        source_positions=source_positions,
        target_names=target_names,
        target_positions=target_positions,
        samples_per_source=settings.samples_per_src,
        max_depth=settings.max_depth,
        seed=settings.seed,
        cell_model=cell_model,
        source_irs_normals=source_irs_normals,
        target_irs_normals=target_irs_normals,
        amplitude_scale=channel_amplitude_scale,
    )
    return np.asarray(np.abs(channel) ** 2, dtype=np.float32)


def compare_panel_to_direct_power(
    *,
    direct_power: FloatArray,
    transmitter_to_panel_power: FloatArray,
    panel_to_receiver_power: FloatArray,
    panel_node_count: int,
    threshold_db: float,
) -> PanelPowerFilterResult:
    """The current ideal coherent panel filter, relative to the best direct site."""
    site = int(np.argmax(transmitter_to_panel_power))
    direct: FloatArray = direct_power.max(1)
    cascade: FloatArray = np.asarray(
        float(transmitter_to_panel_power[site]) * panel_to_receiver_power * panel_node_count**2,
        dtype=np.float32,
    )
    margin: FloatArray = np.full(direct.shape, -np.inf, dtype=np.float32)
    positive = (cascade > 0) & (direct > 0)
    margin[positive] = 10 * np.log10(cascade[positive] / direct[positive])
    margin[(cascade > 0) & (direct == 0)] = np.inf
    keep: BoolArray = (cascade > 0) & (margin >= threshold_db)
    return PanelPowerFilterResult(cascade, margin, keep, site)
