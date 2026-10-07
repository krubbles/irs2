"""Wall tiling and aligned physical antenna/IRS coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import numpy.typing as npt
from deepmimo.core.scene import BoundingBox, Face, PhysicalElement
from deepmimo.core.scene import Scene as DeepMIMOScene
from scipy.spatial import ConvexHull

from .config import IRS_APERTURE_M, IRS_COLUMNS, IRS_ROWS, MIMO_ANTENNA_COORDINATES

FloatArray = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]
NameArray = npt.NDArray[np.str_]
Vector3 = tuple[float, float, float]
TX_ROWS: int = 8
TX_COLUMNS: int = 8
TX_ELEMENT_SPACING_M: float = 0.020


@dataclass(frozen=True)
class IRSPanel:
    """One wall-mounted IRS represented by co-located TX/RX probe nodes."""

    name: str
    building_name: str
    rows: int
    columns: int
    spacing_m: float
    normal: Vector3
    positions: tuple[Vector3, ...]


@dataclass(frozen=True)
class _WallPlacement:
    building: PhysicalElement
    face: Face
    center: FloatArray
    horizontal_direction: FloatArray
    outward_normal: FloatArray
    distance_to_preview_center_m: float


@dataclass(frozen=True)
class _WallCandidate:
    """Suitable wall geometry before a vertical panel position is chosen."""

    building: PhysicalElement
    face: Face
    face_center: FloatArray
    horizontal_direction: FloatArray
    outward_normal: FloatArray
    horizontal_min_m: float
    horizontal_max_m: float
    horizontal_center_offset_m: float
    wall_bottom_z: float
    wall_top_z: float
    distance_to_preview_center_m: float
    facing_score: float


def create_tiled_wall_irs_panels(
    deepmimo_scene: DeepMIMOScene,
    *,
    rows: int,
    columns: int,
    spacing_m: float,
    panel_gap_m: float = 0.10,
    mounting_margin_m: float = 0.15,
    wall_offset_m: float = 0.02,
) -> tuple[IRSPanel, ...]:
    """Pack identical IRS grids across every suitable building wall.

    Horizontal and vertical tile counts are chosen independently for each
    facade. Any unused space is split evenly around the packed grid, so the
    panels are centered on the wall. Roofs, floors, and faces too small to
    contain one complete panel are skipped.

    """
    if rows <= 1 or columns <= 1:
        raise ValueError("rows and columns must both exceed one.")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive.")
    if panel_gap_m < 0.0:
        raise ValueError("panel_gap_m cannot be negative.")
    if mounting_margin_m < 0.0:
        raise ValueError("mounting_margin_m cannot be negative.")
    if wall_offset_m < 0.0:
        raise ValueError("wall_offset_m cannot be negative.")

    panel_width_m: float = (columns - 1) * spacing_m
    panel_height_m: float = (rows - 1) * spacing_m
    scene_bounds: BoundingBox = deepmimo_scene.bounding_box
    scene_center: tuple[float, float] = (
        float((scene_bounds.x_min + scene_bounds.x_max) / 2.0),
        float((scene_bounds.y_min + scene_bounds.y_max) / 2.0),
    )

    panels: list[IRSPanel] = []
    for building in deepmimo_scene.objects:
        if building.label != "buildings":
            continue
        candidates: tuple[_WallCandidate, ...] = _wall_candidates(
            building,
            preview_center=scene_center,
            panel_width_m=panel_width_m,
            panel_height_m=panel_height_m,
            mounting_margin_m=mounting_margin_m,
        )
        for candidate in candidates:
            horizontal_centers: tuple[float, ...] = _packed_centers(
                minimum_m=candidate.horizontal_min_m,
                maximum_m=candidate.horizontal_max_m,
                aperture_m=panel_width_m,
                gap_m=panel_gap_m,
                margin_m=mounting_margin_m,
            )
            vertical_centers: tuple[float, ...] = _packed_centers(
                minimum_m=candidate.wall_bottom_z,
                maximum_m=candidate.wall_top_z,
                aperture_m=panel_height_m,
                gap_m=panel_gap_m,
                margin_m=mounting_margin_m,
            )
            for vertical_center_m in vertical_centers:
                for horizontal_center_m in horizontal_centers:
                    centered_candidate = _WallCandidate(
                        building=candidate.building,
                        face=candidate.face,
                        face_center=candidate.face_center,
                        horizontal_direction=candidate.horizontal_direction,
                        outward_normal=candidate.outward_normal,
                        horizontal_min_m=candidate.horizontal_min_m,
                        horizontal_max_m=candidate.horizontal_max_m,
                        horizontal_center_offset_m=horizontal_center_m,
                        wall_bottom_z=candidate.wall_bottom_z,
                        wall_top_z=candidate.wall_top_z,
                        distance_to_preview_center_m=(candidate.distance_to_preview_center_m),
                        facing_score=candidate.facing_score,
                    )
                    placement: _WallPlacement = _placement_from_candidate(
                        centered_candidate,
                        panel_center_z=vertical_center_m,
                    )
                    panels.append(
                        _create_panel(
                            panel_index=len(panels),
                            placement=placement,
                            rows=rows,
                            columns=columns,
                            spacing_m=spacing_m,
                            wall_offset_m=wall_offset_m,
                        )
                    )
    if not panels:
        raise ValueError("No building wall can contain a complete IRS panel.")
    return tuple(panels)


def _wall_candidates(
    building: PhysicalElement,
    *,
    preview_center: tuple[float, float],
    panel_width_m: float,
    panel_height_m: float,
    mounting_margin_m: float,
) -> tuple[_WallCandidate, ...]:
    """Return every suitable vertical wall on one building."""
    suitable_walls: list[_WallCandidate] = []
    up: FloatArray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    footprint_hull: ConvexHull = ConvexHull(np.asarray(building.vertices[:, :2], dtype=np.float32))
    default_camera_direction: FloatArray = _normalized(np.array([1.0, 1.0, 0.0], dtype=np.float32))

    for face in building.faces:
        # Some DeepMIMO city meshes contain degenerate line-segment faces.
        # They have no surface normal and cannot support an IRS panel.
        if face.vertices.shape[0] < 3:
            continue
        normal: FloatArray = np.asarray(face.normal, dtype=np.float32)
        if abs(float(normal[2])) > 0.15:
            continue
        face_center: FloatArray = np.asarray(face.centroid, dtype=np.float32)
        normal = _outward_normal(
            normal,
            face_center=face_center,
            footprint_hull=footprint_hull,
        )
        horizontal_direction: FloatArray = _normalized(np.cross(up, normal))
        relative_vertices: FloatArray = np.asarray(face.vertices, dtype=np.float32) - face_center
        horizontal_coordinates: FloatArray = relative_vertices @ horizontal_direction
        vertical_coordinates: FloatArray = relative_vertices @ up
        wall_width_m: float = float(np.ptp(horizontal_coordinates))
        wall_height_m: float = float(np.ptp(vertical_coordinates))
        if (
            wall_width_m < panel_width_m + 2.0 * mounting_margin_m
            or wall_height_m < panel_height_m + 2.0 * mounting_margin_m
        ):
            continue

        horizontal_center_offset_m: float = float(
            (np.min(horizontal_coordinates) + np.max(horizontal_coordinates)) / 2.0
        )
        direction_to_preview_center: FloatArray = np.array(
            [
                preview_center[0] - float(face_center[0]),
                preview_center[1] - float(face_center[1]),
                0.0,
            ],
            dtype=np.float32,
        )
        suitable_walls.append(
            _WallCandidate(
                building=building,
                face=face,
                face_center=face_center,
                horizontal_direction=horizontal_direction,
                outward_normal=normal,
                horizontal_min_m=float(np.min(horizontal_coordinates)),
                horizontal_max_m=float(np.max(horizontal_coordinates)),
                horizontal_center_offset_m=horizontal_center_offset_m,
                wall_bottom_z=float(np.min(face.vertices[:, 2])),
                wall_top_z=float(np.max(face.vertices[:, 2])),
                distance_to_preview_center_m=_length(direction_to_preview_center),
                facing_score=float(np.dot(normal, default_camera_direction)),
            )
        )
    return tuple(suitable_walls)


def _packed_centers(
    *,
    minimum_m: float,
    maximum_m: float,
    aperture_m: float,
    gap_m: float,
    margin_m: float,
) -> tuple[float, ...]:
    """Return centered panel coordinates that fit inside one wall axis."""
    usable_span_m: float = maximum_m - minimum_m - 2.0 * margin_m
    if usable_span_m + 1e-6 < aperture_m:
        return ()
    count: int = int((usable_span_m + gap_m + 1e-6) // (aperture_m + gap_m))
    occupied_span_m: float = count * aperture_m + (count - 1) * gap_m
    first_center_m: float = (minimum_m + maximum_m - occupied_span_m) / 2.0 + aperture_m / 2.0
    step_m: float = aperture_m + gap_m
    return tuple(first_center_m + index * step_m for index in range(count))


def _placement_from_candidate(
    candidate: _WallCandidate,
    *,
    panel_center_z: float,
) -> _WallPlacement:
    """Resolve a wall candidate at a selected center height."""
    up: FloatArray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    panel_center: FloatArray = (
        candidate.face_center
        + candidate.horizontal_center_offset_m * candidate.horizontal_direction
        + (panel_center_z - float(candidate.face_center[2])) * up
    )
    return _WallPlacement(
        building=candidate.building,
        face=candidate.face,
        center=panel_center,
        horizontal_direction=candidate.horizontal_direction,
        outward_normal=candidate.outward_normal,
        distance_to_preview_center_m=candidate.distance_to_preview_center_m,
    )


def _create_panel(
    *,
    panel_index: int,
    placement: _WallPlacement,
    rows: int,
    columns: int,
    spacing_m: float,
    wall_offset_m: float,
) -> IRSPanel:
    """Create one row-major IRS element grid for a resolved wall placement."""
    positions: list[Vector3] = []
    for row in range(rows):
        vertical_offset_m: float = (row - (rows - 1) / 2.0) * spacing_m
        for column in range(columns):
            horizontal_offset_m: float = (column - (columns - 1) / 2.0) * spacing_m
            position: FloatArray = (
                placement.center
                + horizontal_offset_m * placement.horizontal_direction
                + vertical_offset_m * np.array([0.0, 0.0, 1.0], dtype=np.float32)
                + wall_offset_m * placement.outward_normal
            )
            positions.append((float(position[0]), float(position[1]), float(position[2])))
    return IRSPanel(
        name=f"irs_panel_{panel_index:02d}",
        building_name=placement.building.name,
        rows=rows,
        columns=columns,
        spacing_m=spacing_m,
        normal=(
            float(placement.outward_normal[0]),
            float(placement.outward_normal[1]),
            float(placement.outward_normal[2]),
        ),
        positions=tuple(positions),
    )


def _normalized(vector: FloatArray) -> FloatArray:
    """Return a normalized float32 vector."""
    length: float = _length(vector)
    if length == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return np.asarray(vector / length, dtype=np.float32)


def _outward_normal(
    normal: FloatArray,
    *,
    face_center: FloatArray,
    footprint_hull: ConvexHull,
) -> FloatArray:
    """Orient a wall normal toward the exterior of a convex building footprint."""
    normalized_normal: FloatArray = _normalized(normal)
    horizontal_normal: FloatArray = _normalized(
        np.array(
            [normalized_normal[0], normalized_normal[1], 0.0],
            dtype=np.float32,
        )
    )
    probe_distance_m: float = 0.10
    positive_probe: FloatArray = face_center[:2] + probe_distance_m * horizontal_normal[:2]
    negative_probe: FloatArray = face_center[:2] - probe_distance_m * horizontal_normal[:2]
    positive_exterior_score: float = _footprint_exterior_score(
        positive_probe,
        footprint_hull,
    )
    negative_exterior_score: float = _footprint_exterior_score(
        negative_probe,
        footprint_hull,
    )
    return (
        horizontal_normal
        if positive_exterior_score >= negative_exterior_score
        else -horizontal_normal
    )


def _footprint_exterior_score(point: FloatArray, footprint_hull: ConvexHull) -> float:
    """Return a positive score outside a convex footprint and non-positive inside."""
    equations: npt.NDArray[np.float64] = np.asarray(
        footprint_hull.equations,
        dtype=np.float64,
    )
    point_64: npt.NDArray[np.float64] = np.asarray(point, dtype=np.float64)
    signed_distances: npt.NDArray[np.float64] = equations[:, :2] @ point_64 + equations[:, 2]
    return float(np.max(signed_distances))


def _length(vector: FloatArray) -> float:
    """Return a vector's Euclidean length."""
    return sqrt(float(np.dot(vector, vector)))


@dataclass(frozen=True)
class CandidateGeometry:
    """Aligned tiled candidate-panel metadata."""

    indices: IndexArray
    names: NameArray
    building_names: NameArray
    centers_m: FloatArray
    normals: FloatArray


def horizontal_array_offsets(
    *,
    rows: int,
    columns: int,
    spacing_m: float,
) -> FloatArray:
    """Return Sionna-ordered offsets for a horizontal XY planar array.

    The ordering and axes match a Sionna ``PlanarArray`` rotated by -90 degrees
    about y: columns are outermost, rows are innermost, rows advance along +x,
    and columns advance along +y.
    """
    if rows <= 1 or columns <= 1:
        raise ValueError("rows and columns must both exceed one.")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive.")

    offsets: FloatArray = np.zeros((rows * columns, 3), dtype=np.float32)
    for column in range(columns):
        for row in range(rows):
            element_index: int = column * rows + row
            offsets[element_index] = (
                (row - (rows - 1) / 2.0) * spacing_m,
                (column - (columns - 1) / 2.0) * spacing_m,
                0.0,
            )
    return offsets


def irs_element_positions(centers: FloatArray, normals: FloatArray) -> FloatArray:
    """Return 4x4 element locations spanning the current 0.62 m aperture."""
    horizontal: FloatArray = np.asarray(
        np.stack((-normals[:, 1], normals[:, 0], np.zeros(normals.shape[0])), axis=1),
        dtype=np.float32,
    )
    column_offsets: FloatArray = np.linspace(
        -IRS_APERTURE_M / 2.0,
        IRS_APERTURE_M / 2.0,
        IRS_COLUMNS,
        dtype=np.float32,
    )
    row_offsets: FloatArray = np.linspace(
        IRS_APERTURE_M / 2.0,
        -IRS_APERTURE_M / 2.0,
        IRS_ROWS,
        dtype=np.float32,
    )
    result: FloatArray = np.zeros((centers.shape[0], IRS_ROWS * IRS_COLUMNS, 3), dtype=np.float32)
    element: int = 0
    for row_offset in row_offsets:
        for column_offset in column_offsets:
            result[:, element] = centers + column_offset * horizontal
            result[:, element, 2] += row_offset
            element += 1
    return result


def transmitter_element_positions(centers: FloatArray) -> FloatArray:
    """Return the same four entries of the 8x8 array as the existing pool."""
    offsets: FloatArray = (
        horizontal_array_offsets(
            rows=TX_ROWS,
            columns=TX_COLUMNS,
            spacing_m=TX_ELEMENT_SPACING_M,
        )
        .reshape(TX_COLUMNS, TX_ROWS, 3)
        .transpose(1, 0, 2)
    )
    return np.asarray(
        [
            [centers[site] + offsets[row, column] for row, column in MIMO_ANTENNA_COORDINATES]
            for site in range(centers.shape[0])
        ],
        dtype=np.float32,
    )
