"""Physical constants and runtime settings of the current multi-city generator."""

from __future__ import annotations

from dataclasses import dataclass

RAY_TRACING_SEED: int = 20260902
RELATIVE_THRESHOLD_DB: float = -15.0
MAXIMUM_IRS_PATH_LOSS_DB: float = 100.0
PHYSICAL_IRS_ROWS: int = 32
PHYSICAL_IRS_COLUMNS: int = 32
PHYSICAL_CELL_SIZE_M: float = 0.02
IRS_REFLECTION_AMPLITUDE: float = 0.9
IRS_PATTERN_EXPONENT: float = 3.0
CANDIDATE_ROWS: int = 8
CANDIDATE_COLUMNS: int = 8
CANDIDATE_SPACING_M: float = 0.25
CANDIDATE_PANEL_GAP_M: float = 0.15
CANDIDATE_WALL_MARGIN_M: float = 0.25
CANDIDATE_WALL_OFFSET_M: float = 0.04

IRS_ROWS: int = 4
IRS_COLUMNS: int = 4
IRS_APERTURE_M: float = 0.62
IRS_CELL_SIZE_M: float = 0.02
MIMO_ANTENNA_COORDINATES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class MaterializationSettings:
    """Ray-tracing and checkpoint controls."""

    samples_per_source: int = 50_000
    panel_batch_size: int = 8
    tx_target_panel_batch_size: int = 32

    def __post_init__(self) -> None:
        if (
            min(self.samples_per_source, self.panel_batch_size, self.tx_target_panel_batch_size)
            <= 0
        ):
            raise ValueError("Materialization settings must be positive")


SELECTED_SCENARIOS: tuple[str, ...] = (
    "city_0_newyork_3p5_s",
    "city_1_losangeles_3p5_s",
    "city_2_chicago_3p5_s",
    "city_3_houston_3p5_s",
    "city_4_phoenix_3p5_s",
    "city_5_philadelphia_3p5_s",
    "city_6_miami_3p5_s",
    "city_7_sandiego_3p5_s",
    "city_8_dallas_3p5_s",
    "city_9_sanfrancisco_3p5_s",
    "city_10_austin_3p5_s",
    "city_12_fortworth_3p5_s",
    "city_13_columbus_3p5_s",
    "city_14_charlotte_3p5_s",
    "city_15_indianapolis_3p5_s",
    "city_16_sanfrancisco_3p5_s",
    "city_17_seattle_3p5_s",
    "city_18_denver_3p5_s",
    "city_19_oklahoma_3p5_s",
)
MAXIMUM_USER_HEIGHT_M: float = 5.0
USER_EDGE_BUFFER_M: float = 5.0
DEFAULT_USER_COUNT: int = 512
DEFAULT_MAXIMUM_PANELS_PER_SCENE: int = 10_000
SELECTION_SEED: int = 20260913


@dataclass(frozen=True)
class MultiCitySettings:
    """Controls shared by every scene in the multi-city run."""

    samples_per_source: int = 50_000
    panel_source_batch_size: int = 16
    panel_target_batch_size: int = 512
    user_count: int = DEFAULT_USER_COUNT
    maximum_panels_per_scene: int = DEFAULT_MAXIMUM_PANELS_PER_SCENE
    selection_seed: int = SELECTION_SEED

    def __post_init__(self) -> None:
        if (
            min(
                self.samples_per_source,
                self.panel_source_batch_size,
                self.panel_target_batch_size,
                self.user_count,
                self.maximum_panels_per_scene,
            )
            <= 0
        ):
            raise ValueError("Dataset generation sizes must be positive")
