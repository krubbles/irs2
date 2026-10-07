from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from irs.generation.cell import TangCellModel, arrival_amplitude_weights
from irs.io import file_hash, write_json


def test_cell_pattern_is_one_sided_and_normalization_is_applied_once() -> None:
    model = TangCellModel(0.02, 0.02)
    np.testing.assert_allclose(
        model.path_amplitude_from_cosine(np.array([-1, 0, 1], dtype=np.float32)), [0, 0, 1]
    )
    # At broadside F=1; opposite-facing normals give zero arrival amplitude.
    theta = np.full((2, 1, 1), np.pi / 2, dtype=np.float32)
    phi = np.zeros_like(theta)
    normals = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(
        arrival_amplitude_weights(theta, phi, normals, model).flatten(), [1, 0]
    )
    assert model.cascaded_amplitude_scale(0.1) == pytest.approx(
        0.9 * 2 * np.sqrt(np.pi * 8 * 0.02**2) / 0.1
    )


def test_completed_generator_stages_check_settings_and_source(tmp_path: Path) -> None:
    pytest.importorskip("deepmimo")
    pytest.importorskip("sionna.rt")
    from irs.generation.config import MaterializationSettings, MultiCitySettings
    from irs.generation.filtering import _build_scene, _stable_seed
    from irs.generation.materialize import _materialize_scene

    scenario = "city_0_newyork_3p5_s"
    settings = MultiCitySettings()
    filter_path = tmp_path / "filter/scenes" / f"{scenario}.npz"
    filter_path.parent.mkdir(parents=True)
    filter_path.write_bytes(b"cached filter")
    summary: dict[str, object] = {
        "user_count": settings.user_count,
        "maximum_panels_per_scene": settings.maximum_panels_per_scene,
        "panel_selection_seed": _stable_seed(settings.selection_seed, scenario, "panels"),
        "user_selection_seed": _stable_seed(settings.selection_seed, scenario, "users"),
        "ray_tracing": {
            "samples_per_source": 50000,
            "panel_source_batch_size": 16,
            "panel_target_batch_size": 512,
        },
        "ray_tracing_seed": 20260902,
        "relative_threshold_db": -15.0,
        "maximum_irs_path_loss_db": 100.0,
    }
    write_json(filter_path.with_suffix(".json"), summary)
    assert (
        _build_scene(scenario, output_root=tmp_path / "filter", settings=settings, plan_only=False)
        == summary
    )
    with pytest.raises(ValueError, match="settings differ"):
        _build_scene(
            scenario,
            output_root=tmp_path / "filter",
            settings=replace(settings, user_count=10),
            plan_only=False,
        )

    material = MaterializationSettings()
    path = tmp_path / "channels/scenes" / f"{scenario}.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cached channels")
    saved = {"settings": asdict(material), "source_filter_sha256": file_hash(filter_path)}
    write_json(path.with_suffix(".json"), saved)
    assert (
        _materialize_scene(
            scenario, filter_path=filter_path, output_root=tmp_path / "channels", settings=material
        )
        == saved
    )
    filter_path.write_bytes(b"changed filter")
    with pytest.raises(ValueError, match="source differ"):
        _materialize_scene(
            scenario, filter_path=filter_path, output_root=tmp_path / "channels", settings=material
        )


def test_combined_pool_fingerprint_is_stable_on_resume(dataset: Path, tmp_path: Path) -> None:
    pytest.importorskip("deepmimo")
    pytest.importorskip("sionna.rt")
    from irs.generation.config import MaterializationSettings
    from irs.generation.materialize import _combine_scene_pools

    root = tmp_path / "materialized"
    (root / "scenes").mkdir(parents=True)
    (root / "scenes/city.npz").write_bytes(dataset.read_bytes())
    output = tmp_path / "combined.npz"
    _combine_scene_pools(
        ("city",), output_root=root, combined_output=output, settings=MaterializationSettings()
    )
    fingerprint = file_hash(output)
    _combine_scene_pools(
        ("city",), output_root=root, combined_output=output, settings=MaterializationSettings()
    )
    assert file_hash(output) == fingerprint
