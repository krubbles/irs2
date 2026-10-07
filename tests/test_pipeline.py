from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import pytest
import torch
from torch import nn

from irs import (
    ChannelEstimatorConfig,
    Channels,
    NeuralChannelEstimator,
    SolverConfig,
    channel_rate,
    optimize_phases,
)
from irs.data import load_pool
from irs.estimator import Lift
from irs.training import TrainingConfig, estimation_loss, train


@pytest.mark.parametrize(("pilots", "lift"), [(8, "learned"), (17, "ridge")])
def test_learned_pilots_and_inference_symmetries(dataset: Path, pilots: int, lift: str) -> None:
    pool = load_pool(dataset)
    batch = pool.sample(pool.split()["train"], 3, 2, 2, torch.Generator().manual_seed(4))
    model = NeuralChannelEstimator(ChannelEstimatorConfig(pilots=pilots, lift=cast(Lift, lift)))
    received = model.observe(batch, 0.01, torch.Generator().manual_seed(3))
    prediction: Channels = model(
        received, batch.user_mask, batch.irs_locations, batch.transmitter_locations, 0.01
    )
    coarse = model.linear(received, batch.user_mask)
    torch.testing.assert_close(prediction.cascade, coarse.cascade)
    estimation_loss(prediction, batch.channels).backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and bool(parameter.grad.isfinite().all())
    for parameter in (model.symbol_phases, model.probe_phases):
        assert parameter.grad is not None and float(parameter.grad.norm()) > 0
    torch.testing.assert_close(model.symbols().abs(), torch.ones(pilots))
    torch.testing.assert_close(model.probes().abs(), torch.ones(pilots, 16))
    with torch.no_grad():
        nn.init.normal_(cast(nn.Conv2d, model.fusion[-1]).weight, std=0.01)
        original: Channels = model(
            received, batch.user_mask, batch.irs_locations, batch.transmitter_locations, 0.01
        )
        permuted: Channels = model(
            received[:, [1, 0]][:, :, [1, 0]],
            batch.user_mask[:, [1, 0]],
            batch.irs_locations,
            batch.transmitter_locations[:, [1, 0]],
            0.01,
        )
        torch.testing.assert_close(
            permuted.cascade, original.cascade[:, [1, 0]][:, :, [1, 0]], rtol=1e-4, atol=1e-10
        )
        rotation = torch.polar(torch.ones((3, 2, 1, 1)), torch.rand((3, 2, 1, 1)) * 6)
        rotated: Channels = model(
            received * rotation,
            batch.user_mask,
            batch.irs_locations,
            batch.transmitter_locations,
            0.01,
        )
        torch.testing.assert_close(
            rotated.cascade, original.cascade * rotation, rtol=1e-4, atol=1e-10
        )


def test_solver_is_deterministic_detaches_inputs_and_preserves_identity(dataset: Path) -> None:
    pool = load_pool(dataset)
    channels = pool.sample(
        pool.split()["train"], 4, 2, 2, torch.Generator().manual_seed(6)
    ).channels
    channels.direct.requires_grad_(True)
    channels.cascade.requires_grad_(True)
    config = SolverConfig(steps=5, restarts=2)
    with torch.no_grad():
        phases = optimize_phases(channels, config)
    torch.testing.assert_close(phases, optimize_phases(channels, config), rtol=0, atol=0)
    assert channels.direct.grad is None and channels.cascade.grad is None
    assert not phases.requires_grad and bool(phases.isfinite().all())
    assert bool(
        (channel_rate(channels, phases) >= channel_rate(channels, torch.zeros_like(phases))).all()
    )
    torch.testing.assert_close(
        channel_rate(channels, phases), channel_rate(channels, phases, estimated=channels)
    )


def test_rzf_masks_padded_users(dataset: Path) -> None:
    pool = load_pool(dataset)
    channels = pool.sample(
        pool.split()["train"], 2, 2, 4, torch.Generator().manual_seed(1)
    ).channels
    phases = torch.rand((2, 16))
    padded = Channels(
        torch.cat((channels.direct, torch.ones_like(channels.direct[:, :1]) * 100), 1),
        torch.cat((channels.cascade, torch.ones_like(channels.cascade[:, :1]) * 100), 1),
        torch.cat((channels.mask, torch.zeros((2, 1), dtype=torch.bool)), 1),
    )
    torch.testing.assert_close(
        channel_rate(padded, phases), channel_rate(channels, phases), rtol=1e-4, atol=1e-5
    )


def test_checkpoint_resume_matches_uninterrupted_training(dataset: Path, tmp_path: Path) -> None:
    config = TrainingConfig(
        model=ChannelEstimatorConfig(width=8, blocks=1, lift="ridge"),
        counts=((2, 2),),
        steps=2,
        batch_size=2,
        evaluate_every=1,
        validation_scenes=2,
        validation_solver=SolverConfig(steps=2, restarts=1),
    )
    device = torch.device("cpu")
    train(dataset, tmp_path / "whole", config, device)
    train(dataset, tmp_path / "resumed", replace(config, steps=1), device)
    train(dataset, tmp_path / "resumed", config, device, resume=True)
    whole = torch.load(tmp_path / "whole/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    for name, value in whole["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
    for name in ("generator", "noise_generator", "torch_rng"):
        torch.testing.assert_close(whole[name], resumed[name], rtol=0, atol=0)
    assert whole["best_gain"] == resumed["best_gain"]
    with pytest.raises(ValueError, match="Run directory exists"):
        train(dataset, tmp_path / "resumed", config, device)
    with pytest.raises(ValueError, match="Resume configuration"):
        train(dataset, tmp_path / "resumed", replace(config, noise=0.03), device, resume=True)


class SolverOverrides(TypedDict, total=False):
    steps: int
    restarts: int
    learning_rate: float


@pytest.mark.parametrize("config", [{"steps": 0}, {"restarts": 0}, {"learning_rate": float("nan")}])
def test_reject_invalid_solver_config(config: SolverOverrides) -> None:
    with pytest.raises(ValueError):
        SolverConfig(**config)
