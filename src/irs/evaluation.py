"""Held-out scoring; physical CSI supplies labels/references, never estimator inputs."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from .channels import Channels, channel_rate
from .data import ChannelBatch, ChannelPool
from .estimator import NeuralChannelEstimator
from .io import JsonObject, save_npz, write_json
from .solver import SolverConfig, optimize_phases


def channel_errors(predicted: Channels, target: Channels) -> dict[str, float]:
    cascade_error = (predicted.cascade - target.cascade).abs().square().sum((2, 3))
    cascade_power = target.cascade.abs().square().sum((2, 3))
    direct_error = (predicted.direct - target.direct).abs().square().sum(2)
    direct_power = target.direct.abs().square().sum(2)
    per_user = cascade_error[target.mask] / cascade_power[target.mask].clamp_min(1e-30)
    return {
        "cascade_global_nmse": float(
            cascade_error.sum().double() / cascade_power.sum().double().clamp_min(1e-30)
        ),
        "cascade_median_user_nmse": float(per_user.median()),
        "cascade_mean_user_nmse": float(per_user.mean()),
        "direct_global_nmse": float(
            direct_error.sum().double() / direct_power.sum().double().clamp_min(1e-30)
        ),
    }


@torch.no_grad()
def predict(
    model: NeuralChannelEstimator,
    examples: ChannelBatch,
    device: torch.device,
    noise: float,
    seed: int = 20261012,
    ridge: float | None = None,
) -> Channels:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    chunks: list[Channels] = []
    for start in range(0, examples.direct.shape[0], 128):
        batch = examples.slice(start, start + 128, device)
        received = model.observe(batch, noise, generator)
        channels: Channels
        if ridge is None:
            channels = model(
                received, batch.user_mask, batch.irs_locations, batch.transmitter_locations, noise
            )
        else:
            channels = model.linear(received, batch.user_mask, ridge)
        chunks.append(channels.slice(0, channels.direct.shape[0]))
    return Channels(
        torch.cat([c.direct for c in chunks]),
        torch.cat([c.cascade for c in chunks]),
        torch.cat([c.mask for c in chunks]),
    )


def optimize_and_score(
    estimated: Channels, truth: Channels, device: torch.device, solver: SolverConfig
) -> tuple[Tensor, Tensor]:
    rates: list[Tensor] = []
    phases: list[Tensor] = []
    for start in range(0, estimated.direct.shape[0], 128):
        observed = estimated.slice(start, start + 128, device)
        phase = optimize_phases(observed, replace(solver, seed=solver.seed + start))
        with torch.no_grad():
            rates.append(channel_rate(truth.slice(start, start + 128, device), phase).cpu())
        phases.append(phase.cpu())
    return torch.cat(rates), torch.cat(phases)


def validate(
    model: NeuralChannelEstimator,
    examples: ChannelBatch,
    device: torch.device,
    noise: float,
    solver: SolverConfig,
) -> dict[str, float]:
    truth = examples.channels
    estimated = predict(model, examples, device, noise)
    rates, phases = optimize_and_score(estimated, truth, device, solver)
    with torch.no_grad():
        identity = channel_rate(truth, torch.zeros_like(phases))
    result = channel_errors(estimated, truth)
    result.update(
        gain=float((rates.double() - identity.double()).mean()), rate=float(rates.double().mean())
    )
    return result


def capture(
    gain: Tensor, reference_gain: Tensor, panels: Tensor
) -> tuple[float | None, list[float | None]]:
    """Ratio of summed gains; bootstrap globally unique physical-panel rows."""
    denominator = float(reference_gain.double().sum())
    if denominator <= 1e-30:
        return None, [None, None]
    _, inverse = panels.unique(return_inverse=True)
    count = int(inverse.max()) + 1
    sums = torch.bincount(inverse, weights=gain.double(), minlength=count)
    denominators = torch.bincount(inverse, weights=reference_gain.double(), minlength=count)
    indices = torch.randint(count, (1000, count), generator=torch.Generator().manual_seed(17))
    totals = denominators[indices].sum(-1)
    bootstrap = 100 * sums[indices].sum(-1) / totals.clamp_min(1e-30)
    valid = bootstrap[totals > 1e-30]
    interval: list[float | None] = [None, None]
    if valid.numel():
        interval = cast(
            list[float | None],
            torch.quantile(valid, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
        )
    return 100 * float(gain.double().sum()) / denominator, interval


def evaluate(
    model: NeuralChannelEstimator,
    pool: ChannelPool,
    counts: tuple[tuple[int, int], ...],
    scenes: int,
    noise: float,
    device: torch.device,
    solver: SolverConfig,
    output: Path,
) -> JsonObject:
    if scenes <= 0 or noise < 0 or not torch.isfinite(torch.tensor(noise)):
        raise ValueError("Evaluation scenes must be positive and noise finite/nonnegative")
    if pool.elements != model.config.elements:
        raise ValueError("Dataset and estimator element counts differ")
    if not counts:
        raise ValueError("Evaluation needs at least one user/antenna count")
    splits = pool.split()
    for name in ("validation", "test"):
        for users, antennas in counts:
            if (
                users <= 0
                or users > int(pool.counts[splits[name]].max())
                or not 1 <= antennas <= pool.antennas
            ):
                raise ValueError(
                    f"The {name} split cannot support {users} users / {antennas} antennas"
                )
    users, antennas = counts[0]
    validation = pool.sample(
        splits["validation"],
        min(scenes, 512),
        users,
        antennas,
        torch.Generator().manual_seed(20261003),
    )
    # Retain the useful same-bank linear baseline. Tune only on validation.
    ridge_rates: dict[float, float] = {}
    validation_solver = SolverConfig(steps=50, restarts=1)
    for ridge in (1e-6, 1e-3, 0.03, 0.3, 3.0):
        estimate = predict(model, validation, device, noise, ridge=ridge)
        rates, _ = optimize_and_score(estimate, validation.channels, device, validation_solver)
        ridge_rates[ridge] = float(rates.double().mean())
    best_ridge = max(ridge_rates, key=lambda value: ridge_rates[value])
    strata: JsonObject = {}
    report: JsonObject = {
        "model": asdict(model.config),
        "noise": noise,
        "scenes_per_count": scenes,
        "solver": asdict(solver),
        "selected_ridge": best_ridge,
        "ridge_validation_rates": {str(k): v for k, v in ridge_rates.items()},
        "metric": "true-channel RZF scores returned phases; estimated CSI selects phases",
        "reference": "same multi-start local optimizer on full CSI; not a global upper bound",
        "bootstrap": "1000 resamples of globally unique physical-panel rows",
        "strata": strata,
    }
    for users, antennas in counts:
        generator = torch.Generator().manual_seed(20261004 + users * 100 + antennas + 10000)
        examples = pool.sample(splits["test"], scenes, users, antennas, generator)
        truth = examples.channels
        reference, _ = optimize_and_score(truth, truth, device, solver)
        with torch.no_grad():
            identity = channel_rate(truth, torch.zeros((scenes, pool.elements)))
        methods: JsonObject = {}
        arrays = {
            "identity": identity.numpy(),
            "reference": reference.numpy(),
            "panels": examples.panel_rows.numpy(),
        }
        for label, method_ridge in (("neural", None), ("ridge", best_ridge)):
            estimated = predict(model, examples, device, noise, ridge=method_ridge)
            rates, phases = optimize_and_score(estimated, truth, device, solver)
            captured, interval = capture(
                rates.double() - identity.double(),
                reference.double() - identity.double(),
                examples.panel_rows,
            )
            with torch.no_grad():
                deployed = channel_rate(truth, phases, estimated=estimated)
            deployed_capture, _ = capture(
                deployed.double() - identity.double(),
                reference.double() - identity.double(),
                examples.panel_rows,
            )
            methods[label] = {
                "rate": float(rates.double().mean()),
                "identity": float(identity.double().mean()),
                "gain": float((rates.double() - identity.double()).mean()),
                "reference_capture_percent": captured,
                "capture_95_ci_percent": interval,
                "estimated_precoder_reference_capture_percent": deployed_capture,
                "channel_errors": channel_errors(estimated, truth),
            }
            arrays[f"{label}_rates"] = rates.numpy()
            arrays[f"{label}_phases"] = phases.numpy()
        key = f"{users}x{antennas}"
        strata[key] = methods
        save_npz(output.with_name(f"{output.stem}_{key}.npz"), **arrays)
        write_json(output, report)
        print(f"{key}: {methods}", flush=True)
    return report
