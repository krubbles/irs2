"""Train the current estimator with learned pilots and exact checkpoint resumption."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from torch import Tensor, nn

from .channels import Channels, channel_rate
from .data import load_pool
from .estimator import ChannelEstimatorConfig, NeuralChannelEstimator
from .evaluation import optimize_and_score, validate
from .io import JsonObject, file_hash, save_checkpoint, write_json
from .solver import SolverConfig


@dataclass(frozen=True)
class TrainingConfig:
    model: ChannelEstimatorConfig = ChannelEstimatorConfig(lift="ridge")
    counts: tuple[tuple[int, int], ...] = ((2, 2), (2, 4))  # users, antennas
    steps: int = 24000
    batch_size: int = 32
    learning_rate: float = 3e-4
    pilot_learning_rate: float = 3e-4
    evaluate_every: int = 1000
    validation_scenes: int = 512
    noise: float = 0.01
    validation_solver: SolverConfig = SolverConfig(steps=50, restarts=1)

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.evaluate_every, self.validation_scenes) <= 0:
            raise ValueError("Training sizes must be positive")
        if not self.counts or any(users <= 0 or antennas <= 0 for users, antennas in self.counts):
            raise ValueError("Training counts must be positive")
        if not isfinite(self.noise) or self.noise < 0:
            raise ValueError("Noise must be finite and nonnegative")
        if any(
            not isfinite(rate) or rate <= 0
            for rate in (self.learning_rate, self.pilot_learning_rate)
        ):
            raise ValueError("Learning rates must be finite and positive")


def estimation_loss(predicted: Channels, target: Channels) -> Tensor:
    """Keep weak cascade errors visible instead of hiding them behind the direct path."""
    error = (predicted.cascade - target.cascade).abs().square().sum((2, 3))
    power = target.cascade.abs().square().sum((2, 3)).clamp_min(1e-30)
    cascade = (error / power)[target.mask].log1p().mean()
    direct_error = (predicted.direct - target.direct).abs().square().sum(2)
    direct_power = target.direct.abs().square().sum(2)
    direct = (direct_error / (direct_power + power))[target.mask].log1p().mean()
    return cascade + 0.1 * direct


def train(
    dataset: Path,
    output: Path,
    config: TrainingConfig,
    device: torch.device,
    *,
    resume: bool = False,
) -> None:
    pool = load_pool(dataset)
    if pool.elements != config.model.elements:
        raise ValueError("Dataset and estimator element counts differ")
    splits = pool.split()
    for name in ("train", "validation"):
        for users, antennas in config.counts:
            if users > int(pool.counts[splits[name]].max()) or antennas > pool.antennas:
                raise ValueError(
                    f"The {name} split cannot support {users} users / {antennas} antennas"
                )
    configuration: JsonObject = asdict(config)
    configuration.update(dataset_sha256=file_hash(dataset), device=str(device))
    # JSON normalizes count tuples; compare the same form on resume.
    configuration = cast(JsonObject, json.loads(json.dumps(configuration)))
    torch.manual_seed(config.model.seed)
    model = NeuralChannelEstimator(config.model).to(device)
    groups: list[dict[str, object]] = [
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if name not in ("symbol_phases", "probe_phases") and p.requires_grad
            ]
        },
        {
            "params": [model.symbol_phases, model.probe_phases],
            "lr": config.pilot_learning_rate,
            "weight_decay": 0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(config.model.seed)
    noise_generator = torch.Generator(device=device).manual_seed(config.model.seed + 1)
    history: list[JsonObject] = []
    best = -float("inf")
    start_step = 0
    if resume:
        checkpoint = cast(
            JsonObject, torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        )
        previous = cast(JsonObject, checkpoint["configuration"])
        if {k: v for k, v in previous.items() if k != "steps"} != {
            k: v for k, v in configuration.items() if k != "steps"
        }:
            raise ValueError("Resume configuration or dataset differs")
        start_step = cast(int, checkpoint["step"])
        if config.steps < cast(int, previous["steps"]):
            raise ValueError("Resume cannot shorten the saved training horizon")
        model.load_state_dict(cast(dict[str, Tensor], checkpoint["model"]))
        optimizer.load_state_dict(cast(dict[str, object], checkpoint["optimizer"]))
        generator.set_state(cast(Tensor, checkpoint["generator"]))
        noise_generator.set_state(cast(Tensor, checkpoint["noise_generator"]))
        torch.set_rng_state(cast(Tensor, checkpoint["torch_rng"]))
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cast(list[Tensor], checkpoint["cuda_rng"]))
        best = cast(float, checkpoint["best_gain"])
        history = cast(list[JsonObject], checkpoint["history"])
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Run directory exists; use a new output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "configuration.json", configuration)
    users, antennas = config.counts[0]
    validation = pool.sample(
        splits["validation"],
        config.validation_scenes,
        users,
        antennas,
        torch.Generator().manual_seed(20261003),
    )
    truth = validation.channels
    oracle, phases = optimize_and_score(truth, truth, device, config.validation_solver)
    with torch.no_grad():
        identity = channel_rate(truth, torch.zeros_like(phases))
    reference_gain = float((oracle.double() - identity.double()).mean())
    write_json(output / "validation_reference.json", {"gain": reference_gain})
    started = perf_counter()

    def record(step: int) -> None:
        nonlocal best
        result = validate(model, validation, device, config.noise, config.validation_solver)
        history.append(
            {
                "step": step,
                "seconds": perf_counter() - started,
                "validation": result,
                "reference_capture_percent": 100 * result["gain"] / reference_gain
                if reference_gain > 1e-30
                else None,
            }
        )
        improved = result["gain"] > best
        best = max(best, result["gain"])
        checkpoint: JsonObject = {
            "step": step,
            "config": asdict(config.model),
            "configuration": configuration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
            "noise_generator": noise_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "best_gain": best,
            "history": history,
        }
        save_checkpoint(output / "latest.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
        write_json(output / "history.json", history)
        print(
            f"Step {step}: gain={result['gain']:.6g}, cascade NMSE={result['cascade_global_nmse']:.6g}",
            flush=True,
        )

    if not history:
        record(0)
    for step in range(start_step + 1, config.steps + 1):
        users, antennas = config.counts[(step - 1) % len(config.counts)]
        batch = pool.sample(splits["train"], config.batch_size, users, antennas, generator).slice(
            0, config.batch_size, device
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        received = model.observe(batch, config.noise, noise_generator)
        predicted: Channels = model(
            received,
            batch.user_mask,
            batch.irs_locations,
            batch.transmitter_locations,
            config.noise,
        )
        loss = estimation_loss(predicted, batch.channels)
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 5)
        if not bool(loss.isfinite() and gradient.isfinite()):
            raise RuntimeError(f"Non-finite training at step {step}")
        optimizer.step()
        if step % config.evaluate_every == 0 or step == config.steps:
            record(step)
    write_json(
        output / "complete.json",
        {"steps": config.steps, "best_gain": best, "seconds": perf_counter() - started},
    )
