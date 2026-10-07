"""One command-line entry point for generation, training, and held-out evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

from .estimator import ChannelEstimatorConfig, Lift, load_estimator
from .solver import SolverConfig


def parse_counts(value: str) -> tuple[tuple[int, int], ...]:
    try:
        result = tuple(tuple(map(int, pair.split("x"))) for pair in value.split(","))
        if not result or any(
            len(pair) != 2 or pair[0] < 1 or not 1 <= pair[1] <= 4 for pair in result
        ):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use USERSxANTENNAS, e.g. 2x2,2x4; antennas must be 1..4"
        ) from error
    return cast(tuple[tuple[int, int], ...], result)


def resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(value)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Multi-city IRS datasets and neural CSI -> phase optimization"
    )
    commands = root.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Plan or resume the multi-city raw channel pool"
    )
    generate.add_argument("--output-root", type=Path, default=Path("data"))
    generate.add_argument(
        "--scenarios", nargs="+", help="DeepMIMO scenario names; defaults to the current 19 cities"
    )
    generate.add_argument("--plan-only", action="store_true")
    generate.add_argument("--samples-per-source", type=int, default=50000)
    generate.add_argument("--user-count", type=int, default=512)
    generate.add_argument("--maximum-panels-per-scene", type=int, default=10000)
    generate.add_argument("--selection-seed", type=int, default=20260913)
    generate.add_argument("--panel-source-batch-size", type=int, default=16)
    generate.add_argument("--panel-target-batch-size", type=int, default=512)
    generate.add_argument("--panel-batch-size", type=int, default=8)
    generate.add_argument("--tx-target-panel-batch-size", type=int, default=32)

    gain = commands.add_parser(
        "filter-gain", help="Cache full-CSI single-user gains and filter qualifying links"
    )
    gain.add_argument("--dataset", type=Path, required=True)
    gain.add_argument("--output", type=Path, required=True)
    gain.add_argument(
        "--gain-cache",
        type=Path,
        help="Reuse an existing exhaustive gain NPZ instead of recomputing",
    )
    gain.add_argument("--minimum-gain", type=float, default=0.1)
    gain.add_argument("--batch-size", type=int, default=128)
    gain.add_argument("--seed", type=int, default=20260827)

    train = commands.add_parser(
        "train", help="Train the newest estimator; continue an existing run with --resume"
    )
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--counts", type=parse_counts, default=parse_counts("2x2,2x4"))
    train.add_argument("--steps", type=int, default=24000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--pilots", type=int, default=17)
    train.add_argument("--lift", choices=("learned", "ridge"), default="ridge")
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--blocks", type=int, default=4)
    train.add_argument("--seed", type=int, default=20261006)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--pilot-learning-rate", type=float, default=3e-4)
    train.add_argument("--evaluate-every", type=int, default=1000)
    train.add_argument("--validation-scenes", type=int, default=512)
    train.add_argument("--validation-phase-steps", type=int, default=50)
    train.add_argument("--noise", type=float, default=0.01)
    train.add_argument("--resume", action="store_true")

    evaluate = commands.add_parser(
        "evaluate", help="Score a new or existing estimator checkpoint on held-out panels"
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--counts", type=parse_counts, default=parse_counts("2x2,2x4,4x2,4x4,8x4")
    )
    evaluate.add_argument("--scenes", type=int, default=1024)
    evaluate.add_argument("--noise", type=float, default=0.01)

    for command in (gain, train, evaluate):
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for command, steps, restarts in ((gain, 250, 8), (evaluate, 100, 3)):
        command.add_argument("--phase-steps", type=int, default=steps)
        command.add_argument("--restarts", type=int, default=restarts)
    return root


def main() -> None:
    root = parser()
    args = root.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "generate":
            # The solver and its CLI remain usable without Sionna/DeepMIMO.
            from .generation.config import (
                SELECTED_SCENARIOS,
                MaterializationSettings,
                MultiCitySettings,
            )
            from .generation.filtering import build_multicity_dataset
            from .generation.materialize import materialize_collection

            scenarios = tuple(args.scenarios) if args.scenarios else SELECTED_SCENARIOS
            invalid = set(scenarios) - set(SELECTED_SCENARIOS)
            if invalid or len(set(scenarios)) != len(scenarios):
                raise ValueError(
                    f"Expected unique scenarios from the current 19 cities; invalid: {sorted(invalid)}"
                )
            settings = MultiCitySettings(
                samples_per_source=args.samples_per_source,
                user_count=args.user_count,
                maximum_panels_per_scene=args.maximum_panels_per_scene,
                selection_seed=args.selection_seed,
                panel_source_batch_size=args.panel_source_batch_size,
                panel_target_batch_size=args.panel_target_batch_size,
            )
            tracing = MaterializationSettings(
                args.samples_per_source, args.panel_batch_size, args.tx_target_panel_batch_size
            )
            build_multicity_dataset(
                scenarios,
                output_root=args.output_root / "filter",
                settings=settings,
                plan_only=args.plan_only,
            )
            if not args.plan_only:
                materialize_collection(
                    filter_root=args.output_root / "filter",
                    output_root=args.output_root / "channels",
                    combined_output=args.output_root / "channels.npz",
                    settings=tracing,
                )
        elif args.command == "filter-gain":
            from .generation.gain import cache_gains, filter_by_gain

            if args.output.resolve() == args.dataset.resolve():
                raise ValueError("Filtered output must differ from the source dataset")
            from math import isfinite

            if not isfinite(args.minimum_gain) or args.minimum_gain < 0:
                raise ValueError("Minimum gain must be finite and nonnegative")
            solver = SolverConfig(steps=args.phase_steps, restarts=args.restarts)
            cache = args.gain_cache or cache_gains(
                args.dataset,
                args.output.parent / "gain_cache",
                resolve_device(args.device),
                solver,
                args.batch_size,
                args.seed,
            )
            filter_by_gain(args.dataset, cache, args.output, args.minimum_gain)
        elif args.command == "train":
            from .training import TrainingConfig, train

            model = ChannelEstimatorConfig(
                pilots=args.pilots,
                width=args.width,
                blocks=args.blocks,
                seed=args.seed,
                lift=cast(Lift, args.lift),
            )
            config = TrainingConfig(
                model=model,
                counts=args.counts,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                pilot_learning_rate=args.pilot_learning_rate,
                evaluate_every=args.evaluate_every,
                validation_scenes=args.validation_scenes,
                noise=args.noise,
                validation_solver=SolverConfig(steps=args.validation_phase_steps, restarts=1),
            )
            train(
                args.dataset, args.output, config, resolve_device(args.device), resume=args.resume
            )
        else:
            from .data import load_pool
            from .evaluation import evaluate
            from .io import file_hash, write_json

            device = resolve_device(args.device)
            estimator = load_estimator(args.checkpoint, device)
            report = evaluate(
                estimator,
                load_pool(args.dataset),
                args.counts,
                args.scenes,
                args.noise,
                device,
                SolverConfig(steps=args.phase_steps, restarts=args.restarts),
                args.output,
            )
            report.update(
                dataset_sha256=file_hash(args.dataset), checkpoint_sha256=file_hash(args.checkpoint)
            )
            write_json(args.output, report)
    except (ValueError, FileNotFoundError) as error:
        root.error(str(error))
    except ModuleNotFoundError as error:
        root.error(f"{error}. Install irs-v2[generation] for ray tracing.")
