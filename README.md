# IRS2

A standalone extraction of the current multi-city dataset generator and newest
solver pipeline: **learned pilots → neural channel estimates → phase optimization**.
The package is self-contained and does not import the original project's Python
modules. Datasets, trained checkpoints, and experiment outputs are supplied or
generated separately; they are not included in this repository.

## Setup

Clone the repository and create a Python 3.11+ virtual environment:

```bash
git clone https://github.com/krubbles/irs2.git
cd irs2
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'             # Estimation, optimization, training
python -m pip install -e '.[generation,dev]'  # Also install DeepMIMO/Sionna ray tracing
```

Commands below use `irs`; `python -m irs` is equivalent. Paths are explicit and
relative to the current working directory. New artifacts default to `data/`;
the examples put training artifacts in `runs/`.

## Use existing data and the trained model

```bash
irs evaluate \
  --dataset /path/to/multicity_tiled_panel_optimizer_channels_gain_ge_0p1.npz \
  --checkpoint /path/to/sc_attention_ridge_p17_mixed_24k/best.pt \
  --output runs/existing-model.json
```

The estimator keeps the existing model configuration and state-dict names, so
current neural-channel-estimator checkpoints load directly. No dataset conversion
or copying is needed. Training defaults to the successful 17-pilot, width-32,
four-block ridge-lift model, mixed two/four TX antennas, two users, and 24k steps.

```bash
irs train \
  --dataset data/filtered.npz \
  --output runs/model

# Continue a v2 run, preserving optimizer and random-generator states.
irs train \
  --dataset data/filtered.npz \
  --output runs/model --resume --steps 30000
```

Use `--pilots 8 --lift learned --counts 2x2` for the current short-pilot variant.
CPU is the default; `--device cuda` selects an available GPU. Thread count is one.
Training on an existing nonempty output directory requires `--resume`.
Legacy checkpoints support inference; exact training resumption is for v2 runs.

## Generate the current dataset

```bash
# Inspect deterministic panel/user selection without tracing channels.
irs generate --output-root data --plan-only

# Resume power filtering, trace complex channels, and combine the city pools.
irs generate --output-root data

# Cache single-user full-CSI gains, then retain gains >= 0.1 bit/s/Hz.
irs filter-gain --dataset data/channels.npz --output data/filtered.npz
```

`generate` retains the 19 selected cities, 512 users per city, 10k candidate-panel
cap, deterministic seeds, coherent path sums, Tang cell normalization, and
per-batch checkpoints. The first screen uses the existing 32×32 physical-panel
power approximation, a −15 dB margin to direct power, and 100 dB maximum IRS
path loss. Materialization retains the existing 4×4 sampled IRS across a 0.62 m
aperture and four physical TX elements. Those distinct representations are
intentional and match the current dataset.

The final gain filter retains the current sampling rule: each link is scored as
a single-user case with a random one-to-four-antenna subset, then filtered by
absolute full-CSI gain. Its work is resumable and no old training configuration
is needed. If the exhaustive gain cache already exists, pass
`--gain-cache /path/to/samples.npz` to reuse it. Recomputed gains use the retained
antenna-space RZF solver, so links exactly at the cutoff can differ through
numerical roundoff from older user-space reference calculations.

For a smaller generation run, restrict `--scenarios`, `--user-count`, and
`--maximum-panels-per-scene`. Run `irs generate --help` for tracing batch controls.
Planning can download missing DeepMIMO scenarios. Rerun with the same settings
to resume; incompatible settings or changed sources are rejected.

## Python inference

```python
from pathlib import Path
import torch
from irs import load_estimator, optimize_phases

model = load_estimator(Path("runs/model/best.pt"))

# Acquire complex64 [batch,user,antenna,pilot] observations using these banks.
symbols, probes = model.symbols(), model.probes()

with torch.no_grad():
    estimated = model(received, user_mask, irs_locations, transmitter_locations, 0.01)
    phases = optimize_phases(estimated)  # [batch,16], radians
```

`received`, `user_mask`, and locations come from the caller. The mask is bool
`[batch,user]`; locations are float32 `[batch,element,3]` and
`[batch,antenna,3]`. All inputs must share the model's device. The pilot/probe bank
must match the checkpoint. The nominal noise argument is complex RMS amplitude
relative to the initial-bank pilot RMS. Simulation fixes that initial-bank noise
power while pilots are learned.

The phase solver enables its own gradients under `torch.no_grad()` and detaches
the supplied channel estimates. Use `no_grad()` for this pipeline, since
`inference_mode()` prevents the gradients needed for phase optimization.

## Structure and validation

| Module | Responsibility |
| --- | --- |
| `channels.py` | Channel contract, pilot physics, and shared coordinated RZF rate |
| `data.py` | Ragged NPZ loading, physical-panel splits, and aligned sampling |
| `estimator.py` | Current trainable pilot bank and neural channel estimator |
| `solver.py` | Best-iterate, multi-start phase optimization |
| `training.py`, `evaluation.py` | Checkpointed training and held-out scoring |
| `generation/` | Wall geometry, shared ray tracing, power filter, materialization, gain filter |
| `cli.py`, `io.py` | Thin command entry point and atomic artifact storage |

The seeded 80/10/10 physical-panel split is preserved. Validation selects
checkpoints and ridge regularization; test panels supply final scores. The
primary metric scores returned phases with true-channel RZF, while a separate
estimated-precoder metric includes precoding error. Full CSI enters labels,
references, and scoring only. The multi-start reference is a local solution,
not a global upper bound. Bootstrap statistics use globally unique panel rows.

```bash
pytest
mypy
ruff check src tests
ruff format --check src tests
```

Tests exercise channel alignment, pilot gradients, inference symmetries, masking,
solver determinism, gain filtering, cache validation, and exact training resume.
Generator cache tests run when the optional ray-tracing dependencies are present.
Old architectures, dashboards, plotting scripts, notebooks, and experiment
sweeps are outside this package.
