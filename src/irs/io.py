"""Atomic artifact writes and streaming fingerprints shared by both workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

Array = npt.NDArray[np.generic]
JsonObject = dict[str, object]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> JsonObject:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(JsonObject, value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npy(path: Path, value: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def save_npz(path: Path, **arrays: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, allow_pickle=False, **arrays)
    temporary.replace(path)


def save_checkpoint(path: Path, checkpoint: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def validate_manifest(directory: Path, manifest: JsonObject) -> None:
    path = directory / "manifest.json"
    normalized = cast(JsonObject, json.loads(json.dumps(manifest)))
    if path.exists():
        if read_json(path) != normalized:
            raise ValueError(f"Checkpoint manifest differs in {directory}")
    else:
        write_json(path, normalized)
