"""Validate recorded FastReID training evidence without loading weights or PyTorch.

This is an internal consistency check, not independent attestation that a
training run occurred or that its upstream datasets were authentic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from vera_mot.fastreid_uavdt import EXPECTED
from vera_mot.reid_data import ManifestError, sha256_file


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"missing or invalid training provenance file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"training provenance must be a JSON object: {path}")
    return value


def _require_integer(document: dict, name: str, expected: int | None = None) -> int:
    value = document.get(name)
    if type(value) is not int or value < 0 or (expected is not None and value != expected):
        raise ManifestError(f"invalid training provenance integer {name}: {value!r}")
    return value


def _require_digest(document: dict, name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ManifestError(f"invalid training provenance sha256 {name}")
    return value


def verify_training_provenance(checkpoint_dir: Path, required_checkpoints: tuple[str, ...]) -> dict:
    """Fail closed on missing, inconsistent or modified local training artifacts.

    The hash and byte-size contracts come from the training script's own
    completion.json. Hashes do not establish the external origin of that file.
    """
    root = checkpoint_dir.resolve()
    launch_path = root / "launch_manifest.json"
    completion_path = root / "completion.json"
    launch, completion = _read_object(launch_path), _read_object(completion_path)
    for name, expected in (
        ("train_identities", EXPECTED["train"]["identities"]),
        ("train_images", EXPECTED["train"]["images"]),
        ("validation_identities", EXPECTED["val"]["identities"]),
        ("validation_images", EXPECTED["val"]["images"]),
        ("epochs", 60),
        ("seed", 42),
        ("frozen_test_records", 0),
        ("private_records", 0),
    ):
        _require_integer(launch, name, expected)
    iterations_per_epoch = _require_integer(launch, "iterations_per_epoch")
    total_iterations = _require_integer(launch, "total_iterations")
    if iterations_per_epoch == 0 or total_iterations != 60 * iterations_per_epoch:
        raise ManifestError("training iteration counts do not match the declared schedule")
    _require_integer(completion, "epochs_completed", 60)
    _require_integer(completion, "iterations_completed", total_iterations)
    initialization_sha = _require_digest(launch, "initialization_observed_sha256")
    for name in ("git_revision", "botsort_revision"):
        revision = launch.get(name)
        if not isinstance(revision, str) or _GIT_SHA.fullmatch(revision) is None:
            raise ManifestError(f"invalid training provenance revision {name}")

    entries = completion.get("checkpoints")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("training completion has no checkpoint inventory")
    inventory = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ManifestError("invalid training checkpoint inventory entry")
        name = entry.get("filename")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pth"):
            raise ManifestError(f"unsafe or invalid training checkpoint filename: {name!r}")
        if name in inventory:
            raise ManifestError(f"duplicate training checkpoint entry: {name}")
        recorded_bytes = _require_integer(entry, "bytes")
        recorded_sha = _require_digest(entry, "sha256")
        checkpoint = root / name
        if not checkpoint.is_file():
            raise ManifestError(f"missing recorded training checkpoint: {name}")
        if checkpoint.stat().st_size != recorded_bytes or sha256_file(checkpoint) != recorded_sha:
            raise ManifestError(f"training checkpoint differs from completion evidence: {name}")
        inventory[name] = recorded_sha
    missing = sorted(set(required_checkpoints) - set(inventory))
    if missing:
        raise ManifestError(f"training completion missing required checkpoints: {', '.join(missing)}")
    return {
        "launch_manifest_sha256": sha256_file(launch_path),
        "completion_sha256": sha256_file(completion_path),
        "training_git_revision": launch["git_revision"],
        "botsort_revision": launch["botsort_revision"],
        "initialization_observed_sha256": initialization_sha,
        "epochs_completed": completion["epochs_completed"],
        "iterations_completed": completion["iterations_completed"],
        "checkpoint_sha256": {name: inventory[name] for name in sorted(required_checkpoints)},
        "verification": "internal_record_and_bytes_only",
    }
