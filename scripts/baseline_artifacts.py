#!/usr/bin/env python3
"""Validate frozen no-ReID result files against their recorded provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vera_mot.mot_io import parse_mot_text  # noqa: E402

MANIFEST_PATH = (
    PROJECT_ROOT
    / "experiments/tracking_baselines/botsort_no_reid/manifest.yaml"
)
REGISTRY_PATH = (
    PROJECT_ROOT
    / "experiments/tracking_baselines/botsort_no_reid/result_registry.json"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path | None = None) -> dict:
    # JSON is a strict YAML subset, keeping this parser dependency-free.
    path = MANIFEST_PATH if path is None else path
    return json.loads(path.read_text(encoding="utf-8"))


def configuration_fingerprint(manifest: dict) -> str:
    frozen = manifest["frozen_configuration"]
    encoded = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_registry(path: Path | None = None) -> dict:
    path = REGISTRY_PATH if path is None else path
    return json.loads(path.read_text(encoding="utf-8"))


def verify_result(sequence: str, result_path: Path) -> tuple[bool, str]:
    if not result_path.is_file() or result_path.stat().st_size == 0:
        return False, f"{result_path} is missing or empty"
    try:
        parse_mot_text(result_path.read_text(encoding="utf-8"), source=str(result_path))
    except (OSError, ValueError) as error:
        return False, str(error)

    manifest_fingerprint = configuration_fingerprint(load_manifest())
    registry = load_registry()
    if registry.get("configuration_fingerprint") != manifest_fingerprint:
        return False, "registry configuration fingerprint does not match manifest"
    record = registry.get("results", {}).get(sequence)
    if not record:
        return False, f"no provenance record exists for {sequence}"
    actual = sha256_path(result_path)
    if record.get("sha256") != actual:
        return False, f"{sequence} checksum mismatch: expected {record.get('sha256')}, found {actual}"
    return True, f"{sequence} matches frozen configuration and checksum"


def record_result(sequence: str, result_path: Path) -> None:
    parse_mot_text(result_path.read_text(encoding="utf-8"), source=str(result_path))
    registry = load_registry()
    fingerprint = configuration_fingerprint(load_manifest())
    if registry.get("configuration_fingerprint") != fingerprint:
        raise ValueError("refusing to update registry with a mismatched configuration")
    registry.setdefault("results", {})[sequence] = {
        "path": str(result_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_path(result_path),
        "bytes": result_path.stat().st_size,
    }
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-result")
    verify.add_argument("--sequence", required=True)
    verify.add_argument("--result", type=Path, required=True)
    record = subparsers.add_parser("record-result")
    record.add_argument("--sequence", required=True)
    record.add_argument("--result", type=Path, required=True)
    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    if args.command == "verify-result":
        valid, message = verify_result(args.sequence, args.result)
        print(message)
        return 0 if valid else 1
    if args.command == "record-result":
        record_result(args.sequence, args.result)
        print(f"recorded {args.sequence}")
        return 0
    print(configuration_fingerprint(load_manifest(args.manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
