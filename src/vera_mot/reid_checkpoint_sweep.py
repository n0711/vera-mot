"""Provenance and safe result reuse for the fixed six-checkpoint ReID sweep."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml

from vera_mot.reid_data import ManifestError, sha256_file
from vera_mot.reid_temporal_eval import PROTOCOL_VERSION


CHECKPOINT_NAMES = tuple(f"model_{epoch:04d}.pth" for epoch in (9, 19, 29, 39, 49)) + ("model_final.pth",)
PRIMARY_CHECKPOINT = "model_final.pth"
ARTIFACTS = (
    "summary.json", "protocol_manifest.json", "per_sequence.csv", "per_query.csv",
    "hardest_positive_pairs.csv", "nearest_negative_pairs.csv",
)


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def runtime_versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "numpy", "Pillow", "opencv-python", "opencv-python-headless", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def configuration_hashes(path: Path, active: tuple[Path, ...] = ()) -> dict[str, str]:
    """Hash the complete FastReID YAML inheritance chain, rejecting cycles."""
    path = path.resolve()
    if path in active:
        raise ManifestError(f"configuration inheritance cycle: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ManifestError(f"configuration is not a mapping: {path}")
        result = {str(path): sha256_file(path)}
        bases = value.get("_BASE_", [])
        if isinstance(bases, str):
            bases = [bases]
        if not isinstance(bases, list) or any(not isinstance(base, str) for base in bases):
            raise ManifestError(f"invalid configuration _BASE_: {path}")
        for base in bases:
            result.update(configuration_hashes(path.parent / base, (*active, path)))
        return result
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read configuration dependency {path}: {exc}") from exc


def build_plan(project_root: Path, checkpoint_dir: Path, config: Path, registry: Path,
               output_dir: Path, *, device: str, batch_size: int, num_workers: int,
               max_identities: int | None, verify_crops: bool) -> dict:
    if batch_size < 1 or num_workers < 0 or (max_identities is not None and max_identities < 1):
        raise ManifestError("invalid batch size, worker count or identity limit")
    checkpoint_dir, output_dir = checkpoint_dir.resolve(), output_dir.resolve()
    missing = [name for name in CHECKPOINT_NAMES if not (checkpoint_dir / name).is_file()]
    if missing:
        raise ManifestError(f"missing trained checkpoints in {checkpoint_dir}: {', '.join(missing)}")
    config, registry = config.resolve(), registry.resolve()
    # Include project evaluator code and the pinned upstream implementation, so
    # a changed evaluator cannot silently reuse an earlier result.
    sources = [project_root / "scripts/evaluate_uavdt_reid_temporal.py",
               project_root / "scripts/evaluate_uavdt_reid_checkpoints.py"]
    sources += sorted((project_root / "src/vera_mot").glob("*.py"))
    upstream = project_root / "baselines/BoT-SORT/fast_reid"
    if not upstream.is_dir():
        raise ManifestError("initialize the pinned BoT-SORT submodule before planning evaluation")
    sources += sorted(upstream.rglob("*.py"))
    inputs = {
        "protocol_version": PROTOCOL_VERSION,
        "configuration_path": str(config), "configuration_sha256": sha256_file(config),
        "configuration_dependencies": configuration_hashes(config),
        "registry_path": str(registry), "registry_sha256": sha256_file(registry),
        "split_configuration_sha256": sha256_file(project_root / "configs/uavdt_reid_split.yaml"),
        "source_sha256": {str(path.relative_to(project_root)): sha256_file(path) for path in sources},
        "runtime_versions": runtime_versions(),
        "evaluation_scope": "smoke" if max_identities is not None else "full_validation",
        "device": device, "batch_size": batch_size, "num_workers": num_workers,
        "max_identities": max_identities, "verify_crops": verify_crops,
    }
    jobs = []
    for name in CHECKPOINT_NAMES:
        checkpoint = checkpoint_dir / name
        contract = {**inputs, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}
        jobs.append({"checkpoint": name, "primary": name == PRIMARY_CHECKPOINT,
                     "output_dir": str(output_dir / checkpoint.stem),
                     "input_fingerprint": fingerprint(contract), "inputs": contract})
    return {"primary_checkpoint": PRIMARY_CHECKPOINT, "evaluation_scope": inputs["evaluation_scope"],
            "checkpoint_selection": "fixed final epoch; other checkpoints are diagnostic only", "jobs": jobs}


def validate_result(job: dict) -> dict:
    output = Path(job["output_dir"])
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "protocol_manifest.json").read_text(encoding="utf-8"))
        expected = {key: value for key, value in job["inputs"].items()
                    if key not in {"configuration_dependencies", "source_sha256", "runtime_versions"}}
        for document in (summary, manifest):
            if any(key not in document or document[key] != value for key, value in expected.items()):
                raise ManifestError(f"evaluation provenance mismatch: {output}")
        for label in ("macro_average", "micro_average"):
            if summary[label]["queries"] < 1:
                raise ManifestError(f"empty evaluation result: {output}")
        selected = manifest["split"]
        if not selected["identities"]:
            raise ManifestError(f"empty temporal split: {output}")
        for name in ARTIFACTS:
            if not (output / name).is_file():
                raise ManifestError(f"incomplete evaluation output: {output / name}")
        return {"checkpoint": job["checkpoint"], "primary": job["primary"],
                "evaluation_scope": job["inputs"]["evaluation_scope"],
                "checkpoint_sha256": job["inputs"]["checkpoint_sha256"],
                "split_fingerprint": fingerprint(selected),
                "macro_average": summary["macro_average"], "micro_average": summary["micro_average"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"invalid or incomplete evaluation in {output}: {exc}") from exc


def record_result(job: dict) -> dict:
    result = validate_result(job)
    output = Path(job["output_dir"])
    record = {"input_fingerprint": job["input_fingerprint"],
              "artifact_sha256": {name: sha256_file(output / name) for name in ARTIFACTS}}
    (output / "run_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def reuse_result(job: dict) -> dict:
    output = Path(job["output_dir"])
    try:
        record = json.loads((output / "run_record.json").read_text(encoding="utf-8"))
        if record["input_fingerprint"] != job["input_fingerprint"]:
            raise ManifestError(f"stale evaluation inputs: {output}")
        if record["artifact_sha256"] != {name: sha256_file(output / name) for name in ARTIFACTS}:
            raise ManifestError(f"modified evaluation artifacts: {output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"cannot safely resume {output}: {exc}") from exc
    return validate_result(job)
