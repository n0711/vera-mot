#!/usr/bin/env python3
"""Read-only inventory of M4 input paths. Presence never proves integrity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHECKPOINTS = (
    "model_0009.pth", "model_0019.pth", "model_0029.pth",
    "model_0039.pth", "model_0049.pth", "model_final.pth",
)


def _split(root: Path) -> dict:
    config_path = root / "configs/uavdt_reid_split.yaml"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"missing or invalid frozen split: {config_path}: {exc}") from exc
    groups = {
        "train": config.get("train_sequences"),
        "validation": config.get("validation_sequences"),
        "test": config.get("frozen_test_sequences"),
    }
    for key, count in (("train", 24), ("validation", 6), ("test", 20)):
        values = groups[key]
        if (not isinstance(values, list) or len(values) != count
                or any(not isinstance(x, str) or not x for x in values)
                or len(set(values)) != count):
            raise ValueError(f"invalid frozen {key} sequence list")
    if any(set(groups[a]) & set(groups[b]) for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    )):
        raise ValueError("frozen train/validation/test sequences overlap")
    return groups


def audit(root: Path) -> dict:
    root = root.resolve()
    groups = _split(root)
    allowed = groups["train"] + groups["validation"]
    images = "datasets/UAVDT/full_mot/raw/UAV-benchmark-M"
    annotations = "datasets/UAVDT/full_mot/toolkit/UAV-benchmark-MOTD_v1.0/GT"
    training = "experiments/reid_training/uavdt_sbs_R50_ibn"
    paths = {
        "source_sequence_directories": [f"{images}/{seq}" for seq in allowed],
        "source_ground_truth": [f"{annotations}/{seq}_gt_whole.txt" for seq in allowed],
        "reid_manifests": [f"datasets/UAVDT/reid_manifests/{s}.csv" for s in ("train", "val")],
        "reid_crop_directories": [f"datasets/UAVDT/reid_crops/{s}" for s in ("train", "val")],
        "reid_crop_registries": [f"datasets/UAVDT/reid_manifests/{s}_crop_registry.csv" for s in ("train", "val")],
        "training_records": [f"{training}/{name}" for name in ("launch_manifest.json", "completion.json")],
        "training_checkpoints": [f"{training}/{name}" for name in CHECKPOINTS],
        "fastreid_submodule": ["baselines/BoT-SORT/fast_reid"],
    }
    folders = {"source_sequence_directories", "reid_crop_directories", "fastreid_submodule"}
    checks = {}
    for key, expected in paths.items():
        missing = [
            item for item in expected
            if not ((root / item).is_dir() if key in folders else (root / item).is_file())
        ]
        checks[key] = {"required": len(expected), "present": len(expected) - len(missing), "missing": missing}
    found = all(not entry["missing"] for entry in checks.values())
    return {
        "scope": "read_only_filesystem_presence_not_integrity_or_inference",
        "split_sequence_counts": {key: len(val) for key, val in groups.items()},
        "all_files_present": found,
        "checks": checks,
        "next_action": "validate hashes and data content" if found
                       else "obtain authorized UAVDT inputs; no real inference possible",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        report = audit(args.project_root)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_files_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
