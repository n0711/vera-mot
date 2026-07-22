#!/usr/bin/env python3
"""Materialize and verify lossless native-size UAVDT ReID crops."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vera_mot.reid_crops import (  # noqa: E402
    create_qa_sheets, dataset_checksum, materialize_split, preflight,
    read_manifest, summarize_registry, write_registry,
)
from vera_mot.reid_data import ManifestError, sha256_file  # noqa: E402


def dirty_note(root: Path) -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=root, text=True, capture_output=True, check=False)
    return "Generated from working-tree code; repository was dirty." if result.stdout else "Generated from clean committed code."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
        if branch != "milestone-4-reid-baseline":
            raise ManifestError(f"wrong branch: {branch}")
        config_path = root / "configs/uavdt_reid_split.yaml"
        config = json.loads(config_path.read_text())
        frozen_test = set(config["frozen_test_sequences"])
        manifest_summary_path = root / "experiments/reid_data/uavdt_reid_manifest_summary.json"
        manifest_summary = json.loads(manifest_summary_path.read_text())
        rows_by_split = {}
        manifest_checksums = {}
        for split in ("train", "val"):
            path = root / f"datasets/UAVDT/reid_manifests/{split}.csv"
            checksum = sha256_file(path)
            expected = manifest_summary["checksums"][f"{split}_manifest_sha256"]
            if checksum != expected:
                raise ManifestError(f"{split} manifest checksum mismatch")
            manifest_checksums[split] = checksum
            rows_by_split[split] = read_manifest(path, split, frozen_test)
        flight = preflight(root, rows_by_split)
        if args.preflight_only:
            print(json.dumps({"branch": branch, "dirty_state": dirty_note(root), **flight}, indent=2, sort_keys=True))
            return 0
        registry_rows = {}
        run_stats = {}
        registry_checksums = {}
        dataset_checksums = {}
        for split in ("train", "val"):
            registry_rows[split], run_stats[split] = materialize_split(root, rows_by_split[split], force=args.force)
            registry_path = root / f"datasets/UAVDT/reid_manifests/{split}_crop_registry.csv"
            registry_checksums[split] = write_registry(registry_path, registry_rows[split])
            dataset_checksums[split] = dataset_checksum(registry_rows[split])
        qa = create_qa_sheets(root, rows_by_split, root / "experiments/reid_data/qa")
        summary = {
            "schema_version": 1, "manifest_checksums": manifest_checksums,
            "crop_registry_checksums": registry_checksums,
            "dataset_checksums": dataset_checksums, "preflight": flight,
            "splits": {split: summarize_registry(registry_rows[split]) | run_stats[split] for split in ("train", "val")},
            "total_crop_bytes": sum(int(row["file_size"]) for rows in registry_rows.values() for row in rows),
            "identity_rejections": {split: manifest_summary["splits"][split]["identity_accounting"] for split in ("train", "val")},
            "category_conflicts": {split: manifest_summary["splits"][split]["category_conflicts"] for split in ("train", "val")},
            "missing_crops": 0, "failed_crops": 0,
            "pixel_equality": {"verified_crops": sum(len(rows) for rows in registry_rows.values()), "mismatches": 0},
            "leakage": {"test_sequence_records": 0, "private_or_local_video_records": 0, "result": "passed"},
            "visual_qa_artifacts": qa, "code_and_dirty_state_note": dirty_note(root),
        }
        summary_path = root / "experiments/reid_data/uavdt_reid_crop_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (ManifestError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
