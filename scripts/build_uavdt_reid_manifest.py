#!/usr/bin/env python3
"""Build deterministic UAVDT ReID manifests; never create image crops."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vera_mot.reid_data import ManifestError, build_manifests  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--split-config", type=Path, default=Path("configs/uavdt_reid_split.yaml"))
    parser.add_argument("--gt-root", type=Path, default=Path("datasets/UAVDT/full_mot/toolkit/UAV-benchmark-MOTD_v1.0/GT"))
    parser.add_argument("--image-root", type=Path, default=Path("datasets/UAVDT/full_mot/raw/UAV-benchmark-M"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/UAVDT/reid_manifests"))
    parser.add_argument("--summary", type=Path, default=Path("experiments/reid_data/uavdt_reid_manifest_summary.json"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    try:
        summary = build_manifests(
            root, resolve(args.split_config), resolve(args.gt_root),
            resolve(args.image_root), resolve(args.output_root), resolve(args.summary),
        )
    except (ManifestError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
