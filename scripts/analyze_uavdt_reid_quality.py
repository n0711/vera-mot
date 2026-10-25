#!/usr/bin/env python3
"""Extract deterministic UAVDT ReID crop-quality metadata."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vera_mot.reid_data import ManifestError
from vera_mot.reid_quality import analyze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-crops", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        summary = analyze(
            PROJECT_ROOT, args.manifest.resolve(), args.registry.resolve(), args.output_dir.resolve(),
            verify_crops=args.verify_crops, max_records=args.max_records, overwrite=args.overwrite,
        )
    except ManifestError as exc:
        parser.error(str(exc))
    print(f"wrote {summary['record_count']} {summary['split']} records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
