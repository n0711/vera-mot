#!/usr/bin/env python3
"""Print a read-only UAVDT GT comparison report. Exit 1 for structural differences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vera_mot.reid_ablation_audit import AuditError
from vera_mot.uavdt_gt_compare import compare_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--ignore-regions", type=Path)
    args = parser.parse_args()
    try:
        report = compare_files(
            ROOT, sequence=args.sequence, source=args.source,
            reference=args.reference, ignore_regions=args.ignore_regions,
        )
    except (AuditError, OSError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["structural_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
