#!/usr/bin/env python3
"""Generate a fresh, provenance-audited GT file for pinned TrackEval; no evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vera_mot.reid_ablation_audit import AuditError
from vera_mot.uavdt_gt_convert import prepare_conversion, write_conversion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="One of six frozen ReID validation sequences")
    parser.add_argument("--source", type=Path, required=True, help="Original <sequence>_gt_whole.txt")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Existing fresh directory, separate from original GT and frozen benchmarks")
    args = parser.parse_args()
    # The source and output must not be the same directory.
    if args.source.resolve().parent == args.output_dir.resolve():
        parser.error("output directory must differ from the original source directory")
    historical = ROOT / "experiments/tracking_baselines/botsort_no_reid"
    if args.output_dir.resolve().is_relative_to(historical.resolve()):
        parser.error("cannot write into frozen no-ReID baseline directories")
    if args.output_dir.resolve().is_relative_to((ROOT / "datasets/UAVDT").resolve()):
        parser.error("cannot write GT conversion into original UAVDT source directories")
    try:
        converted, report = prepare_conversion(
            ROOT, sequence=args.sequence, source=args.source,
        )
        gt_path, metadata_path = write_conversion(
            args.output_dir, sequence=args.sequence, converted=converted, report=report,
        )
    except (AuditError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": report["status"],
        "sequence": args.sequence,
        "source_rows": report["record_count"],
        "converted_gt_file": str(gt_path),
        "conversion_provenance_file": str(metadata_path),
        "note": "Only formatting and source-policy checks; no TrackEval accuracy results",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
