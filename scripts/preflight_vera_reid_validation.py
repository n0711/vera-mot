#!/usr/bin/env python3
"""Fail-closed source and annotation preflight for a validation ReID pair."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vera_mot.reid_ablation_audit import AuditError  # noqa: E402
from vera_mot.reid_eval_preflight import build_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="Frozen ReID validation sequence, e.g. M0201")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Original nine-column <sequence>_gt.txt")
    parser.add_argument("--ungated", type=Path, required=True, help="Ungated ReID MOT output")
    parser.add_argument("--gated", type=Path, required=True, help="VERA ReID MOT output")
    parser.add_argument("--video", type=Path, required=True, help="Original video file used in both runs")
    parser.add_argument("--detections", type=Path, required=True, help="Original detection CSV used in both runs")
    parser.add_argument("--expected-threshold", type=float, required=True, help="Predeclared VERA threshold")
    parser.add_argument("--output", type=Path, required=True, help="New aggregate report JSON")
    args = parser.parse_args()
    protected = {
        path.resolve()
        for path in (args.ground_truth, args.video, args.detections,
                     args.ungated, args.gated,
                     args.ungated.with_suffix(".provenance.json"),
                     args.gated.with_suffix(".provenance.json"),
                     ROOT / "configs/uavdt_reid_split.yaml",
                     ROOT / "docs/uavdt_mot_audit.json")
    }
    if args.output.resolve() in protected:
        parser.error("output cannot replace any source/evidence file")
    try:
        result = build_preflight(
            ROOT, sequence=args.sequence, ground_truth=args.ground_truth,
            ungated=args.ungated, gated=args.gated, video=args.video,
            detections=args.detections, expected_threshold=args.expected_threshold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except (AuditError, OSError) as exc:
        parser.error(str(exc))
    print("M4.8 preflight passed prerequisite consistency checks only.")
    print(f"Aggregate evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
