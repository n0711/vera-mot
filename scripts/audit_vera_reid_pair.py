#!/usr/bin/env python3
"""Audit comparability and output integrity of two offline BoT-SORT ReID runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vera_mot.reid_ablation_audit import AuditError, audit_pair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ungated", type=Path, required=True, help="Ungated appearance-reid MOT file")
    parser.add_argument("--gated", type=Path, required=True, help="VERA-gated appearance-reid MOT file")
    parser.add_argument("--output", type=Path, required=True, help="New audit JSON path")
    args = parser.parse_args()
    # Never overwrite tracker outputs, evidence, or an existing audit.
    protected = {p.resolve() for m in (args.ungated, args.gated)
                 for p in (m, m.with_suffix(".provenance.json"))}
    if args.output.resolve() in protected:
        parser.error("audit output must not replace MOT/provenance inputs")
    try:
        report = audit_pair(args.ungated, args.gated)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except (AuditError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved read-only paired evidence audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
