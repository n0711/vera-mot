#!/usr/bin/env python3
"""Evaluate the six saved UAVDT ReID checkpoints with one fixed validation protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vera_mot.fastreid_uavdt import EXPECTED
from vera_mot.reid_checkpoint_sweep import build_plan, record_result, reuse_result
from vera_mot.reid_data import ManifestError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "experiments/reid_training/uavdt_sbs_R50_ibn")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/fastreid/uavdt_sbs_R50_ibn.yml")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "datasets/UAVDT/reid_manifests/val_crop_registry.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments/reid_evaluation/checkpoints")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-identities", type=int)
    parser.add_argument("--verify-crops", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="validate inputs and print the plan without inference or writes")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="reuse only fingerprint- and checksum-matching completed runs")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_plan(PROJECT_ROOT, args.checkpoint_dir, args.config, args.registry, args.output_dir,
                          device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
                          max_identities=args.max_identities, verify_crops=args.verify_crops)
        if plan["jobs"][0]["inputs"]["registry_sha256"] != EXPECTED["val"]["registry_sha256"]:
            raise ManifestError("stale or modified validation crop registry checksum")
        if args.plan_only:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()) and not (args.resume or args.overwrite):
            raise ManifestError("sweep output is non-empty; use --resume or --overwrite")
        # Validate all resumable runs before starting any expensive inference.
        reusable = {}
        for job in plan["jobs"]:
            directory = Path(job["output_dir"])
            if directory.exists() and any(directory.iterdir()) and args.resume:
                reusable[job["checkpoint"]] = reuse_result(job)
        output.mkdir(parents=True, exist_ok=True)
        # A failed replacement must not leave an older comparison looking current.
        (output / "comparison.json").unlink(missing_ok=True)
        (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        comparison = []
        for job in plan["jobs"]:
            if job["checkpoint"] in reusable:
                result = reusable[job["checkpoint"]]
            else:
                command = [sys.executable, str(PROJECT_ROOT / "scripts/evaluate_uavdt_reid_temporal.py"),
                           "--config", str(args.config.resolve()), "--checkpoint", job["inputs"]["checkpoint_path"],
                           "--registry", str(args.registry.resolve()), "--output-dir", job["output_dir"],
                           "--device", args.device, "--batch-size", str(args.batch_size),
                           "--num-workers", str(args.num_workers)]
                for enabled, option in ((args.verify_crops, "--verify-crops"), (args.overwrite, "--overwrite")):
                    if enabled:
                        command.append(option)
                if args.max_identities is not None:
                    command.extend(["--max-identities", str(args.max_identities)])
                print(f"Evaluating {job['checkpoint']}", flush=True)
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                result = record_result(job)
            if comparison and result["split_fingerprint"] != comparison[0]["split_fingerprint"]:
                raise ManifestError("checkpoints used different query/gallery splits")
            comparison.append(result)
        report = {"primary_checkpoint": plan["primary_checkpoint"], "evaluation_scope": plan["evaluation_scope"],
                  "checkpoint_selection": plan["checkpoint_selection"], "checkpoints": comparison}
        (output / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {output / 'comparison.json'}", flush=True)
        return 0
    except (ManifestError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
