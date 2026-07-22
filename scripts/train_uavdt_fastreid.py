#!/usr/bin/env python3
"""Preflight and train the ordinary UAVDT FastReID baseline."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTREID_PARENT = PROJECT_ROOT / "baselines/BoT-SORT"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(FASTREID_PARENT))

import torch  # noqa: E402

from fast_reid.fastreid.config import CfgNode, get_cfg  # noqa: E402
from fast_reid.fastreid.engine import DefaultTrainer, default_setup  # noqa: E402
from fast_reid.fastreid.utils.events import EventStorage  # noqa: E402
from vera_mot.fastreid_uavdt import (  # noqa: E402
    EXPECTED,
    TRAIN_DATASET_NAME,
    VAL_DATASET_NAME,
    load_uavdt_registry,
    register_fastreid_datasets,
)
from vera_mot.reid_data import ManifestError  # noqa: E402

LOG = logging.getLogger("vera_mot.reid_training")
APPROVED_FILENAME = "resnet50_ibn_a-d9d0bb7b.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.VERA_MOT = CfgNode({
        "SEED": 42,
        "IDENTITIES_PER_BATCH": 4,
        "TRAIN_DATASET": TRAIN_DATASET_NAME,
        "VALIDATION_DATASET": VAL_DATASET_NAME,
        "VALIDATION_EVALUATOR_ENABLED": False,
    })
    cfg.merge_from_file(str(path))
    cfg.set_new_allowed(False)
    return cfg


def validate_config(cfg, weight_path: Path) -> None:
    checks = {
        "train dataset": tuple(cfg.DATASETS.NAMES) == (TRAIN_DATASET_NAME,),
        "evaluation disabled": not cfg.DATASETS.TESTS,
        "input": list(cfg.INPUT.SIZE_TRAIN) == [256, 256],
        "batch": cfg.SOLVER.IMS_PER_BATCH == 16,
        "sampler": cfg.DATALOADER.SAMPLER_TRAIN == "BalancedIdentitySampler",
        "instances": cfg.DATALOADER.NUM_INSTANCE == 4,
        "seed": cfg.VERA_MOT.SEED == 42,
        "weights empty": cfg.MODEL.WEIGHTS == "",
        "pretrain": cfg.MODEL.BACKBONE.PRETRAIN is True,
        "explicit pretrain": Path(cfg.MODEL.BACKBONE.PRETRAIN_PATH) == weight_path.relative_to(PROJECT_ROOT),
        "R50 IBN NL": cfg.MODEL.BACKBONE.DEPTH == "50x" and cfg.MODEL.BACKBONE.WITH_IBN is True and cfg.MODEL.BACKBONE.WITH_NL is True,
        "feature dimension": cfg.MODEL.BACKBONE.FEAT_DIM == 2048,
        "losses": tuple(cfg.MODEL.LOSSES.NAME) == ("CrossEntropyLoss", "TripletLoss"),
        "AMP": cfg.SOLVER.AMP.ENABLED is True,
        "epochs": cfg.SOLVER.MAX_EPOCH == 60,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ManifestError(f"configuration invariants failed: {failed}")


def prepare_cfg(base_cfg, output_dir: Path):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = str((PROJECT_ROOT / cfg.MODEL.BACKBONE.PRETRAIN_PATH).resolve())
    cfg.SEED = cfg.VERA_MOT.SEED
    cfg.freeze()
    return cfg


def verify_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ManifestError(f"training output is not empty: {path}")
    if (path / "last_checkpoint").exists():
        raise ManifestError(f"stale last_checkpoint exists: {path}")


def preflight(cfg, output_dir: Path) -> dict:
    trainer = DefaultTrainer(cfg)
    if trainer.cfg.MODEL.HEADS.NUM_CLASSES != EXPECTED["train"]["identities"]:
        raise ManifestError("classifier class count is not 1048")
    batch = next(iter(trainer.data_loader))
    shape = list(batch["images"].shape)
    counts = sorted(Counter(batch["targets"].tolist()).values())
    if shape != [16, 3, 256, 256] or counts != [4, 4, 4, 4]:
        raise ManifestError(f"invalid preflight batch: shape={shape}, counts={counts}")
    approved_root = (PROJECT_ROOT / "datasets/UAVDT/reid_crops/train").resolve()
    if any(not Path(p).resolve().is_relative_to(approved_root) for p in batch["img_paths"]):
        raise ManifestError("preflight batch escaped approved train crop root")

    torch.cuda.reset_peak_memory_stats()
    trainer.model.train()
    with EventStorage(0) as storage:
        trainer._trainer.run_step()
        latest = storage.latest()
    values = {
        name: float(value[0]) for name, value in latest.items()
        if name in {"loss_cls", "loss_triplet"}
    }
    total_loss = float(latest["total_loss"][0])
    if set(values) != {"loss_cls", "loss_triplet"} or not torch.isfinite(torch.tensor(total_loss)):
        raise FloatingPointError(f"invalid production training step metrics: {latest}")
    result = {
        "status": "pass",
        "input_shape": shape,
        "identity_counts": counts,
        "identities": sorted(Counter(batch["targets"].tolist())),
        "losses": values,
        "total_loss": total_loss,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_step": True,
        "classifier_classes": trainer.cfg.MODEL.HEADS.NUM_CLASSES,
    }
    del trainer, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/fastreid/uavdt_sbs_R50_ibn.yml")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    base_cfg = load_config(args.config.resolve())
    weight_path = (PROJECT_ROOT / base_cfg.MODEL.BACKBONE.PRETRAIN_PATH).resolve()
    validate_config(base_cfg, weight_path)
    if weight_path.name != APPROVED_FILENAME or not weight_path.is_file():
        raise ManifestError(f"approved initialization is absent: {weight_path}")

    train_registry = load_uavdt_registry(PROJECT_ROOT, "train", verify_crops=False)
    val_registry = load_uavdt_registry(PROJECT_ROOT, "val", verify_crops=False)
    if len(train_registry.records) != EXPECTED["train"]["images"] or len(val_registry.records) != EXPECTED["val"]["images"]:
        raise ManifestError("registry counts changed")
    register_fastreid_datasets(PROJECT_ROOT, loaded={"train": train_registry, "val": val_registry})

    run_root = PROJECT_ROOT / "experiments/reid_training"
    preflight_dir = run_root / "preflight_uavdt_sbs_R50_ibn"
    output_dir = run_root / "uavdt_sbs_R50_ibn"
    verify_empty_output(preflight_dir)
    verify_empty_output(output_dir)
    preflight_cfg = prepare_cfg(base_cfg, preflight_dir)
    setup_args = argparse.Namespace(config_file=str(args.config), eval_only=False, resume=False, num_gpus=1, num_machines=1, machine_rank=0, dist_url="auto", opts=[])
    default_setup(preflight_cfg, setup_args)
    result = preflight(preflight_cfg, preflight_dir)
    preflight_dir.mkdir(parents=True, exist_ok=True)
    (preflight_dir / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PREFLIGHT_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0

    verify_empty_output(output_dir)
    cfg = prepare_cfg(base_cfg, output_dir)
    default_setup(cfg, setup_args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    manifest = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": git_revision(PROJECT_ROOT),
        "botsort_revision": git_revision(FASTREID_PARENT),
        "initialization_path": str(weight_path.relative_to(PROJECT_ROOT)),
        "initialization_observed_sha256": sha256(weight_path),
        "train_identities": len(train_registry.identity_to_label),
        "train_images": len(train_registry.records),
        "validation_identities": len(val_registry.identity_to_label),
        "validation_images": len(val_registry.records),
        "frozen_test_records": 0,
        "private_records": 0,
        "epochs": trainer.max_epoch,
        "iterations_per_epoch": trainer.iters_per_epoch,
        "total_iterations": trainer.max_iter,
        "seed": cfg.SEED,
        "preflight": result,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    training_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    checkpoints = [
        {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.glob("*.pth"))
    ]
    completion = {
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": time.monotonic() - training_started,
        "epochs_completed": trainer.max_epoch,
        "iterations_completed": trainer.max_iter,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "checkpoints": checkpoints,
    }
    (output_dir / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
