#!/usr/bin/env python3
"""Validate the UAVDT FastReID configuration and pull data-only batches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTREID_PARENT = PROJECT_ROOT / "baselines/BoT-SORT"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(FASTREID_PARENT))

import torch  # noqa: E402
from torch.utils.data import BatchSampler, DataLoader  # noqa: E402
from fast_reid.fastreid.config import CfgNode, get_cfg  # noqa: E402
from fast_reid.fastreid.data.build import fast_batch_collator  # noqa: E402
from fast_reid.fastreid.data.common import CommDataset  # noqa: E402
from fast_reid.fastreid.data.samplers import BalancedIdentitySampler  # noqa: E402
from fast_reid.fastreid.data.transforms import build_transforms  # noqa: E402
from vera_mot.fastreid_uavdt import (  # noqa: E402
    EXPECTED, TRAIN_DATASET_NAME, VAL_DATASET_NAME, load_uavdt_registry,
    register_fastreid_datasets, validate_batch,
)
from vera_mot.reid_data import ManifestError  # noqa: E402


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


def validate_config(cfg) -> None:
    expected = {
        "datasets": tuple(cfg.DATASETS.NAMES) == (TRAIN_DATASET_NAME,),
        "tests_disabled": len(cfg.DATASETS.TESTS) == 0,
        "train_size": list(cfg.INPUT.SIZE_TRAIN) == [256, 256],
        "test_size": list(cfg.INPUT.SIZE_TEST) == [256, 256],
        "batch": cfg.SOLVER.IMS_PER_BATCH == 16,
        "instances": cfg.DATALOADER.NUM_INSTANCE == 4,
        "identities_per_batch": cfg.VERA_MOT.IDENTITIES_PER_BATCH == 4,
        "seed": cfg.VERA_MOT.SEED == 42,
        "weights_empty": cfg.MODEL.WEIGHTS == "",
        "imagenet_pretrain": cfg.MODEL.BACKBONE.PRETRAIN is True,
        "resnet50_ibn": cfg.MODEL.BACKBONE.DEPTH == "50x" and cfg.MODEL.BACKBONE.WITH_IBN is True,
        "losses": tuple(cfg.MODEL.LOSSES.NAME) == ("CrossEntropyLoss", "TripletLoss"),
        "amp": cfg.SOLVER.AMP.ENABLED is True,
        "validation_evaluator_disabled": cfg.VERA_MOT.VALIDATION_EVALUATOR_ENABLED is False,
    }
    failed = [key for key, passed in expected.items() if not passed]
    if failed:
        raise ManifestError(f"configuration invariants failed: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/fastreid/uavdt_sbs_R50_ibn.yml")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--verify-crops", action="store_true")
    args = parser.parse_args()
    if args.batches < 1 or args.batches > 4:
        parser.error("--batches must be between 1 and 4")
    try:
        cfg = load_config(args.config)
        validate_config(cfg)
        train_registry = load_uavdt_registry(PROJECT_ROOT, "train", verify_crops=args.verify_crops)
        val_registry = load_uavdt_registry(PROJECT_ROOT, "val", verify_crops=args.verify_crops)
        datasets = register_fastreid_datasets(
            PROJECT_ROOT, verify_crops=args.verify_crops,
            loaded={"train": train_registry, "val": val_registry},
        )
        registered_train = datasets.get(TRAIN_DATASET_NAME)(root=str(PROJECT_ROOT), verbose=False)
        registered_val = datasets.get(VAL_DATASET_NAME)(root=str(PROJECT_ROOT), verbose=False)
        if len(registered_train.train) != EXPECTED["train"]["images"] or len(registered_val.train) != EXPECTED["val"]["images"]:
            raise ManifestError("registered FastReID dataset counts are incorrect")
        train_dataset = CommDataset(train_registry.fastreid_items, build_transforms(cfg, is_train=True), relabel=True)
        sampler = BalancedIdentitySampler(
            train_dataset.img_items, cfg.SOLVER.IMS_PER_BATCH,
            cfg.DATALOADER.NUM_INSTANCE, seed=cfg.VERA_MOT.SEED,
        )
        loader = DataLoader(
            train_dataset, batch_sampler=BatchSampler(sampler, cfg.SOLVER.IMS_PER_BATCH, True),
            num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=fast_batch_collator,
            pin_memory=False,
        )
        batches = []
        iterator = iter(loader)
        for _ in range(args.batches):
            batch = next(iterator)
            composition = validate_batch(batch["targets"].tolist())
            if list(batch["images"].shape) != [16, 3, 256, 256]:
                raise ManifestError(f"unexpected tensor shape: {list(batch['images'].shape)}")
            paths = [Path(path).resolve() for path in batch["img_paths"]]
            approved = (PROJECT_ROOT / "datasets/UAVDT/reid_crops/train").resolve()
            if any(not path.is_relative_to(approved) for path in paths):
                raise ManifestError("batch contains a path outside the train crop root")
            if min(composition["labels"]) < 0 or max(composition["labels"]) >= EXPECTED["train"]["identities"]:
                raise ManifestError("batch label outside contiguous training range")
            batches.append({"tensor_shape": list(batch["images"].shape), **composition})
        report = {
            "configuration": str(args.config.relative_to(PROJECT_ROOT)),
            "registered_dataset_names": [TRAIN_DATASET_NAME, VAL_DATASET_NAME],
            "train": {"identities": len(train_registry.identity_to_label), "images": len(train_registry.records)},
            "validation": {"identities": len(val_registry.identity_to_label), "images": len(val_registry.records)},
            "identity_labels": {"minimum": 0, "maximum": max(train_registry.identity_to_label.values()), "contiguous": True},
            "batches": batches, "frozen_test_records": 0, "private_video_records": 0,
            "missing_crops": 0, "invalid_paths": 0, "crop_verification": args.verify_crops,
            "training_started": False, "model_constructed": False, "forward_pass": False,
            "validation_evaluator": "disabled: same-sequence temporal retrieval semantics unresolved",
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ManifestError, KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
