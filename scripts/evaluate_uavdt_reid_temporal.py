#!/usr/bin/env python3
"""Evaluate a trained FastReID checkpoint on temporal UAVDT validation splits."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTREID_PARENT = PROJECT_ROOT / "baselines/BoT-SORT"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(FASTREID_PARENT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from fast_reid.fastreid.config import CfgNode, get_cfg  # noqa: E402
from fast_reid.fastreid.data.data_utils import read_image  # noqa: E402
from fast_reid.fastreid.data.transforms import build_transforms  # noqa: E402
from fast_reid.fastreid.modeling.meta_arch import build_model  # noqa: E402
from vera_mot.fastreid_uavdt import EXPECTED, TRAIN_DATASET_NAME, VAL_DATASET_NAME  # noqa: E402
from vera_mot.reid_data import ManifestError, sha256_file, validate_split_config  # noqa: E402
from vera_mot.reid_temporal_eval import (  # noqa: E402
    PROTOCOL_VERSION, build_temporal_splits, evaluate_sequence_local, l2_normalize,
    read_registry, split_manifest, write_evaluation_outputs,
)


class CropDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(read_image(self.paths[index])), self.paths[index]


def load_split(path: Path) -> dict:
    # The committed split file is deliberately JSON-compatible YAML.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read split configuration: {exc}") from exc
    validate_split_config(value)
    return value


def load_config(path: Path, device: str):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.VERA_MOT = CfgNode({"SEED": 42, "IDENTITIES_PER_BATCH": 4,
                            "TRAIN_DATASET": TRAIN_DATASET_NAME,
                            "VALIDATION_DATASET": VAL_DATASET_NAME,
                            "VALIDATION_EVALUATOR_ENABLED": False})
    cfg.merge_from_file(str(path))
    cfg.set_new_allowed(False)
    cfg.defrost()
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = EXPECTED["train"]["identities"]
    cfg.freeze()
    if list(cfg.INPUT.SIZE_TEST) != [256, 256] or cfg.MODEL.BACKBONE.DEPTH != "50x" or not cfg.MODEL.BACKBONE.WITH_IBN:
        raise ManifestError("configuration is not the pinned UAVDT R50-IBN model")
    return cfg


def load_checkpoint_strict(model, path: Path) -> None:
    if not path.is_file():
        raise ManifestError(f"checkpoint is absent: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ManifestError("checkpoint does not contain a model state dictionary")
        model.load_state_dict(state, strict=True)
    except ManifestError:
        raise
    except Exception as exc:
        raise ManifestError(f"checkpoint is incompatible with the pinned model: {exc}") from exc


def verify_crop_contract(registry_path: Path, paths: set[str], verify_hashes: bool) -> None:
    approved = (PROJECT_ROOT / "datasets/UAVDT/reid_crops/val").resolve()
    with registry_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["crop_path"]: row for row in csv.DictReader(handle)}
    for value in sorted(paths):
        crop = Path(value)
        full = crop if crop.is_absolute() else PROJECT_ROOT / crop
        full = full.resolve()
        if not full.is_relative_to(approved):
            raise ManifestError(f"crop is outside approved validation root: {value}")
        if not full.is_file():
            raise ManifestError(f"missing crop: {value}")
        if verify_hashes and sha256_file(full) != rows[value]["sha256"]:
            raise ManifestError(f"crop checksum mismatch: {value}")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-identities", type=int)
    parser.add_argument("--verify-crops", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.batch_size < 1 or args.num_workers < 0:
            raise ManifestError("batch-size must be positive and num-workers non-negative")
        config, checkpoint, registry = args.config.resolve(), args.checkpoint.resolve(), args.registry.resolve()
        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise ManifestError(f"output directory is non-empty (use --overwrite): {output}")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ManifestError("CUDA was requested but is unavailable")
        for label, path in (("configuration", config), ("registry", registry)):
            if not path.is_file():
                raise ManifestError(f"{label} is absent: {path}")
        split_config = load_split(PROJECT_ROOT / "configs/uavdt_reid_split.yaml")
        observations = read_registry(registry)
        observed_sequences = {row.sequence for row in observations}
        expected_sequences = set(split_config["validation_sequences"])
        if observed_sequences != expected_sequences:
            raise ManifestError(
                "registry must contain exactly the six validation sequences; "
                f"missing={sorted(expected_sequences - observed_sequences)}, "
                f"unexpected={sorted(observed_sequences - expected_sequences)}"
            )
        splits, rejected = build_temporal_splits(observations, split_config["validation_sequences"],
                                                  split_config["frozen_test_sequences"],
                                                  max_identities=args.max_identities)
        if not splits:
            raise ManifestError("no eligible identities remain")
        selected_paths = sorted({row.crop_path for item in splits for row in (*item.gallery, *item.query)})
        verify_crop_contract(registry, set(selected_paths), args.verify_crops)
        absolute_paths = {value: str((Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value).resolve())
                          for value in selected_paths}
        relative_by_absolute = {absolute: relative for relative, absolute in absolute_paths.items()}
        cfg = load_config(config, args.device)
        model = build_model(cfg)
        load_checkpoint_strict(model, checkpoint)
        model.eval()
        dataset = CropDataset([absolute_paths[path] for path in selected_paths], build_transforms(cfg, is_train=False))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.device == "cuda")
        embeddings = {}
        with torch.inference_mode():
            for images, paths in loader:
                features = model({"images": images.to(args.device)})
                features = torch.nn.functional.normalize(features, p=2, dim=1).cpu().numpy()
                for path, feature in zip(paths, features):
                    relative = relative_by_absolute[path]
                    embeddings[relative] = l2_normalize(np.asarray(feature).reshape(1, -1))[0]
        result = evaluate_sequence_local(splits, embeddings)
        metadata = {
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "configuration_path": str(config), "configuration_sha256": sha256_file(config),
            "registry_path": str(registry), "registry_sha256": sha256_file(registry),
            "git_commit_sha": git_sha(), "protocol_version": PROTOCOL_VERSION,
            "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_sequences": split_config["validation_sequences"],
        }
        write_evaluation_outputs(output, split_manifest(splits, rejected), result, metadata)
        print(json.dumps({"output_dir": str(output), "macro_average": result["macro_average"],
                          "micro_average": result["micro_average"]}, indent=2, sort_keys=True))
        return 0
    except (ManifestError, OSError, RuntimeError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
