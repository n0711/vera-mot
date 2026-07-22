"""Project-owned UAVDT registry adapter for the pinned FastReID revision."""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from vera_mot.reid_crops import safe_relative_path
from vera_mot.reid_data import ManifestError, sha256_file


TRAIN_DATASET_NAME = "UAVDTReIDTrain"
VAL_DATASET_NAME = "UAVDTReIDVal"
APPROVED_CROP_ROOT = "datasets/UAVDT/reid_crops"
EXPECTED = {
    "train": {
        "registry_sha256": "b89819789c4782f19ce132b81a114770554ee7e34e17b772cc56af05e5a515c5",
        "dataset_sha256": "2eb9d723697913c3e1f9df2892a27a8dccb617865bf0bd2833d0edce3cfedb80",
        "identities": 1048, "images": 64338,
    },
    "val": {
        "registry_sha256": "7f0c824103d5e2915ffb4f644dbe3bb6bf4f89026e10e9cdfef9df5c999eeb68",
        "dataset_sha256": "b468a03524ea5419df4b4b0b0068d3c9e9362b9f397dfc654525cb5a330c9d06",
        "identities": 264, "images": 17121,
    },
}
FROZEN_TEST_SEQUENCES = frozenset({
    "M0203", "M0205", "M0208", "M0209", "M0403", "M0601", "M0602",
    "M0606", "M0701", "M0801", "M0802", "M1001", "M1004", "M1007",
    "M1009", "M1101", "M1301", "M1302", "M1303", "M1401",
})


@dataclass(frozen=True)
class UAVDTReIDRecord:
    crop_path: str
    identity: str
    label: int
    sequence: str
    camera_id: int
    frame: int
    category: int


@dataclass(frozen=True)
class UAVDTRegistry:
    split: str
    records: tuple[UAVDTReIDRecord, ...]
    identity_to_label: dict[str, int]
    sequence_to_camera: dict[str, int]
    registry_sha256: str
    dataset_sha256: str

    @property
    def fastreid_items(self) -> list[tuple[str, int, int]]:
        return [(record.crop_path, record.label, record.camera_id) for record in self.records]


def _dataset_checksum(rows: Iterable[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["crop_path"]):
        digest.update(row["crop_path"].encode("utf-8"))
        digest.update(row["sha256"].encode("ascii"))
    return digest.hexdigest()


def load_uavdt_registry(
    project_root: Path,
    split: str,
    *,
    verify_crops: bool = False,
    expected: dict | None = EXPECTED,
) -> UAVDTRegistry:
    """Load one frozen crop registry with deterministic labels and cameras.

    Camera IDs represent sequence-stream membership only. They make no claim about a
    physical UAV/camera shared across sequences.
    """
    if split not in ("train", "val"):
        raise ManifestError(f"unsupported UAVDT ReID split: {split}")
    root = project_root.resolve()
    registry_path = root / f"datasets/UAVDT/reid_manifests/{split}_crop_registry.csv"
    if not registry_path.is_file():
        raise ManifestError(f"missing crop registry: {registry_path}")
    registry_sha = sha256_file(registry_path)
    if expected and registry_sha != expected[split]["registry_sha256"]:
        raise ManifestError(f"stale or modified {split} crop registry checksum")
    with registry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ManifestError(f"empty {split} crop registry")

    identities = sorted({row["identity"] for row in rows})
    identity_to_label = {identity: label for label, identity in enumerate(identities)}
    sequences = sorted({row["sequence"] for row in rows})
    sequence_to_camera = {sequence: camera for camera, sequence in enumerate(sequences)}
    counts = Counter(row["identity"] for row in rows)
    too_short = sorted(identity for identity, count in counts.items() if count < 4)
    if too_short:
        raise ManifestError(f"identities with fewer than four samples: {too_short[:3]}")

    seen = set()
    records = []
    for line_number, row in enumerate(rows, 2):
        if row["split"] != split:
            raise ManifestError(f"unsupported split at registry line {line_number}: {row['split']}")
        sequence = row["sequence"]
        if sequence in FROZEN_TEST_SEQUENCES:
            raise ManifestError(f"frozen-test record at registry line {line_number}: {sequence}")
        crop_relative = safe_relative_path(row["crop_path"], required_prefix=APPROVED_CROP_ROOT)
        split_root = Path(APPROVED_CROP_ROOT) / split
        if not crop_relative.is_relative_to(split_root):
            raise ManifestError(f"crop is outside approved {split} root: {crop_relative}")
        # Only materialized UAVDT crop records are accepted; source/private paths are never inputs.
        source_relative = safe_relative_path(
            row["source_image"], required_prefix="datasets/UAVDT/full_mot/raw/UAV-benchmark-M"
        )
        if source_relative.parts[-2] != sequence or crop_relative.parts[3] != split:
            raise ManifestError(f"sequence or split path mismatch at registry line {line_number}")
        key = (sequence, int(row["frame"]), row["identity"])
        if key in seen:
            raise ManifestError(f"duplicate registry record: {key}")
        seen.add(key)
        crop_path = root / crop_relative
        if not crop_path.is_file():
            raise ManifestError(f"missing crop: {crop_relative}")
        if verify_crops:
            if sha256_file(crop_path) != row["sha256"]:
                raise ManifestError(f"modified crop: {crop_relative}")
            try:
                with Image.open(crop_path) as image:
                    image.load()
                    if image.format != "PNG" or image.size != (int(row["width"]), int(row["height"])):
                        raise ManifestError(f"invalid crop dimensions or format: {crop_relative}")
            except OSError as exc:
                raise ManifestError(f"unreadable crop: {crop_relative}") from exc
        records.append(UAVDTReIDRecord(
            crop_path=str(crop_path), identity=row["identity"],
            label=identity_to_label[row["identity"]], sequence=sequence,
            camera_id=sequence_to_camera[sequence], frame=int(row["frame"]),
            category=int(row["category"]),
        ))

    dataset_sha = _dataset_checksum(rows)
    if expected:
        contract = expected[split]
        if dataset_sha != contract["dataset_sha256"]:
            raise ManifestError(f"stale or modified {split} dataset checksum")
        if len(records) != contract["images"] or len(identities) != contract["identities"]:
            raise ManifestError(f"{split} registry counts do not match frozen contract")
    if sorted(identity_to_label.values()) != list(range(len(identity_to_label))):
        raise ManifestError("identity labels are not contiguous")
    return UAVDTRegistry(
        split=split, records=tuple(records), identity_to_label=identity_to_label,
        sequence_to_camera=sequence_to_camera, registry_sha256=registry_sha,
        dataset_sha256=dataset_sha,
    )


def validate_batch(targets, batch_size: int = 16, instances_per_identity: int = 4) -> dict:
    labels = [int(value) for value in targets]
    counts = Counter(labels)
    if len(labels) != batch_size:
        raise ManifestError(f"expected batch size {batch_size}, found {len(labels)}")
    if len(counts) != batch_size // instances_per_identity:
        raise ManifestError(f"expected {batch_size // instances_per_identity} identities, found {len(counts)}")
    if set(counts.values()) != {instances_per_identity}:
        raise ManifestError(f"expected {instances_per_identity} samples per identity, found {dict(counts)}")
    return {"batch_size": len(labels), "distinct_identities": len(counts),
            "samples_per_identity": instances_per_identity, "labels": sorted(counts)}


def register_fastreid_datasets(
    project_root: Path, *, verify_crops: bool = False,
    loaded: dict[str, UAVDTRegistry] | None = None,
):
    """Register explicit classes in the pinned FastReID registry."""
    from fast_reid.fastreid.data.datasets import DATASET_REGISTRY
    from fast_reid.fastreid.data.datasets.bases import ImageDataset

    root = project_root.resolve()

    def make_dataset_class(name: str, split: str):
        def __init__(self, root=None, **kwargs):
            registry = loaded[split] if loaded and split in loaded else load_uavdt_registry(
                project_root, split, verify_crops=verify_crops
            )
            # Validation data stays in its own collection for loader/count checks.
            # Query/gallery remain empty until a defensible temporal evaluator exists.
            ImageDataset.__init__(self, registry.fastreid_items, [], [], **kwargs)
            self.uavdt_registry = registry
        return type(name, (ImageDataset,), {
            "__init__": __init__, "dataset_name": name,
            "sequence_camera_semantics": "deterministic sequence-stream index",
        })

    registered = set(DATASET_REGISTRY._obj_map)
    for name, split in ((TRAIN_DATASET_NAME, "train"), (VAL_DATASET_NAME, "val")):
        if name not in registered:
            DATASET_REGISTRY.register(make_dataset_class(name, split))
    return DATASET_REGISTRY
