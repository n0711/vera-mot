from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from vera_mot.reid_data import ManifestError
from vera_mot.reid_quality import (
    IDENTITY_FIELDS, PER_CROP_FIELDS, aggregate_identities, analyze,
    appearance_features, extract_record, join_inputs,
)


MANIFEST_FIELDS = [
    "sequence", "frame", "identity", "target_id", "category", "identity_majority_category",
    "identity_category_conflict", "out_of_view", "occlusion", "original_left", "original_top",
    "original_width", "original_height", "clipped_left", "clipped_top", "clipped_width",
    "clipped_height", "clipped_area", "source_image", "planned_crop_path", "split", "sampling_index",
]
REGISTRY_FIELDS = [
    "split", "sequence", "frame", "identity", "category", "identity_majority_category",
    "identity_category_conflict", "crop_path", "width", "height", "file_size", "sha256",
    "source_image", "source_region_sha256",
]


def write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_record(root: Path, *, frame: int = 1, sequence: str = "M0101", identity: str = "M0101_1",
                left: int = 10, top: int = 20, width: int = 20, height: int = 10,
                original: tuple[int, int, int, int] | None = None, value: int = 64) -> tuple[dict, dict]:
    source_rel = f"sources/{sequence}/img{frame:06d}.png"
    crop_rel = f"crops/train/{identity}/{sequence}_{frame}.png"
    source_path, crop_path = root / source_rel, root / crop_rel
    source_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((80, 100, 3), 127, dtype=np.uint8)).save(source_path)
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(crop_path)
    original = original or (left, top, width, height)
    manifest = {
        "sequence": sequence, "frame": frame, "identity": identity, "target_id": 1, "category": 1,
        "identity_majority_category": 1, "identity_category_conflict": 0, "out_of_view": 2,
        "occlusion": 1, "original_left": original[0], "original_top": original[1],
        "original_width": original[2], "original_height": original[3], "clipped_left": left,
        "clipped_top": top, "clipped_width": width, "clipped_height": height,
        "clipped_area": width * height, "source_image": source_rel, "planned_crop_path": crop_rel,
        "split": "train", "sampling_index": frame - 1,
    }
    registry = {
        "split": "train", "sequence": sequence, "frame": frame, "identity": identity, "category": 1,
        "identity_majority_category": 1, "identity_category_conflict": 0, "crop_path": crop_rel,
        "width": width, "height": height, "file_size": crop_path.stat().st_size,
        "sha256": sha256(crop_path), "source_image": source_rel, "source_region_sha256": "unused",
    }
    return manifest, registry


def test_exact_geometry_aspect_relative_area_and_border_distances(tmp_path: Path):
    manifest, registry = make_record(tmp_path)
    row = extract_record(tmp_path, {k: str(v) for k, v in manifest.items()},
                         {k: str(v) for k, v in registry.items()}, verify_crop=True)
    assert (row["width"], row["height"], row["area"]) == (20, 10, 200)
    assert row["aspect_ratio"] == 2
    assert row["relative_area"] == pytest.approx(200 / 8000)
    assert (row["relative_width"], row["relative_height"]) == pytest.approx((.2, .125))
    assert (row["distance_right"], row["distance_bottom"], row["minimum_border_distance"]) == (70, 50, 10)
    assert not any(row[field] for field in ("touches_left_border", "touches_top_border", "touches_right_border", "touches_bottom_border"))


def test_border_touch_and_clipped_box_detection(tmp_path: Path):
    manifest, registry = make_record(tmp_path, left=0, top=70, width=20, height=10,
                                     original=(-2, 70, 22, 12))
    row = extract_record(tmp_path, {k: str(v) for k, v in manifest.items()},
                         {k: str(v) for k, v in registry.items()}, verify_crop=False)
    assert row["minimum_border_distance"] == 0
    assert row["touches_left_border"] == 1
    assert row["touches_bottom_border"] == 1
    assert row["original_box_was_clipped"] == 1


def test_constant_image_has_zero_contrast_and_laplacian():
    features = appearance_features(np.full((12, 13), 97, dtype=np.uint8))
    assert features["grayscale_standard_deviation"] == 0
    assert features["rms_contrast"] == 0
    assert features["laplacian_variance"] == 0


def test_sharp_edge_has_greater_laplacian_variance():
    constant = np.full((32, 32), 127, dtype=np.uint8)
    edge = np.zeros((32, 32), dtype=np.uint8)
    edge[:, 16:] = 255
    assert appearance_features(edge)["laplacian_variance"] > appearance_features(constant)["laplacian_variance"]


def test_entropy_and_saturation_fractions():
    gray = np.array([[0, 0, 255, 255], [0, 0, 255, 255]], dtype=np.uint8)
    features = appearance_features(gray)
    assert features["entropy"] == pytest.approx(1.0)
    assert features["saturated_dark_fraction"] == .5
    assert features["saturated_bright_fraction"] == .5


def test_deterministic_output_order_and_max_record_selection(tmp_path: Path):
    pairs = [make_record(tmp_path, frame=frame, value=frame) for frame in (3, 1, 2)]
    manifest_path, registry_path = tmp_path / "manifest.csv", tmp_path / "registry.csv"
    write_csv(manifest_path, MANIFEST_FIELDS, [pair[0] for pair in pairs])
    write_csv(registry_path, REGISTRY_FIELDS, [pair[1] for pair in reversed(pairs)])
    first = analyze(tmp_path, manifest_path, registry_path, tmp_path / "out1", max_records=2)
    second = analyze(tmp_path, manifest_path, registry_path, tmp_path / "out2", max_records=2)
    assert first["record_count"] == second["record_count"] == 2
    with (tmp_path / "out1/per_crop.csv").open(newline="", encoding="utf-8") as handle:
        rows1 = list(csv.DictReader(handle))
    with (tmp_path / "out2/per_crop.csv").open(newline="", encoding="utf-8") as handle:
        rows2 = list(csv.DictReader(handle))
    assert [row["frame"] for row in rows1] == ["1", "2"]
    assert rows1 == rows2


def test_duplicate_join_rejected(tmp_path: Path):
    manifest, registry = make_record(tmp_path)
    write_csv(tmp_path / "m.csv", MANIFEST_FIELDS, [manifest, manifest])
    write_csv(tmp_path / "r.csv", REGISTRY_FIELDS, [registry])
    with pytest.raises(ManifestError, match="duplicate manifest"):
        join_inputs(tmp_path / "m.csv", tmp_path / "r.csv")


def test_missing_join_rejected(tmp_path: Path):
    manifest, registry = make_record(tmp_path)
    registry["frame"] = 2
    write_csv(tmp_path / "m.csv", MANIFEST_FIELDS, [manifest])
    write_csv(tmp_path / "r.csv", REGISTRY_FIELDS, [registry])
    with pytest.raises(ManifestError, match="join mismatch"):
        join_inputs(tmp_path / "m.csv", tmp_path / "r.csv")


def test_crop_checksum_mismatch_rejected_when_verified(tmp_path: Path):
    manifest, registry = make_record(tmp_path)
    registry["sha256"] = "0" * 64
    with pytest.raises(ManifestError, match="checksum mismatch"):
        extract_record(tmp_path, {k: str(v) for k, v in manifest.items()},
                       {k: str(v) for k, v in registry.items()}, verify_crop=True)
    extract_record(tmp_path, {k: str(v) for k, v in manifest.items()},
                   {k: str(v) for k, v in registry.items()}, verify_crop=False)


def test_frozen_test_rejected(tmp_path: Path):
    manifest, registry = make_record(tmp_path, sequence="M0203", identity="M0203_1")
    write_csv(tmp_path / "m.csv", MANIFEST_FIELDS, [manifest])
    write_csv(tmp_path / "r.csv", REGISTRY_FIELDS, [registry])
    with pytest.raises(ManifestError, match="frozen-test"):
        join_inputs(tmp_path / "m.csv", tmp_path / "r.csv")


def test_identity_aggregation():
    base = {field: 0 for field in PER_CROP_FIELDS}
    rows = []
    for frame, width, occlusion, border in ((1, 10, 2, 1), (5, 20, 1, 0)):
        row = dict(base, split="train", identity="M0101_1", frame=frame, width=width,
                   height=10, area=width * 10, relative_area=width / 1000,
                   laplacian_variance=width, tenengrad_mean=width * 2, rms_contrast=width / 100,
                   entropy=1 + width / 100, touches_left_border=border, occlusion=occlusion,
                   out_of_view=0)
        rows.append(row)
    aggregate = aggregate_identities(rows)[0]
    assert aggregate["crop_count"] == 2
    assert (aggregate["frame_minimum"], aggregate["frame_maximum"]) == (1, 5)
    assert aggregate["width_median"] == 15
    assert aggregate["proportion_touching_any_image_border"] == .5
    assert json.loads(aggregate["occlusion_code_counts"]) == {"1": 1, "2": 1}
