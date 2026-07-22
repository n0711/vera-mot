"""Deterministic, CPU-only crop-quality measurements for UAVDT ReID data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vera_mot.fastreid_uavdt import FROZEN_TEST_SEQUENCES
from vera_mot.reid_data import ManifestError, sha256_file


PROTOCOL_VERSION = "1.0.0"
JOIN_FIELDS = ("split", "sequence", "frame", "identity")
NUMERIC_QUALITY_FIELDS = (
    "width", "height", "area", "aspect_ratio", "source_width", "source_height",
    "relative_width", "relative_height", "relative_area", "clipped_left", "clipped_top",
    "distance_right", "distance_bottom", "minimum_border_distance",
    "grayscale_mean", "grayscale_standard_deviation", "rms_contrast",
    "laplacian_variance", "tenengrad_mean", "entropy", "saturated_dark_fraction",
    "saturated_bright_fraction",
)
PER_CROP_FIELDS = (
    "split", "sequence", "frame", "identity", "category", "identity_majority_category",
    "identity_category_conflict", "crop_path", "source_image", "crop_sha256", "width",
    "height", "area", "aspect_ratio", "source_width", "source_height", "relative_width",
    "relative_height", "relative_area", "clipped_left", "clipped_top", "distance_right",
    "distance_bottom", "minimum_border_distance", "touches_left_border", "touches_top_border",
    "touches_right_border", "touches_bottom_border", "original_box_was_clipped", "occlusion",
    "out_of_view", "grayscale_mean", "grayscale_standard_deviation", "rms_contrast",
    "laplacian_variance", "tenengrad_mean", "entropy", "saturated_dark_fraction",
    "saturated_bright_fraction",
)
IDENTITY_FIELDS = (
    "split", "identity", "crop_count", "frame_minimum", "frame_maximum", "width_median",
    "height_median", "area_median", "relative_area_median", "laplacian_variance_median",
    "tenengrad_mean_median", "rms_contrast_median", "entropy_median",
    "proportion_touching_any_image_border", "occlusion_code_counts", "out_of_view_code_counts",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ManifestError(f"cannot read CSV: {path}") from exc
    if not rows:
        raise ManifestError(f"empty CSV: {path}")
    return rows


def _integer(row: dict[str, str], field: str, source: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"invalid integer field {field!r} in {source}") from exc


def _join_key(row: dict[str, str], path_field: str, source: Path) -> tuple[str, str, int, str, str]:
    try:
        return (row["split"], row["sequence"], int(row["frame"]), row["identity"], row[path_field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"invalid join fields in {source}") from exc


def join_inputs(manifest_path: Path, registry_path: Path) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Strictly join manifest and registry records using identity fields and crop path."""
    manifests, registries = _read_csv(manifest_path), _read_csv(registry_path)
    manifest_map: dict[tuple, dict] = {}
    registry_map: dict[tuple, dict] = {}
    for row in manifests:
        key = _join_key(row, "planned_crop_path", manifest_path)
        if key in manifest_map:
            raise ManifestError(f"duplicate manifest join key: {key}")
        manifest_map[key] = row
    for row in registries:
        key = _join_key(row, "crop_path", registry_path)
        if key in registry_map:
            raise ManifestError(f"duplicate registry join key: {key}")
        registry_map[key] = row
    missing_registry = sorted(set(manifest_map) - set(registry_map))
    missing_manifest = sorted(set(registry_map) - set(manifest_map))
    if missing_registry or missing_manifest:
        raise ManifestError(
            f"manifest/registry join mismatch: {len(missing_registry)} missing registry, "
            f"{len(missing_manifest)} missing manifest records"
        )
    joined = []
    for key in sorted(manifest_map):
        manifest, registry = manifest_map[key], registry_map[key]
        if manifest["sequence"] in FROZEN_TEST_SEQUENCES:
            raise ManifestError(f"frozen-test sequence in quality input: {manifest['sequence']}")
        for field in ("split", "sequence", "frame", "identity", "category",
                      "identity_majority_category", "identity_category_conflict", "source_image"):
            if manifest.get(field) != registry.get(field):
                raise ManifestError(f"manifest/registry {field} mismatch for {key}")
        if _integer(manifest, "clipped_width", manifest_path) != _integer(registry, "width", registry_path):
            raise ManifestError(f"manifest/registry width mismatch for {key}")
        if _integer(manifest, "clipped_height", manifest_path) != _integer(registry, "height", registry_path):
            raise ManifestError(f"manifest/registry height mismatch for {key}")
        joined.append((manifest, registry))
    return joined


def appearance_features(gray: np.ndarray) -> dict[str, float]:
    """Measure the specified appearance features from a uint8 grayscale image."""
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ManifestError("appearance input must be a non-empty uint8 grayscale image")
    gray_float = gray.astype(np.float64)
    gray_mean = float(gray_float.mean())
    mean = gray_mean / 255.0
    deviation = float(gray_float.std(ddof=0) / 255.0)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    return {
        "grayscale_mean": mean,
        "grayscale_standard_deviation": deviation,
        "rms_contrast": float(np.sqrt(np.mean((gray_float - gray_mean) ** 2)) / 255.0),
        "laplacian_variance": float(laplacian.var(ddof=0)),
        "tenengrad_mean": float(np.mean(sobel_x ** 2 + sobel_y ** 2)),
        "entropy": float(-np.sum(probabilities * np.log2(probabilities))),
        "saturated_dark_fraction": float(np.mean(gray <= 5)),
        "saturated_bright_fraction": float(np.mean(gray >= 250)),
    }


def extract_record(project_root: Path, manifest: dict[str, str], registry: dict[str, str], *, verify_crop: bool) -> dict:
    width, height = int(registry["width"]), int(registry["height"])
    left, top = int(manifest["clipped_left"]), int(manifest["clipped_top"])
    crop_path, source_path = project_root / registry["crop_path"], project_root / manifest["source_image"]
    if not crop_path.is_file() or not source_path.is_file():
        raise ManifestError(f"missing crop or source image for {registry['crop_path']}")
    crop_sha = sha256_file(crop_path)
    if verify_crop and crop_sha != registry["sha256"]:
        raise ManifestError(f"crop checksum mismatch: {registry['crop_path']}")
    try:
        with Image.open(source_path) as source:
            source_width, source_height = source.size
        with Image.open(crop_path) as crop:
            crop.load()
            if crop.size != (width, height):
                raise ManifestError(f"crop dimension mismatch: {registry['crop_path']}")
            gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise ManifestError(f"unreadable image for {registry['crop_path']}") from exc
    if width <= 0 or height <= 0 or left < 0 or top < 0 or left + width > source_width or top + height > source_height:
        raise ManifestError(f"invalid crop geometry: {registry['crop_path']}")
    area = width * height
    if int(manifest["clipped_area"]) != area:
        raise ManifestError(f"crop area mismatch: {registry['crop_path']}")
    right_distance = source_width - (left + width)
    bottom_distance = source_height - (top + height)
    original_clipped = any(
        int(manifest[field]) != value for field, value in (
            ("original_left", left), ("original_top", top),
            ("original_width", width), ("original_height", height),
        )
    )
    row = {
        "split": manifest["split"], "sequence": manifest["sequence"],
        "frame": int(manifest["frame"]), "identity": manifest["identity"],
        "category": int(manifest["category"]),
        "identity_majority_category": int(manifest["identity_majority_category"]),
        "identity_category_conflict": int(manifest["identity_category_conflict"]),
        "crop_path": registry["crop_path"], "source_image": manifest["source_image"],
        "crop_sha256": crop_sha, "width": width, "height": height, "area": area,
        "aspect_ratio": width / height, "source_width": source_width, "source_height": source_height,
        "relative_width": width / source_width, "relative_height": height / source_height,
        "relative_area": area / (source_width * source_height), "clipped_left": left,
        "clipped_top": top, "distance_right": right_distance, "distance_bottom": bottom_distance,
        "minimum_border_distance": min(left, top, right_distance, bottom_distance),
        "touches_left_border": int(left == 0), "touches_top_border": int(top == 0),
        "touches_right_border": int(right_distance == 0), "touches_bottom_border": int(bottom_distance == 0),
        "original_box_was_clipped": int(original_clipped), "occlusion": int(manifest["occlusion"]),
        "out_of_view": int(manifest["out_of_view"]),
    }
    row.update(appearance_features(gray))
    return row


def aggregate_identities(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["identity"])].append(row)
    results = []
    for (split, identity), values in sorted(groups.items()):
        touching = [any(row[field] for field in ("touches_left_border", "touches_top_border", "touches_right_border", "touches_bottom_border")) for row in values]
        code_counts = lambda field: json.dumps({str(k): v for k, v in sorted(Counter(row[field] for row in values).items())}, sort_keys=True, separators=(",", ":"))
        results.append({
            "split": split, "identity": identity, "crop_count": len(values),
            "frame_minimum": min(row["frame"] for row in values), "frame_maximum": max(row["frame"] for row in values),
            "width_median": median(row["width"] for row in values), "height_median": median(row["height"] for row in values),
            "area_median": median(row["area"] for row in values), "relative_area_median": median(row["relative_area"] for row in values),
            "laplacian_variance_median": median(row["laplacian_variance"] for row in values),
            "tenengrad_mean_median": median(row["tenengrad_mean"] for row in values),
            "rms_contrast_median": median(row["rms_contrast"] for row in values), "entropy_median": median(row["entropy"] for row in values),
            "proportion_touching_any_image_border": sum(touching) / len(touching),
            "occlusion_code_counts": code_counts("occlusion"), "out_of_view_code_counts": code_counts("out_of_view"),
        })
    return results


def _percentiles(values: list[float]) -> dict[str, float]:
    return {str(point): float(np.percentile(values, point)) for point in (0, 25, 50, 75, 100)}


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet(project_root: Path, rows: list[dict], destination: Path, metric: str, limit: int = 24) -> None:
    chosen = rows[:limit]
    if not chosen:
        return
    cell_width, cell_height, columns = 260, 210, 4
    sheet = Image.new("RGB", (cell_width * columns, cell_height * math.ceil(len(chosen) / columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        with Image.open(project_root / row["crop_path"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail((cell_width - 10, cell_height - 56))
        x, y = index % columns * cell_width, index // columns * cell_height
        sheet.paste(preview, (x + (cell_width - preview.width) // 2, y + 2))
        label = f"{row['identity']} {row['sequence']} f{row['frame']} {row['width']}x{row['height']}\n{metric}={row[metric]:.6g}"
        draw.multiline_text((x + 4, y + cell_height - 50), label, fill="black", spacing=2)
    sheet.save(destination, format="PNG", compress_level=9)


def analyze(project_root: Path, manifest_path: Path, registry_path: Path, output_dir: Path, *,
            verify_crops: bool = False, max_records: int | None = None, overwrite: bool = False) -> dict:
    if max_records is not None and max_records < 1:
        raise ManifestError("max_records must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ManifestError(f"output directory is not empty (use --overwrite): {output_dir}")
    joined = join_inputs(manifest_path, registry_path)
    dataset_digest = hashlib.sha256()
    for _, registry in sorted(joined, key=lambda pair: pair[1]["crop_path"]):
        dataset_digest.update(registry["crop_path"].encode("utf-8"))
        dataset_digest.update(registry["sha256"].encode("ascii"))
    if max_records is not None:
        joined = joined[:max_records]
    rows = [extract_record(project_root, manifest, registry, verify_crop=verify_crops) for manifest, registry in joined]
    rows.sort(key=lambda row: (row["split"], row["sequence"], row["identity"], row["frame"], row["crop_path"]))
    if not rows:
        raise ManifestError("no quality records selected")
    splits = sorted({row["split"] for row in rows})
    if len(splits) != 1:
        raise ManifestError(f"quality analysis requires exactly one split, found {splits}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_crop.csv", PER_CROP_FIELDS, rows)
    _write_csv(output_dir / "per_identity.csv", IDENTITY_FIELDS, aggregate_identities(rows))
    rank_limit = min(100, len(rows))
    lowest_laplacian = sorted(rows, key=lambda row: (row["laplacian_variance"], row["crop_path"]))[:rank_limit]
    smallest_area = sorted(rows, key=lambda row: (row["relative_area"], row["crop_path"]))[:rank_limit]
    border_risk = sorted(rows, key=lambda row: (row["minimum_border_distance"], row["crop_path"]))[:rank_limit]
    _write_csv(output_dir / "lowest_laplacian_variance.csv", PER_CROP_FIELDS, lowest_laplacian)
    _write_csv(output_dir / "smallest_relative_area.csv", PER_CROP_FIELDS, smallest_area)
    _write_csv(output_dir / "highest_border_risk.csv", PER_CROP_FIELDS, border_risk)
    lowest_contrast = sorted(rows, key=lambda row: (row["rms_contrast"], row["crop_path"]))[:rank_limit]
    _contact_sheet(project_root, lowest_laplacian, output_dir / "lowest_laplacian_variance.png", "laplacian_variance")
    _contact_sheet(project_root, smallest_area, output_dir / "smallest_relative_area.png", "relative_area")
    _contact_sheet(project_root, lowest_contrast, output_dir / "lowest_contrast.png", "rms_contrast")
    touching = [row for row in border_risk if row["minimum_border_distance"] == 0]
    _contact_sheet(project_root, touching, output_dir / "border_touching_examples.png", "minimum_border_distance")
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, check=True, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(), "git_commit_sha": git_sha,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
        },
        "crop_dataset_checksum": dataset_digest.hexdigest(), "record_count": len(rows),
        "identity_count": len({row["identity"] for row in rows}),
        "sequence_list": sorted({row["sequence"] for row in rows}), "split": splits[0],
        "numeric_quality_percentiles": {field: _percentiles([row[field] for row in rows]) for field in NUMERIC_QUALITY_FIELDS},
        "invalid_or_rejected_row_count": 0, "frozen_test_record_count": 0,
        "verify_crops": verify_crops, "max_records": max_records,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
