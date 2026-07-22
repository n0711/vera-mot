"""Safe, lossless UAVDT ReID crop materialization and verification."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from vera_mot.reid_data import ManifestError, percentile, sha256_file


REGISTRY_FIELDS = [
    "split", "sequence", "frame", "identity", "category",
    "identity_majority_category", "identity_category_conflict", "crop_path",
    "width", "height", "file_size", "sha256", "source_image",
    "source_region_sha256",
]


def safe_relative_path(value: str, *, required_prefix: str | None = None) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or any(part in ("", ".", "..") for part in path.parts):
        raise ManifestError(f"unsafe relative path: {value!r}")
    if required_prefix and not path.as_posix().startswith(required_prefix.rstrip("/") + "/"):
        raise ManifestError(f"path is outside required prefix {required_prefix}: {value}")
    return path


def region_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(f"{image.width}x{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def read_manifest(path: Path, expected_split: str, frozen_test_sequences: set[str]) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, 2):
            if row["split"] not in ("train", "val") or row["split"] != expected_split:
                raise ManifestError(f"invalid split on {path}:{line_number}: {row['split']}")
            if row["sequence"] in frozen_test_sequences:
                raise ManifestError(f"frozen test sequence in manifest: {row['sequence']}")
            source = safe_relative_path(row["source_image"], required_prefix="datasets/UAVDT/full_mot/raw/UAV-benchmark-M")
            crop = safe_relative_path(row["planned_crop_path"], required_prefix="datasets/UAVDT/reid_crops")
            if crop.suffix != ".png":
                raise ManifestError(f"planned crop is not PNG: {crop}")
            expected_crop_prefix = Path("datasets/UAVDT/reid_crops") / expected_split
            if not crop.is_relative_to(expected_crop_prefix):
                raise ManifestError(f"crop path does not match split: {crop}")
            if source.parts[-2] != row["sequence"]:
                raise ManifestError(f"source path sequence mismatch: {source}")
            parsed = dict(row)
            for key in ("frame", "target_id", "category", "identity_majority_category",
                        "identity_category_conflict", "clipped_left", "clipped_top",
                        "clipped_width", "clipped_height", "clipped_area", "sampling_index"):
                parsed[key] = int(row[key])
            parsed["source_image"] = source.as_posix()
            parsed["planned_crop_path"] = crop.as_posix()
            rows.append(parsed)
    keys = [(row["sequence"], row["frame"], row["identity"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ManifestError(f"duplicate records in {path}")
    return sorted(rows, key=lambda row: (row["sequence"], row["identity"], row["frame"]))


def validate_source_region(project_root: Path, row: dict) -> tuple[Image.Image, str]:
    source_path = project_root / row["source_image"]
    if not source_path.is_file():
        raise ManifestError(f"missing source image: {row['source_image']}")
    try:
        with Image.open(source_path) as source:
            source.load()
            left, top = row["clipped_left"], row["clipped_top"]
            width, height = row["clipped_width"], row["clipped_height"]
            if width <= 0 or height <= 0 or left < 0 or top < 0:
                raise ManifestError(f"invalid crop coordinates: {row['planned_crop_path']}")
            if left + width > source.width or top + height > source.height:
                raise ManifestError(f"crop exceeds source dimensions: {row['planned_crop_path']}")
            if width * height != row["clipped_area"]:
                raise ManifestError(f"crop area mismatch: {row['planned_crop_path']}")
            region = source.crop((left, top, left + width, top + height))
            region.load()
    except OSError as exc:
        raise ManifestError(f"unreadable source image: {row['source_image']}") from exc
    return region, region_sha256(region)


def verify_crop(path: Path, expected_region: Image.Image, expected_region_sha: str) -> tuple[bool, str]:
    try:
        with Image.open(path) as crop:
            crop.load()
            valid = (
                crop.format == "PNG"
                and crop.size == expected_region.size
                and crop.mode == expected_region.mode
                and region_sha256(crop) == expected_region_sha
                and crop.tobytes() == expected_region.tobytes()
            )
    except (OSError, ValueError):
        return False, "unreadable or invalid PNG"
    return valid, "verified" if valid else "dimensions, mode, or pixels differ"


def write_png_atomic(path: Path, region: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        region.save(temporary, format="PNG", optimize=False, compress_level=9)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def registry_checksum(path: Path) -> str:
    return sha256_file(path)


def dataset_checksum(registry_rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(registry_rows, key=lambda item: item["crop_path"]):
        digest.update(row["crop_path"].encode("utf-8"))
        digest.update(row["sha256"].encode("ascii"))
    return digest.hexdigest()


def write_registry(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["crop_path"]))
    return registry_checksum(path)


def preflight(project_root: Path, rows_by_split: dict[str, list[dict]], safe_overhead_ratio: float = 1.25) -> dict:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    expected_bytes = 0
    seen_outputs = set()
    for row in all_rows:
        if row["planned_crop_path"] in seen_outputs:
            raise ManifestError(f"duplicate crop output path: {row['planned_crop_path']}")
        seen_outputs.add(row["planned_crop_path"])
        region, _ = validate_source_region(project_root, row)
        expected_bytes += region.width * region.height * len(region.getbands())
    output_root = project_root / "datasets/UAVDT/reid_crops"
    output_root.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(output_root).free
    required = int(expected_bytes * safe_overhead_ratio) + 64 * 1024 * 1024
    if available < required:
        raise ManifestError(f"insufficient disk: need {required} bytes including overhead, have {available}")
    return {
        "expected_crop_count": len(all_rows), "raw_pixel_bytes": expected_bytes,
        "required_bytes_with_safe_overhead": required, "available_bytes": available,
    }


def materialize_split(project_root: Path, rows: list[dict], *, force: bool = False) -> tuple[list[dict], dict]:
    registry_rows = []
    resumed = written = 0
    for row in rows:
        region, source_region_sha = validate_source_region(project_root, row)
        crop_path = project_root / row["planned_crop_path"]
        if crop_path.exists():
            valid, reason = verify_crop(crop_path, region, source_region_sha)
            if valid:
                resumed += 1
            elif not force:
                raise ManifestError(f"existing crop mismatch ({reason}): {row['planned_crop_path']}")
            else:
                write_png_atomic(crop_path, region)
                written += 1
        else:
            write_png_atomic(crop_path, region)
            written += 1
        valid, reason = verify_crop(crop_path, region, source_region_sha)
        if not valid:
            raise ManifestError(f"written crop failed verification ({reason}): {row['planned_crop_path']}")
        registry_rows.append({
            "split": row["split"], "sequence": row["sequence"], "frame": row["frame"],
            "identity": row["identity"], "category": row["category"],
            "identity_majority_category": row["identity_majority_category"],
            "identity_category_conflict": row["identity_category_conflict"],
            "crop_path": row["planned_crop_path"], "width": region.width,
            "height": region.height, "file_size": crop_path.stat().st_size,
            "sha256": sha256_file(crop_path), "source_image": row["source_image"],
            "source_region_sha256": source_region_sha,
        })
    return registry_rows, {"written_crops": written, "resumed_crops": resumed, "failed_crops": 0}


def make_contact_sheet(project_root: Path, rows: list[dict], path: Path, *, limit: int = 24) -> None:
    chosen = rows[:limit]
    if not chosen:
        return
    cell_width, cell_height = 220, 190
    sheet = Image.new("RGB", (cell_width * 4, cell_height * ((len(chosen) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        with Image.open(project_root / row["planned_crop_path"]) as crop:
            preview = crop.convert("RGB")
            preview.thumbnail((cell_width - 10, cell_height - 45))
            x = (index % 4) * cell_width
            y = (index // 4) * cell_height
            sheet.paste(preview, (x + (cell_width - preview.width) // 2, y + 2))
        label = f"{row['sequence']} {row['identity']} f{row['frame']} c{row['category']} {row['clipped_width']}x{row['clipped_height']}"
        draw.text((x + 4, y + cell_height - 39), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", compress_level=9)


def create_qa_sheets(project_root: Path, rows_by_split: dict[str, list[dict]], qa_root: Path) -> list[str]:
    train, val = rows_by_split["train"], rows_by_split["val"]
    definitions = {
        "representative_train.png": train[::max(1, len(train) // 24)],
        "representative_val.png": val[::max(1, len(val) // 24)],
        "smallest_accepted.png": sorted(train + val, key=lambda row: (row["clipped_area"], row["planned_crop_path"])),
        "car_examples.png": [row for row in train + val if row["category"] == 1],
        "truck_examples.png": [row for row in train + val if row["category"] == 2],
        "bus_examples.png": [row for row in train + val if row["category"] == 3],
    }
    raw_code_examples = []
    for key in ("occlusion", "out_of_view"):
        for code in sorted({row[key] for row in train + val}):
            candidates = [row for row in train + val if row[key] == code]
            step = max(1, len(candidates) // 3)
            raw_code_examples.extend(candidates[::step][:3])
    definitions["occlusion_out_of_view_codes.png"] = raw_code_examples
    outputs = []
    for name, candidates in definitions.items():
        # Spread category sheets temporally instead of taking only the earliest identity.
        step = max(1, len(candidates) // 24)
        selected = candidates[::step][:24]
        destination = qa_root / name
        make_contact_sheet(project_root, selected, destination)
        outputs.append(destination.relative_to(project_root).as_posix())
    return outputs


def summarize_registry(rows: list[dict]) -> dict:
    widths = [int(row["width"]) for row in rows]
    heights = [int(row["height"]) for row in rows]
    areas = [width * height for width, height in zip(widths, heights)]
    points = (0, 25, 50, 75, 100)
    return {
        "crop_count": len(rows), "total_bytes": sum(int(row["file_size"]) for row in rows),
        "identities": len({row["identity"] for row in rows}),
        "crops_per_category": {str(category): sum(int(row["category"]) == category for row in rows) for category in (1, 2, 3)},
        "identities_per_majority_category": {str(category): len({row["identity"] for row in rows if int(row["identity_majority_category"]) == category}) for category in (1, 2, 3)},
        "category_conflict_identities": len({row["identity"] for row in rows if int(row["identity_category_conflict"])}),
        "dimension_percentiles": {
            "width": {str(point): percentile(widths, point) for point in points},
            "height": {str(point): percentile(heights, point) for point in points},
            "area": {str(point): percentile(areas, point) for point in points},
        },
    }
