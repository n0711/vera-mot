"""Leakage-safe, manifest-only UAVDT ReID data utilities."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


FIELDS = [
    "sequence", "frame", "identity", "target_id", "category",
    "identity_majority_category", "identity_category_conflict",
    "out_of_view", "occlusion", "original_left", "original_top",
    "original_width", "original_height", "clipped_left", "clipped_top",
    "clipped_width", "clipped_height", "clipped_area", "source_image",
    "planned_crop_path", "split", "sampling_index",
]
SUPPORTED_CATEGORIES = (1, 2, 3)


class ManifestError(ValueError):
    """Raised when an input or split would violate the data contract."""


def parse_gt_line(line: str) -> dict[str, int]:
    """Parse one UAVDT ``*_gt_whole.txt`` row without interpreting raw codes."""
    try:
        values = [int(value.strip()) for value in line.split(",")]
    except ValueError as exc:
        raise ManifestError(f"non-integer UAVDT GT row: {line!r}") from exc
    if len(values) != 9:
        raise ManifestError("UAVDT GT row must contain exactly 9 columns")
    names = ("frame", "target_id", "left", "top", "width", "height",
             "out_of_view", "occlusion", "category")
    return dict(zip(names, values))


def composite_identity(sequence: str, target_id: int) -> str:
    return f"{sequence}_{target_id}"


def clip_box(row: dict[str, int], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    left = max(0, row["left"])
    top = max(0, row["top"])
    right = min(image_width, row["left"] + row["width"])
    bottom = min(image_height, row["top"] + row["height"])
    return left, top, right - left, bottom - top


def uniform_indices(length: int, limit: int) -> list[int]:
    """Return exactly ``limit`` endpoint-preserving, uniformly spaced indices."""
    if length < 0 or limit < 1:
        raise ManifestError("length must be non-negative and limit positive")
    if length <= limit:
        return list(range(length))
    # Integer half-up rounding avoids platform-dependent floating point ties.
    return [(index * (length - 1) + (limit - 1) // 2) // (limit - 1)
            for index in range(limit)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    result = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(result, 2)


def validate_split_config(config: dict) -> None:
    train = config.get("train_sequences", [])
    val = config.get("validation_sequences", [])
    test = config.get("frozen_test_sequences", [])
    if len(train) != 24 or len(val) != 6 or len(test) != 20:
        raise ManifestError("split must contain exactly 24 train, 6 validation and 20 test sequences")
    groups = {"train": train, "validation": val, "test": test}
    for name, sequences in groups.items():
        if len(sequences) != len(set(sequences)):
            raise ManifestError(f"duplicate sequence within {name} split")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(groups[left]) & set(groups[right])
        if overlap:
            raise ManifestError(f"sequence leakage between {left} and {right}: {sorted(overlap)}")
    if config.get("seed") != 42:
        raise ManifestError("manifest seed must be 42")


def validate_manifest_leakage(train_rows: list[dict], val_rows: list[dict], test_sequences: list[str]) -> None:
    """Reject sequence, identity, test, and record-key leakage in candidate rows."""
    train_sequences = {row["sequence"] for row in train_rows}
    val_sequences = {row["sequence"] for row in val_rows}
    if train_sequences & val_sequences:
        raise ManifestError("sequence occurs in both train and validation manifests")
    if (train_sequences | val_sequences) & set(test_sequences):
        raise ManifestError("frozen test sequence entered a development manifest")
    if {row["identity"] for row in train_rows} & {row["identity"] for row in val_rows}:
        raise ManifestError("identity occurs in both train and validation manifests")
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        keys = [(row["sequence"], row["frame"], row["identity"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ManifestError(f"duplicate manifest record in {split}")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ManifestError(f"path is outside project root: {path}") from exc


def _distribution(rows: list[dict], key: str) -> dict[str, int]:
    return {str(code): count for code, count in sorted(Counter(row[key] for row in rows).items())}


def _percentiles(values: list[int]) -> dict[str, float | None]:
    return {str(point): percentile(values, point) for point in (0, 25, 50, 75, 100)}


def build_split_rows(
    project_root: Path,
    sequences: list[str],
    split: str,
    gt_root: Path,
    image_root: Path,
    *,
    minimum_observations: int = 10,
    temporal_stride: int = 5,
    maximum_samples: int = 200,
    minimum_final_samples: int = 4,
) -> tuple[list[dict], dict]:
    """Build rows and compact evidence for one sequence-disjoint split."""
    identities: dict[str, list[dict]] = defaultdict(list)
    raw_identities: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    duplicate_observations: set[tuple[str, int, str]] = set()
    seen_observations: set[tuple[str, int, str]] = set()
    missing_images: list[str] = []
    image_dimensions: dict[Path, tuple[int, int]] = {}

    for sequence in sorted(sequences):
        gt_path = gt_root / f"{sequence}_gt_whole.txt"
        if not gt_path.is_file():
            raise ManifestError(f"required GT is missing: {_relative(project_root, gt_path)}")
        for line_number, line in enumerate(gt_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            counts["gt_rows"] += 1
            row = parse_gt_line(line)
            identity = composite_identity(sequence, row["target_id"])
            raw_identities[identity].append(row)
            key = (sequence, row["frame"], identity)
            if key in seen_observations:
                duplicate_observations.add(key)
            seen_observations.add(key)
            if row["category"] not in SUPPORTED_CATEGORIES:
                counts["unsupported_category"] += 1
                continue
            counts["supported_category"] += 1
            image_path = image_root / sequence / f"img{row['frame']:06d}.jpg"
            if not image_path.is_file():
                missing_images.append(_relative(project_root, image_path))
                continue
            if image_path not in image_dimensions:
                try:
                    with Image.open(image_path) as image:
                        image_dimensions[image_path] = image.size
                except (OSError, ValueError) as exc:
                    raise ManifestError(f"required image is unreadable: {_relative(project_root, image_path)}") from exc
            width, height = image_dimensions[image_path]
            left, top, clipped_width, clipped_height = clip_box(row, width, height)
            if clipped_width <= 0 or clipped_height <= 0:
                counts["invalid_clipped_box"] += 1
                continue
            if clipped_width < 10:
                counts["clipped_width_lt_10"] += 1
                continue
            if clipped_height < 10:
                counts["clipped_height_lt_10"] += 1
                continue
            area = clipped_width * clipped_height
            if area < 100:
                counts["clipped_area_lt_100"] += 1
                continue
            counts["valid_observations"] += 1
            identities[identity].append({
                "sequence": sequence, "frame": row["frame"], "identity": identity,
                "target_id": row["target_id"], "category": row["category"],
                "out_of_view": row["out_of_view"], "occlusion": row["occlusion"],
                "original_left": row["left"], "original_top": row["top"],
                "original_width": row["width"], "original_height": row["height"],
                "clipped_left": left, "clipped_top": top,
                "clipped_width": clipped_width, "clipped_height": clipped_height,
                "clipped_area": area, "source_image": _relative(project_root, image_path),
                "planned_crop_path": f"datasets/UAVDT/reid_crops/{split}/{identity}/{sequence}_{row['frame']}.png",
                "split": split, "sampling_index": 0,
            })

    if duplicate_observations:
        raise ManifestError(f"duplicate (sequence, frame, identity) observations: {len(duplicate_observations)}")
    if missing_images:
        preview = ", ".join(sorted(set(missing_images))[:3])
        raise ManifestError(f"required image frames are missing ({len(missing_images)} observations): {preview}")

    selected: list[dict] = []
    rejected_identities: dict[str, str] = {}
    for identity in sorted(raw_identities):
        valid_count = len(identities.get(identity, []))
        if valid_count == 0:
            rejected_identities[identity] = "no_valid_observations_after_filtering"
        elif valid_count < minimum_observations:
            rejected_identities[identity] = "fewer_than_10_valid_source_observations"
    eligible_source = set(raw_identities) - set(rejected_identities)
    category_conflicts = {}
    for identity in sorted(eligible_source):
        observations = sorted(identities[identity], key=lambda row: row["frame"])
        sampled = observations[::temporal_stride]
        sampled = [sampled[index] for index in uniform_indices(len(sampled), maximum_samples)]
        if len(sampled) < minimum_final_samples:
            rejected_identities[identity] = "fewer_than_4_final_planned_crops"
            continue
        category_counts = Counter(row["category"] for row in observations)
        majority = min(category_counts, key=lambda category: (-category_counts[category], category))
        conflict = len(category_counts) > 1
        ordered_categories = [row["category"] for row in observations]
        transitions = sum(left != right for left, right in zip(ordered_categories, ordered_categories[1:]))
        frame_categories: dict[int, set[int]] = defaultdict(set)
        for row in observations:
            frame_categories[row["frame"]].add(row["category"])
        category_conflicts[identity] = {
            "categories": sorted(category_counts),
            "majority_category": majority,
            "transitions": transitions,
            "same_frame_conflicts": sum(len(categories) > 1 for categories in frame_categories.values()),
        }
        for sampling_index, row in enumerate(sampled):
            row["identity_majority_category"] = majority
            row["identity_category_conflict"] = int(conflict)
            row["sampling_index"] = sampling_index
            selected.append(row)
    selected.sort(key=lambda row: (row["sequence"], row["identity"], row["frame"]))

    manifest_keys = [(row["sequence"], row["frame"], row["identity"]) for row in selected]
    if len(manifest_keys) != len(set(manifest_keys)):
        raise ManifestError("duplicate manifest records after sampling")
    if any(row["category"] not in SUPPORTED_CATEGORIES for row in selected):
        raise ManifestError("unsupported category entered manifest")
    retained_counts = Counter(row["identity"] for row in selected)
    if any(len(identities[identity]) < minimum_observations for identity in retained_counts):
        raise ManifestError("identity with insufficient valid observations entered manifest")
    if any(count < minimum_final_samples for count in retained_counts.values()):
        raise ManifestError("identity with insufficient final planned crops entered manifest")

    rejection_counts = {
        reason: counts[reason] for reason in (
            "unsupported_category", "invalid_clipped_box", "clipped_width_lt_10",
            "clipped_height_lt_10", "clipped_area_lt_100")
    }
    rejection_counts["rejected_identity_observations"] = sum(
        len(identities.get(identity, [])) for identity in rejected_identities
    )
    rejection_counts["rejected_identities"] = len(rejected_identities)
    reason_counts = Counter(rejected_identities.values())
    conflict_rows = [value for identity, value in category_conflicts.items() if identity in retained_counts and len(value["categories"]) > 1]
    combination_counts = Counter("+".join(map(str, value["categories"])) for value in conflict_rows)
    stats = {
        "identity_accounting": {
            "raw_identities": len(raw_identities),
            "accepted_identities": len(retained_counts),
            "rejected_identities": len(rejected_identities),
            "equation_holds": len(raw_identities) == len(retained_counts) + len(rejected_identities),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
            "rejected_identity_table": [
                {"identity": identity, "reason": rejected_identities[identity],
                 "raw_observations": len(raw_identities[identity]),
                 "valid_observations": len(identities.get(identity, [])),
                 "final_planned_crops": len(identities.get(identity, [])[::temporal_stride])}
                for identity in sorted(rejected_identities)
            ],
        },
        "observations": {
            "gt_rows": counts["gt_rows"],
            "supported_categories": counts["supported_category"],
            "after_box_filtering": counts["valid_observations"],
            "after_identity_filtering": sum(len(rows) for identity, rows in identities.items() if identity not in rejected_identities),
            "planned_crops": len(selected),
        },
        "rejection_counts": rejection_counts,
        "identities": len(retained_counts),
        "identities_per_category": {
            str(category): len({row["identity"] for row in selected if row["category"] == category})
            for category in SUPPORTED_CATEGORIES
        },
        "planned_crops_per_category": {
            str(category): sum(row["category"] == category for row in selected)
            for category in SUPPORTED_CATEGORIES
        },
        "occlusion_distribution": _distribution(selected, "occlusion"),
        "out_of_view_distribution": _distribution(selected, "out_of_view"),
        "box_size_percentiles": {
            "width": _percentiles([row["clipped_width"] for row in selected]),
            "height": _percentiles([row["clipped_height"] for row in selected]),
            "area": _percentiles([row["clipped_area"] for row in selected]),
        },
        "samples_per_identity_percentiles": _percentiles(list(retained_counts.values())),
        "category_conflicts": {
            "identities_with_multiple_raw_categories": len(conflict_rows),
            "percentage_of_accepted_identities": round(100 * len(conflict_rows) / len(retained_counts), 4) if retained_counts else 0,
            "category_combinations": dict(sorted(combination_counts.items())),
            "category_transitions": sum(value["transitions"] for value in conflict_rows),
            "same_frame_category_conflicts": sum(value["same_frame_conflicts"] for value in conflict_rows),
            "majority_category_distribution": {
                str(category): sum(value["majority_category"] == category for identity, value in category_conflicts.items() if identity in retained_counts)
                for category in SUPPORTED_CATEGORIES
            },
        },
        "missing_images": [],
        "duplicate_observations": 0,
    }
    return selected, stats


def write_csv(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _dirty_state_note(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=project_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        return "Git state unavailable; generated from working-tree code."
    return "Generated from working-tree code; repository was dirty." if result.stdout else "Generated from clean committed code."


def build_manifests(
    project_root: Path,
    config_path: Path,
    gt_root: Path,
    image_root: Path,
    output_root: Path,
    summary_path: Path,
) -> dict:
    """Build train/val CSVs and deterministic compact evidence."""
    project_root = project_root.resolve()
    config = json.loads(config_path.read_text())
    validate_split_config(config)
    rows_by_split = {}
    stats_by_split = {}
    for split, key in (("train", "train_sequences"), ("val", "validation_sequences")):
        rows_by_split[split], stats_by_split[split] = build_split_rows(
            project_root, config[key], split, gt_root, image_root,
            minimum_observations=config["manifest_policy"]["minimum_valid_observations_per_identity"],
            temporal_stride=config["manifest_policy"]["temporal_stride"],
            maximum_samples=config["manifest_policy"]["maximum_samples_per_identity"],
            minimum_final_samples=config["manifest_policy"]["minimum_final_planned_crops_per_identity"],
        )

    validate_manifest_leakage(
        rows_by_split["train"], rows_by_split["val"], config["frozen_test_sequences"]
    )
    manifest_checksums = {
        split: write_csv(output_root / f"{split}.csv", rows)
        for split, rows in rows_by_split.items()
    }
    summary = {
        "schema_version": 1,
        "configuration": config["manifest_policy"],
        "checksums": {
            "split_configuration_sha256": sha256_file(config_path),
            "train_manifest_sha256": manifest_checksums["train"],
            "val_manifest_sha256": manifest_checksums["val"],
        },
        "split_lists": {
            "train": config["train_sequences"], "val": config["validation_sequences"],
            "frozen_test": config["frozen_test_sequences"],
        },
        "splits": stats_by_split,
        "leakage_checks": {
            "test_sequence_overlap": 0, "train_val_sequence_overlap": 0,
            "train_val_identity_overlap": 0, "duplicate_manifest_records": 0,
            "unsupported_manifest_categories": 0,
            "insufficient_manifest_identities": 0,
        },
        "missing_images": [],
        "duplicates": 0,
        "code_and_dirty_state_note": _dirty_state_note(project_root),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
