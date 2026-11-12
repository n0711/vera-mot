"""Fail-closed conversion of original 9-column UAVDT validation GT for TrackEval.

This is a format adapter only. It does not run the evaluator or claim tracker
accuracy. Source attributes are validated and preserved in typed records and
aggregate provenance, NOT relabelled as MOTChallenge confidence/visibility.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vera_mot.reid_ablation_audit import AuditError
from vera_mot.reid_eval_preflight import _read_json, _validate_sequence, file_sha256

SCHEMA = "uavdt-nine-column-to-trackeval-single-vehicle-v1"
# Explicit source-code domains. These are source IDs, not MOTChallenge class IDs.
OUT_OF_VIEW_CODES = frozenset({1, 2, 3})
OCCLUSION_CODES = frozenset({1, 2, 3, 4})
VEHICLE_CATEGORY_CODES = frozenset({1, 2, 3})
SOURCE_SUFFIX = "_gt_whole.txt"


@dataclass(frozen=True)
class UAVDTRecord:
    frame: int
    identity: int
    left: int
    top: int
    width: int
    height: int
    out_of_view: int
    occlusion: int
    category: int

    def source_values(self) -> tuple[int, ...]:
        return (
            self.frame, self.identity, self.left, self.top, self.width,
            self.height, self.out_of_view, self.occlusion, self.category,
        )

    def converted_values(self) -> tuple[int, ...]:
        # Pinned MotChallenge2DBox accepts one class called 'pedestrian'.
        # Here class ID 1 is an EXPLICIT compatibility alias for ALL vehicles.
        # Mark=1 keeps each source observation; -1,-1 are unused placeholders.
        return (
            self.frame, self.identity, self.left, self.top,
            self.width, self.height, 1, 1, -1, -1,
        )


def _integer(token: str, label: str) -> int:
    if re.fullmatch(r"[+-]?[0-9]+", token) is None:
        raise AuditError(f"{label}: expected a decimal integer")
    return int(token)


def parse_source(text: str, *, frame_count: int) -> tuple[UAVDTRecord, ...]:
    if type(frame_count) is not int or frame_count < 1:
        raise AuditError("frame_count must be a positive integer")
    seen: set[tuple[int, int]] = set()
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        tokens = [token.strip() for token in line.split(",")]
        if len(tokens) != 9:
            raise AuditError(f"source line {lineno}: expected 9 UAVDT fields")
        fields = [_integer(token, f"source line {lineno}") for token in tokens]
        row = UAVDTRecord(*fields)
        if row.frame < 1 or row.frame > frame_count or row.identity < 1:
            raise AuditError(f"source line {lineno}: frame or identity out of range")
        if row.width <= 0 or row.height <= 0:
            raise AuditError(f"source line {lineno}: nonpositive box dimensions")
        for value, allowed, label in (
            (row.out_of_view, OUT_OF_VIEW_CODES, "out_of_view"),
            (row.occlusion, OCCLUSION_CODES, "occlusion"),
            (row.category, VEHICLE_CATEGORY_CODES, "category"),
        ):
            if value not in allowed:
                raise AuditError(f"source line {lineno}: unsupported {label} code {value}")
        key = (row.frame, row.identity)
        if key in seen:
            raise AuditError(f"source line {lineno}: duplicate sequence-local identity")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise AuditError("original UAVDT ground truth is empty")
    return tuple(sorted(rows, key=lambda row: (row.frame, row.identity)))


def _render(records: tuple[UAVDTRecord, ...], *, source: bool) -> str:
    values = (row.source_values() if source else row.converted_values() for row in records)
    return "".join(",".join(map(str, items)) + "\n" for items in values)


def _verify_generated(records: tuple[UAVDTRecord, ...], output: str) -> None:
    lines = output.splitlines()
    if len(lines) != len(records):
        raise AuditError("converted GT row count mismatch")
    for number, (row, line) in enumerate(zip(records, lines), start=1):
        tokens = line.split(",")
        if len(tokens) != 10:
            raise AuditError(f"converted line {number}: expected 10 MOT fields")
        values = tuple(_integer(token, f"converted line {number}") for token in tokens)
        if values != row.converted_values():
            raise AuditError(f"converted line {number}: identity, box or class mismatch")


def convert_text(text: str, *, frame_count: int) -> tuple[str, dict]:
    """Return deterministic 10-col GT and summary after lossless geometry checks."""
    records = parse_source(text, frame_count=frame_count)
    converted = _render(records, source=False)
    _verify_generated(records, converted)
    # A typed 9-col canonical reserialization preserves all nine source fields.
    # The 10-col TrackEval representation intentionally discards the raw codes.
    source_canonical = _render(records, source=True)
    if parse_source(source_canonical, frame_count=frame_count) != records:
        raise AuditError("canonical nine-column roundtrip mismatch")
    return converted, {
        "record_count": len(records),
        "annotated_frames": len({row.frame for row in records}),
        "sequence_local_identities": len({row.identity for row in records}),
        "raw_code_histograms": {
            label: {str(code): count for code, count in sorted(counts.items())}
            for label, counts in (
                ("out_of_view", Counter(row.out_of_view for row in records)),
                ("occlusion", Counter(row.occlusion for row in records)),
                ("category", Counter(row.category for row in records)),
            )
        },
        "canonical_source_sha256": hashlib.sha256(source_canonical.encode("utf-8")).hexdigest(),
        "converted_sha256": hashlib.sha256(converted.encode("utf-8")).hexdigest(),
    }


def prepare_conversion(project_root: Path, *, sequence: str, source: Path) -> tuple[str, dict]:
    """Enforce frozen validation split and source-inventory size before output."""
    root = Path(project_root)
    config = root / "configs/uavdt_reid_split.yaml"
    audit = root / "docs/uavdt_mot_audit.json"
    record = _validate_sequence(sequence, _read_json(config, "frozen split config"),
                                _read_json(audit, "original sequence inventory"))
    source = Path(source)
    if source.name != sequence + SOURCE_SUFFIX:
        raise AuditError("input must be the original *_gt_whole.txt for this sequence")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AuditError("original source annotation cannot be read") from exc
    converted, stats = convert_text(text, frame_count=record["frames"])
    if stats["record_count"] != record["gt_rows"]:
        raise AuditError("source row count differs from committed UAVDT audit")
    report = {
        "schema_version": SCHEMA,
        "status": "format_conversion_only_not_identity_evaluation",
        "sequence": sequence,
        "source_type": "original_nine_column_gt_whole",
        "output_type": "pinned_trackeval_motchallenge_single_vehicle_alias",
        "source_sha256": file_sha256(source),
        "split_configuration_sha256": file_sha256(config),
        "source_inventory_sha256": file_sha256(audit),
        **stats,
        "trackeval_pinned_commit": "12c8791b303e0a0b50f753af204249e622d0281a",
        "policy": {
            "input_fields": [
                "frame", "target_id", "left", "top", "width", "height",
                "out_of_view", "occlusion", "category",
            ],
            "output_fields": [
                "frame", "id", "left", "top", "width", "height",
                "gt_mark", "gt_class", "unused_1", "unused_2",
            ],
            "single_vehicle_gt_mark": 1,
            "single_vehicle_gt_class": 1,
            "gt_class_1_is_trackeval_pedestrian_compatibility_alias": True,
            "raw_source_attribute_codes_not_used_for_metric_filtering": True,
            "ignore_region_policy": "not_applied_requires_separate_validation",
        },
        "limitations": [
            "Single-vehicle class alias is a file compatibility choice, not a pedestrian annotation.",
            "Raw nine-column source attributes are not losslessly reconstructable from ten-column GT alone.",
            "No ignore-area preprocessing, GT-to-video synchronization or identity evaluation performed.",
            "Synthetic tests cannot establish correctness against authorized original UAVDT archives.",
        ],
    }
    return converted, report


def write_conversion(output_dir: Path, *, sequence: str, converted: str, report: dict) -> tuple[Path, Path]:
    """Write only fresh files; never overwrite an existing source or result."""
    out_dir = Path(output_dir)
    if not out_dir.is_dir():
        raise AuditError("output directory must already exist")
    gt = out_dir / f"{sequence}_gt.txt"
    meta = out_dir / f"{sequence}_gt.conversion.json"
    if gt.exists() or meta.exists():
        raise AuditError("output GT or conversion report already exists")
    # Prevalidate the payload before touching any output.
    if hashlib.sha256(converted.encode("utf-8")).hexdigest() != report["converted_sha256"]:
        raise AuditError("output hash does not match conversion report")
    created: list[Path] = []
    try:
        with gt.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(gt)
            handle.write(converted)
        with meta.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(meta)
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return gt, meta
