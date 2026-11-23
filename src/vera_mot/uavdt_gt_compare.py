"""Read-only structural comparison of converted UAVDT GT and a reference export."""
from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from pathlib import Path

from vera_mot.reid_ablation_audit import AuditError
from vera_mot.uavdt_gt_convert import prepare_conversion


def _rows(text: str) -> dict[tuple[int, int], tuple[Decimal, ...]]:
    rows = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        tokens = line.split(",")
        if len(tokens) != 10:
            raise AuditError(f"reference line {number}: expected 10 MOT fields")
        try:
            values = tuple(Decimal(token.strip()) for token in tokens)
        except InvalidOperation as exc:
            raise AuditError(f"reference line {number}: invalid number") from exc
        if not all(value.is_finite() for value in values):
            raise AuditError(f"reference line {number}: nonfinite value")
        if any(value < 1 or value != value.to_integral_value() for value in values[:2]):
            raise AuditError(f"reference line {number}: invalid frame or identity")
        if values[4] <= 0 or values[5] <= 0:
            raise AuditError(f"reference line {number}: nonpositive box dimensions")
        # Restrict this audit to the converter's explicit single-vehicle alias.
        if values[6:8] != (Decimal(1), Decimal(1)):
            raise AuditError(f"reference line {number}: expected mark 1 and class 1")
        key = (int(values[0]), int(values[1]))
        if key in rows:
            raise AuditError(f"reference line {number}: duplicate frame and identity")
        rows[key] = values[2:6]
    return rows


def compare_converted_text(converted: str, reference: str) -> dict:
    """Compare exact numeric geometry by frame/identity, independent of row order.

    Empty reference exports are reported as missing all source observations.
    Trailing numeric metadata is validated but excluded from the comparison.
    """
    expected = _rows(converted)
    actual = _rows(reference)
    common = expected.keys() & actual.keys()
    changed = sum(expected[key] != actual[key] for key in common)
    missing = len(expected.keys() - actual.keys())
    extra = len(actual.keys() - expected.keys())
    return {
        "structural_match": missing == extra == changed == 0,
        "source_rows": len(expected),
        "reference_rows": len(actual),
        "matching_rows": len(common) - changed,
        "missing_reference_rows": missing,
        "extra_reference_rows": extra,
        "changed_geometry_rows": changed,
    }


def compare_files(project_root: Path, *, sequence: str, source: Path,
                  reference: Path, ignore_regions: Path | None = None) -> dict:
    """Enforce existing validation-split/inventory gates and fingerprint inputs."""
    converted, provenance = prepare_conversion(project_root, sequence=sequence, source=source)
    if reference.name != f"{sequence}_gt.txt":
        raise AuditError("reference filename must be <sequence>_gt.txt")
    reference_bytes = reference.read_bytes()
    comparison = compare_converted_text(converted, reference_bytes.decode("utf-8"))
    ignore_sha = None
    if ignore_regions is not None:
        if ignore_regions.name != f"{sequence}_gt_ignore.txt":
            raise AuditError("ignore filename must be <sequence>_gt_ignore.txt")
        ignore_sha = hashlib.sha256(ignore_regions.read_bytes()).hexdigest()
    return {
        "schema_version": "uavdt-gt-structural-comparison-v1",
        "sequence": sequence,
        "status": "structural_comparison_only",
        **comparison,
        "conversion_provenance": provenance,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "ignore_regions_sha256": ignore_sha,
        "policy": {
            "comparison_key": ["frame", "identity"],
            "geometry": "exact_numeric_xywh",
            "reference_mark": 1,
            "reference_class": 1,
            "trailing_fields": "finite_numeric_metadata_excluded_from_comparison",
            "ignore_regions": "fingerprinted_only_not_applied",
        },
        "limitations": [
            "A structural match does not establish reference authenticity or source semantics.",
            "Ignore-region semantics and retention decisions require independent review.",
            "No annotation-to-video alignment or tracking performance is measured.",
        ],
    }
