"""M4.8 read-only UAVDT validation preflight for paired ReID experiments.

Validates locally available source bytes and 9-column UAVDT annotations, without
calculating identity metrics or changing the frozen no-ReID benchmark.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from vera_mot.fastreid_uavdt import FROZEN_TEST_SEQUENCES
from vera_mot.reid_ablation_audit import AuditError, audit_pair


SCHEMA_VERSION = "vera-mot-annotated-preflight-v1"
SEQUENCE_RE = re.compile(r"M[0-9]{4}\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AuditError(f"missing or unreadable input: {path.name}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"missing or invalid {label}") from exc
    if not isinstance(data, dict):
        raise AuditError(f"{label} must contain an object")
    return data


def _validate_sequence(sequence: str, config: dict, audit: dict) -> dict:
    if not isinstance(sequence, str) or SEQUENCE_RE.fullmatch(sequence) is None:
        raise AuditError("invalid UAVDT sequence name")
    if config.get("schema_version") != 1:
        raise AuditError("unsupported UAVDT split schema")
    keys = ("train_sequences", "validation_sequences", "frozen_test_sequences")
    splits: dict[str, set[str]] = {}
    for key in keys:
        values = config.get(key)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or SEQUENCE_RE.fullmatch(value) is None
            for value in values
        ):
            raise AuditError(f"missing or invalid {key}")
        if len(values) != len(set(values)):
            raise AuditError(f"duplicate sequences in {key}")
        splits[key] = set(values)
    if any(splits[a] & splits[b] for a, b in (
        (keys[0], keys[1]), (keys[0], keys[2]), (keys[1], keys[2])
    )):
        raise AuditError("overlapping train, validation or test sequences")
    if splits["frozen_test_sequences"] != FROZEN_TEST_SEQUENCES:
        raise AuditError("the frozen test partition must remain unchanged")
    if sequence in FROZEN_TEST_SEQUENCES:
        raise AuditError("frozen-test sequences cannot be used for this preflight")
    if sequence not in splits["validation_sequences"]:
        raise AuditError("sequence is not in the frozen ReID validation partition")
    entries = audit.get("sequences")
    if not isinstance(entries, list):
        raise AuditError("missing original UAVDT sequence audit records")
    matches = [entry for entry in entries if isinstance(entry, dict)
               and entry.get("sequence") == sequence]
    if len(matches) != 1 or matches[0].get("split") != "train":
        raise AuditError("validation sequence is not unique in the original train-set audit")
    record = matches[0]
    for field in ("frames", "gt_rows"):
        if type(record.get(field)) is not int or record[field] < 1:
            raise AuditError(f"invalid original UAVDT {field} count")
    if record.get("gt_columns") != 9 or record.get("gt_exists") is not True:
        raise AuditError("original audit does not declare available 9-column ground truth")
    return record


def _positive_index(field: str, source: str) -> int:
    try:
        value = float(field)
    except ValueError as exc:
        raise AuditError(f"{source}: invalid frame or identity") from exc
    if not math.isfinite(value) or not value.is_integer() or value < 1:
        raise AuditError(f"{source}: frame and identity must be positive integers")
    return int(value)


def inspect_uavdt_gt(path: Path, *, frame_count: int) -> dict[str, int]:
    """Read original nine-column UAVDT MOT ground truth without reinterpreting codes.

    Columns 7-9 remain uninterpreted source codes, not MOTChallenge 'confidence,
    class, visibility' fields. No TrackEval conversion is attempted.
    """
    observed: set[tuple[int, int]] = set()
    frame_set: set[int] = set()
    identity_set: set[int] = set()
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                source = f"ground truth line {line_number}"
                cols = [s.strip() for s in line.strip().split(",")]
                if len(cols) != 9:
                    raise AuditError(f"{source}: expected 9 UAVDT columns")
                frame, identity = (_positive_index(cols[0], source),
                                   _positive_index(cols[1], source))
                try:
                    values = [float(x) for x in cols[2:]]
                except ValueError as exc:
                    raise AuditError(f"{source}: nonnumeric coordinate/source code") from exc
                if not all(math.isfinite(x) for x in values):
                    raise AuditError(f"{source}: nonfinite coordinate/source code")
                if values[2] <= 0 or values[3] <= 0:
                    raise AuditError(f"{source}: bounding-box dimensions must be positive")
                if frame > frame_count:
                    raise AuditError(f"{source}: annotation frame exceeds declared sequence length")
                if (frame, identity) in observed:
                    raise AuditError(f"{source}: duplicate identity in the same frame")
                observed.add((frame, identity))
                frame_set.add(frame)
                identity_set.add(identity)
                count += 1
    except OSError as exc:
        raise AuditError("missing or unreadable UAVDT ground truth") from exc
    if not count:
        raise AuditError("empty UAVDT ground-truth annotations")
    return {
        "annotation_count": count,
        "annotated_frame_count": len(frame_set),
        "sequence_local_identity_count": len(identity_set),
        "maximum_annotated_frame": max(frame_set),
    }


def build_preflight(
    project_root: Path,
    *,
    sequence: str,
    ground_truth: Path,
    ungated: Path,
    gated: Path,
    video: Path,
    detections: Path,
    expected_threshold: float,
) -> dict:
    """Validate all prerequisite files and paired evidence without inference.

    Input hashes are checked against *actual local source bytes*, improving
    on the self-reported-only comparison in M4.7. Successful preflight is not
    evidence that video contents match GT identities or detector output.
    """
    if type(expected_threshold) not in (int, float) or not math.isfinite(expected_threshold):
        raise AuditError("expected threshold must be finite")
    if not 0 <= expected_threshold <= 1:
        raise AuditError("expected threshold must be within [0, 1]")
    project_root = Path(project_root)
    config_path = project_root / "configs/uavdt_reid_split.yaml"
    audit_path = project_root / "docs/uavdt_mot_audit.json"
    config = _read_json(config_path, "frozen split configuration")
    source_audit = _read_json(audit_path, "original sequence audit")
    record = _validate_sequence(sequence, config, source_audit)

    ground_truth = Path(ground_truth)
    ungated, gated = Path(ungated), Path(gated)
    video, detections = Path(video), Path(detections)
    if ground_truth.name != f"{sequence}_gt.txt":
        raise AuditError("ground-truth filename must match the declared sequence")
    paired = audit_pair(ungated, gated)
    shared = paired["shared_inputs"]
    if shared["frame_count"] != record["frames"]:
        raise AuditError("runner frame count differs from original UAVDT audit")
    if file_sha256(video) != shared["input_video_sha256"]:
        raise AuditError("input video bytes do not match the runner provenance")
    if file_sha256(detections) != shared["input_detection_sha256"]:
        raise AuditError("input detection bytes do not match the runner provenance")
    if not math.isclose(
        paired["quality_gate_configuration"]["threshold"],
        expected_threshold, rel_tol=0, abs_tol=1e-12
    ):
        raise AuditError("quality threshold deviates from predeclared value")
    gt_stats = inspect_uavdt_gt(ground_truth, frame_count=shared["frame_count"])
    if gt_stats["annotation_count"] != record["gt_rows"]:
        raise AuditError("original UAVDT annotation row count mismatch")

    # Only content fingerprints and aggregate counts go into the report.
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_prerequisite_consistency_only",
        "sequence": sequence,
        "split": "frozen_reid_validation",
        "expected_threshold": expected_threshold,
        "frame_count": shared["frame_count"],
        "inputs_sha256": {
            "frozen_split": file_sha256(config_path),
            "original_sequence_audit": file_sha256(audit_path),
            "ground_truth": file_sha256(ground_truth),
            "video": shared["input_video_sha256"],
            "detections": shared["input_detection_sha256"],
            "ungated_mot": paired["ungated"]["mot_sha256"],
            "gated_mot": paired["gated"]["mot_sha256"],
            "ungated_provenance": paired["ungated"]["provenance_sha256"],
            "gated_provenance": paired["gated"]["provenance_sha256"],
        },
        "ground_truth": gt_stats,
        "paired_input_consistency": True,
        "limitations": [
            "No video decoding or detector-to-GT image content authentication was performed.",
            "An exact source-byte match proves consistency with local inputs, not authenticity or authorization.",
            "Original nine-column UAVDT annotation codes were not transformed to TrackEval format.",
            "No HOTA, AssA, IDF1, IDSW, identity recovery or accuracy calculation was performed.",
            "No frozen test sequence, GPU evaluation, real video processing or robot control was executed.",
        ],
    }
