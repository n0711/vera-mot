"""Read-only integrity audit of matched, offline BoT-SORT ReID runs.

This is a *comparability* check, not a ground-truth identity evaluation.
The audit never reads, modifies, or exports source frames or detections.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vera_mot.mot_io import MotRow


SCHEMA_VERSION = "vera-reid-paired-audit-v1"
SCOPE = "unvalidated_pinned_botsort_appearance_tracking"
GATE_SCOPE = "unvalidated_image_quality_proxy_not_identity_reliability"


class AuditError(ValueError):
    """An input, provenance, or comparability invariant failed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_path(mot_path: Path) -> Path:
    # Exactly matches run_botsort_reid_detections.py's with_suffix contract.
    return mot_path.with_suffix(".provenance.json")


def _read_provenance(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"cannot read provenance: {path}") from exc
    if not isinstance(raw, dict):
        raise AuditError(f"provenance must be a JSON object: {path}")
    return raw


def _nonnegative_integer(data: dict, key: str) -> int:
    value = data.get(key)
    if type(value) is not int or value < 0:
        raise AuditError(f"{key} must be a nonnegative integer")
    return value


def _sha256_field(data: dict, name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or len(value) != 64 or any(
        c not in "0123456789abcdef" for c in value
    ):
        raise AuditError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _read_tracks(path: Path, frame_count: int) -> list[MotRow]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditError(f"cannot read MOT output: {path}") from exc
    rows: list[MotRow] = []
    seen: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 10:
            raise AuditError(f"{path}:{line_number}: expected 10 MOT fields")
        for column in (0, 1):
            try:
                numeric = float(fields[column])
            except ValueError as exc:
                raise AuditError(f"{path}:{line_number}: invalid index") from exc
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise AuditError(f"{path}:{line_number}: nonintegral frame or track ID")
        try:
            row = MotRow.parse(line, source=f"{path}:{line_number}")
        except ValueError as exc:
            raise AuditError(str(exc)) from exc
        if not all(math.isfinite(value) for value in (
            row.x, row.y, row.width, row.height, row.score,
            row.world_x, row.world_y, row.world_z,
        )):
            raise AuditError(f"{path}:{line_number}: nonfinite MOT value")
        if row.frame > frame_count:
            raise AuditError(f"{path}:{line_number}: frame exceeds video length")
        if not 0 <= row.score <= 1:
            raise AuditError(f"{path}:{line_number}: tracking score outside [0, 1]")
        key = row.frame, row.track_id
        if key in seen:
            raise AuditError(f"{path}:{line_number}: duplicate track ID in frame")
        seen.add(key)
        rows.append(row)
    return rows


def _validate_gate(gate: Any) -> dict:
    if not isinstance(gate, dict) or gate.get("scope") != GATE_SCOPE:
        raise AuditError("missing or invalid experimental quality-gate provenance")
    counts = {key: _nonnegative_integer(gate, key) for key in (
        "assessed", "appearance_allowed", "appearance_blocked",
    )}
    if counts["appearance_allowed"] + counts["appearance_blocked"] != counts["assessed"]:
        raise AuditError("quality-gate counters do not add up")
    for key in ("threshold", "reference_area", "reference_laplacian_variance"):
        val = gate.get(key)
        if type(val) not in (float, int) or not math.isfinite(val):
            raise AuditError(f"quality-gate {key} must be finite")
        if key == "threshold" and not 0 <= val <= 1:
            raise AuditError("quality-gate threshold outside [0, 1]")
        if key != "threshold" and val <= 0:
            raise AuditError(f"quality-gate {key} must be positive")
    mean = gate.get("mean_heuristic_quality")
    if counts["assessed"] == 0:
        if mean is not None:
            raise AuditError("quality-gate mean must be null for zero observations")
    elif type(mean) not in (int, float) or not math.isfinite(mean) or not 0 <= mean <= 1:
        raise AuditError("invalid quality-gate mean")
    return counts


def _summarize_tracks(rows: list[MotRow]) -> dict[str, int]:
    track_frames: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        track_frames[row.track_id].append(row.frame)
    gaps = sum(
        sum(right - left > 1 for left, right in zip(frames, frames[1:]))
        for frames in (sorted(xs) for xs in track_frames.values())
    )
    lengths = Counter(len(frames) for frames in track_frames.values())
    return {
        "observations": len(rows),
        "frames_with_observations": len({row.frame for row in rows}),
        "tentative_track_ids": len(track_frames),
        "single_observation_tracks": lengths[1],
        "track_gaps": gaps,
    }


def _validate_run(mot: Path, *, mode: str, gated: bool) -> tuple[dict, dict]:
    metadata_path = _provenance_path(mot)
    prov = _read_provenance(metadata_path)
    if prov.get("scope") != SCOPE or prov.get("tracker") != "pinned_BoT-SORT":
        raise AuditError(f"unsupported tracker/evidence scope: {mot}")
    if prov.get("mode") != mode or not isinstance(prov.get("embedding"), str) or not prov["embedding"]:
        raise AuditError(f"wrong ReID mode or missing embedding: {mot}")
    if not isinstance(prov.get("cmc_method"), str) or not prov["cmc_method"]:
        raise AuditError(f"missing camera-motion method: {mot}")
    for field in ("input_video_sha256", "input_detection_sha256"):
        _sha256_field(prov, field)
    frames = _nonnegative_integer(prov, "frame_count")
    if frames < 1:
        raise AuditError("source video must contain frames")
    for field in ("encoder_calls", "extracted_embeddings", "reported_track_observations", "tentative_unique_ids"):
        _nonnegative_integer(prov, field)
    rows = _read_tracks(mot, frames)
    stats = _summarize_tracks(rows)
    if prov["reported_track_observations"] != stats["observations"] or (
        prov["tentative_unique_ids"] != stats["tentative_track_ids"]
    ):
        raise AuditError(f"MOT rows do not match provenance counters: {mot}")
    if gated:
        stats.update({f"gate_{key}": value for key, value in _validate_gate(prov.get("quality_gate")).items()})
    elif prov.get("quality_gate") is not None:
        raise AuditError("ungated run unexpectedly contains quality-gate evidence")
    return prov, {
        "mot_sha256": _file_sha256(mot),
        "provenance_sha256": _file_sha256(metadata_path),
        "counts": stats,
    }


def audit_pair(ungated_mot: Path, gated_mot: Path) -> dict:
    """Check paired inputs, format and read-only, aggregate output properties.

    Numeric tracker IDs are arbitrary across runs. No identity accuracy,
    improvement, or ground-truth association metric is inferred from them.
    """
    ungated_mot, gated_mot = Path(ungated_mot), Path(gated_mot)
    if ungated_mot.resolve() == gated_mot.resolve():
        raise AuditError("paired MOT paths must differ")
    plain, plain_report = _validate_run(ungated_mot, mode="appearance_reid", gated=False)
    gated, gate_report = _validate_run(gated_mot, mode="vera_quality_gated_reid", gated=True)
    fields = (
        "input_video_sha256", "input_detection_sha256", "cmc_method",
        "embedding", "embedding_weights_sha256", "frame_count",
        "encoder_calls", "extracted_embeddings",
    )
    for field in fields:
        if plain.get(field) != gated.get(field):
            raise AuditError(f"paired runs mismatch on {field}")
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "read_only_paired_integrity_and_descriptive_statistics_only",
        "paired_inputs_verified": True,
        "shared_inputs": {field: plain.get(field) for field in fields},
        "ungated": plain_report,
        "gated": gate_report,
        "quality_gate_configuration": {
            key: gated["quality_gate"][key]
            for key in ("method", "threshold", "reference_area", "reference_laplacian_variance")
        },
        "scientific_limitations": [
            "No annotated identities or ground-truth MOT files are consulted.",
            "MOT track IDs are arbitrary across runs and cannot be compared directly.",
            "Track counts and gaps are descriptive, not HOTA, IDF1, AssA, IDSW, or evidence of improvement.",
            "Source video/checkpoint authenticity is not established by reported provenance hashes.",
            "Quality proxies are not calibrated probabilities of correct identity.",
        ],
    }
