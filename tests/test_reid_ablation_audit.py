"""Data-free paired audit tests, no tracker or UAV dataset required."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vera_mot.reid_ablation_audit import AuditError, audit_pair


HASH_A, HASH_B = "a" * 64, "b" * 64


def _run(tmp_path: Path, name: str, gated: bool, rows: str | None = None) -> Path:
    path = tmp_path / f"{name}.txt"
    path.write_text(
        "1,1,10,20,30,40,0.8000,-1,-1,-1\n"
        "2,1,11,20,30,40,0.8100,-1,-1,-1\n"
        if rows is None else rows,
        encoding="utf-8",
    )
    unique = {line.split(",")[1] for line in path.read_text().splitlines() if line.strip()}
    prov = {
        "scope": "unvalidated_pinned_botsort_appearance_tracking",
        "tracker": "pinned_BoT-SORT",
        "mode": "vera_quality_gated_reid" if gated else "appearance_reid",
        "embedding": "hsv-histogram-test",
        "cmc_method": "none",
        "input_video_sha256": HASH_A,
        "input_detection_sha256": HASH_B,
        "embedding_weights_sha256": None,
        "frame_count": 3,
        "encoder_calls": 3,
        "extracted_embeddings": 2,
        "reported_track_observations": len(path.read_text().splitlines()),
        "tentative_unique_ids": len(unique),
        "quality_gate": {
            "method": "vera-crop-quality-gate-v0",
            "scope": "unvalidated_image_quality_proxy_not_identity_reliability",
            "threshold": 0.5,
            "reference_area": 1024.0,
            "reference_laplacian_variance": 120.0,
            "assessed": 2,
            "appearance_allowed": 1,
            "appearance_blocked": 1,
            "mean_heuristic_quality": 0.4,
        } if gated else None,
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return path


def _change(mot: Path, **values):
    path = mot.with_suffix(".provenance.json")
    prov = json.loads(path.read_text())
    prov.update(values)
    path.write_text(json.dumps(prov))


def test_paired_audit_reports_only_descriptive_evidence(tmp_path):
    plain, gated = _run(tmp_path, "plain", False), _run(tmp_path, "gated", True)
    first = audit_pair(plain, gated)
    assert first == audit_pair(plain, gated)
    assert first["paired_inputs_verified"]
    assert first["ungated"]["counts"]["observations"] == 2
    assert first["gated"]["counts"]["gate_appearance_blocked"] == 1
    assert first["ungated"]["mot_sha256"] == hashlib.sha256(plain.read_bytes()).hexdigest()
    assert not any(key.lower() in first for key in ("idf1", "hota", "idsw", "improvement"))
    assert "ground-truth" in " ".join(first["scientific_limitations"])


@pytest.mark.parametrize("key,new_value", [
    ("input_video_sha256", "c" * 64),
    ("input_detection_sha256", "c" * 64),
    ("cmc_method", "orb"),
    ("embedding", "different"),
    ("frame_count", 4),
    ("extracted_embeddings", 1),
])
def test_mismatched_pair_fails_closed(tmp_path, key, new_value):
    plain, gated = _run(tmp_path, "plain", False), _run(tmp_path, "gated", True)
    _change(gated, **{key: new_value})
    with pytest.raises(AuditError, match=key):
        audit_pair(plain, gated)


@pytest.mark.parametrize("rows,error", [
    ("1,1,10,20,30,40,0.8,-1,-1,-1\n1,1,10,20,30,40,0.8,-1,-1,-1\n", "duplicate"),
    ("1.5,1,10,20,30,40,0.8,-1,-1,-1\n", "nonintegral"),
    ("4,1,10,20,30,40,0.8,-1,-1,-1\n", "exceeds"),
    ("1,1,10,20,NaN,40,0.8,-1,-1,-1\n", "nonfinite"),
    ("1,1,10,20,30,40,1.5,-1,-1,-1\n", "score"),
    ("1,1,10,20,0,40,0.8,-1,-1,-1\n", "positive"),
])
def test_invalid_mot_rows_rejected(tmp_path, rows, error):
    plain, gated = _run(tmp_path, "plain", False), _run(tmp_path, "gated", True, rows=rows)
    with pytest.raises(AuditError, match=error):
        audit_pair(plain, gated)


def test_corrupted_counts_and_gate_fail_closed(tmp_path):
    plain, gated = _run(tmp_path, "plain", False), _run(tmp_path, "gated", True)
    _change(plain, reported_track_observations=77)
    with pytest.raises(AuditError, match="provenance counters"):
        audit_pair(plain, gated)
    _change(plain, reported_track_observations=2)
    prov_path = gated.with_suffix(".provenance.json")
    prov = json.loads(prov_path.read_text())
    prov["quality_gate"]["appearance_allowed"] = 2
    prov_path.write_text(json.dumps(prov))
    with pytest.raises(AuditError, match="counters"):
        audit_pair(plain, gated)


def test_empty_track_outputs_are_valid_if_provenance_matches(tmp_path):
    plain, gated = _run(tmp_path, "plain", False, rows=""), _run(tmp_path, "gated", True, rows="")
    report = audit_pair(plain, gated)
    assert report["gated"]["counts"]["tentative_track_ids"] == 0


def test_input_files_are_never_modified(tmp_path):
    plain, gated = _run(tmp_path, "plain", False), _run(tmp_path, "gated", True)
    paths = (plain, gated, plain.with_suffix(".provenance.json"), gated.with_suffix(".provenance.json"))
    before = [p.read_bytes() for p in paths]
    audit_pair(plain, gated)
    assert before == [p.read_bytes() for p in paths]
