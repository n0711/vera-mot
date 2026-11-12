"""M4.8 regression checks, synthetic metadata/annotations only, no video decode."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vera_mot.reid_ablation_audit import AuditError
from vera_mot.reid_eval_preflight import build_preflight


REPO_ROOT = Path(__file__).resolve().parents[1]
GT_ROWS = (
    "1,1,10,20,30,40,1,1,2\n"
    "2,1,11,20,30,40,1,1,2\n"
    "3,2,30,40,22,18,1,1,3\n"
)
TRACK_ROWS = (
    "1,1,10,20,30,40,0.8000,-1,-1,-1\n"
    "2,1,11,20,30,40,0.8100,-1,-1,-1\n"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def experiment(tmp_path: Path) -> dict:
    root = tmp_path / "repository"
    (root / "configs").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    config = json.loads((REPO_ROOT / "configs/uavdt_reid_split.yaml").read_text())
    (root / "configs/uavdt_reid_split.yaml").write_text(json.dumps(config))
    original_audit = {
        "sequences": [{
            "sequence": "M0201", "split": "train", "frames": 3,
            "gt_rows": 3, "gt_columns": 9, "gt_exists": True,
        }],
    }
    (root / "docs/uavdt_mot_audit.json").write_text(json.dumps(original_audit))
    video = tmp_path / "source.mp4"
    detections = tmp_path / "detections.csv"
    video.write_bytes(b"synthetic-placeholder-source-bytes-not-video")
    detections.write_text("frame,x1,y1,x2,y2,score\n")
    gt = tmp_path / "M0201_gt_whole.txt"
    gt.write_text(GT_ROWS)

    def create_run(name: str, gated: bool) -> Path:
        mot = tmp_path / f"{name}.txt"
        mot.write_text(TRACK_ROWS)
        provenance = {
            "scope": "unvalidated_pinned_botsort_appearance_tracking",
            "tracker": "pinned_BoT-SORT",
            "mode": "vera_quality_gated_reid" if gated else "appearance_reid",
            "embedding": "test-hsv-encoder",
            "cmc_method": "none",
            "input_video_sha256": digest(video),
            "input_detection_sha256": digest(detections),
            "embedding_weights_sha256": None,
            "frame_count": 3,
            "encoder_calls": 3,
            "extracted_embeddings": 2,
            "reported_track_observations": 2,
            "tentative_unique_ids": 1,
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
        mot.with_suffix(".provenance.json").write_text(json.dumps(provenance))
        return mot

    return {
        "project_root": root,
        "sequence": "M0201",
        "video": video,
        "detections": detections,
        "ground_truth": gt,
        "ungated": create_run("ungated", False),
        "gated": create_run("gated", True),
        "expected_threshold": 0.5,
    }


def execute(args: dict) -> dict:
    return build_preflight(**args)


def test_success_with_valid_original_nine_column_synthetic_gt(experiment):
    report = execute(experiment)
    assert report["status"] == "passed_prerequisite_consistency_only"
    assert report["split"] == "frozen_reid_validation"
    assert report["ground_truth"]["annotation_count"] == 3
    assert report["ground_truth"]["sequence_local_identity_count"] == 2
    assert report["inputs_sha256"]["video"] == digest(experiment["video"])
    assert report["inputs_sha256"]["ground_truth"] == digest(experiment["ground_truth"])
    assert report == execute(experiment)
    assert "hota" not in report
    assert str(experiment["project_root"]) not in json.dumps(report)


@pytest.mark.parametrize("input_key,expected", [
    ("video", "video bytes"),
    ("detections", "detection bytes"),
])
def test_original_input_byte_tampering_fails(experiment, input_key, expected):
    experiment[input_key].write_bytes(b"tampered-source")
    with pytest.raises(AuditError, match=expected):
        execute(experiment)


@pytest.mark.parametrize("sequence,reason", [
    ("M0203", "frozen-test"),
    ("M0101", "not in the frozen ReID validation"),
    ("../M0201", "invalid UAVDT sequence"),
])
def test_invalid_sequences_rejected_before_using_ground_truth(experiment, sequence, reason):
    experiment["sequence"] = sequence
    with pytest.raises(AuditError, match=reason):
        execute(experiment)


@pytest.mark.parametrize("bad_rows,reason", [
    ("1,1,10,20,30,40,1,1,2,0\n", "expected 9 UAVDT"),
    ("1,1,10,20,30,40,1,1\n", "expected 9 UAVDT"),
    ("1,1,10,20,30,40,1,1,2\n1,1,10,20,30,40,1,1,2\n", "duplicate"),
    ("1.5,1,10,20,30,40,1,1,2\n", "positive integers"),
    ("1,0,10,20,30,40,1,1,2\n", "positive integers"),
    ("4,1,10,20,30,40,1,1,2\n", "exceeds"),
    ("1,1,10,20,nan,40,1,1,2\n", "nonfinite"),
    ("1,1,10,20,-30,40,1,1,2\n", "dimensions"),
    ("", "empty"),
])
def test_invalid_gt_fails_closed(experiment, bad_rows, reason):
    experiment["ground_truth"].write_text(bad_rows)
    with pytest.raises(AuditError, match=reason):
        execute(experiment)


def test_expected_annotation_count_must_match_original_audit(experiment):
    experiment["ground_truth"].write_text(GT_ROWS.splitlines()[0] + "\n")
    with pytest.raises(AuditError, match="row count mismatch"):
        execute(experiment)


def test_ground_truth_name_must_match_sequence(experiment):
    other = experiment["ground_truth"].with_name("M0202_gt_whole.txt")
    other.write_text(GT_ROWS)
    experiment["ground_truth"] = other
    with pytest.raises(AuditError, match="filename"):
        execute(experiment)


def test_predeclared_threshold_mismatch_is_rejected(experiment):
    experiment["expected_threshold"] = 0.6
    with pytest.raises(AuditError, match="threshold deviates"):
        execute(experiment)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_threshold_fails(experiment, threshold):
    experiment["expected_threshold"] = threshold
    with pytest.raises(AuditError, match="expected threshold"):
        execute(experiment)


def test_frame_count_must_match_source_inventory(experiment):
    audit_path = experiment["project_root"] / "docs/uavdt_mot_audit.json"
    source_audit = json.loads(audit_path.read_text())
    source_audit["sequences"][0]["frames"] = 4
    audit_path.write_text(json.dumps(source_audit))
    with pytest.raises(AuditError, match="frame count"):
        execute(experiment)


def test_split_overlap_detected(experiment):
    split = experiment["project_root"] / "configs/uavdt_reid_split.yaml"
    config = json.loads(split.read_text())
    config["train_sequences"].append("M0201")
    split.write_text(json.dumps(config))
    with pytest.raises(AuditError, match="overlapping"):
        execute(experiment)


def test_changed_frozen_partition_detected(experiment):
    split = experiment["project_root"] / "configs/uavdt_reid_split.yaml"
    config = json.loads(split.read_text())
    config["frozen_test_sequences"].remove("M0203")
    split.write_text(json.dumps(config))
    with pytest.raises(AuditError, match="frozen test partition"):
        execute(experiment)


def test_pair_source_mismatch_still_fails(experiment):
    path = experiment["gated"].with_suffix(".provenance.json")
    prov = json.loads(path.read_text())
    prov["input_video_sha256"] = "c" * 64
    path.write_text(json.dumps(prov))
    with pytest.raises(AuditError, match="input_video_sha256"):
        execute(experiment)


def test_original_inputs_are_unchanged(experiment):
    input_paths = [experiment[key] for key in (
        "video", "detections", "ground_truth", "ungated", "gated")]
    before = [path.read_bytes() for path in input_paths]
    execute(experiment)
    assert before == [path.read_bytes() for path in input_paths]
