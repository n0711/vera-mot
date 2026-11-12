"""M4.9: synthetic UAVDT GT conversion tests, no dataset or model required."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vera_mot.reid_ablation_audit import AuditError
from vera_mot.uavdt_gt_convert import (
    convert_text, parse_source, prepare_conversion, write_conversion,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (
    "3,2,15,25,11,18,3,4,3\n"
    "1,2,-2,5,12,16,1,1,1\n"
    "2,5,33,21,10,13,2,2,2\n"
)
EXPECTED = (
    "1,2,-2,5,12,16,1,1,-1,-1\n"
    "2,5,33,21,10,13,1,1,-1,-1\n"
    "3,2,15,25,11,18,1,1,-1,-1\n"
)


def test_all_supported_codes_geometry_and_canonical_order():
    out, report = convert_text(SAMPLE, frame_count=4)
    assert out == EXPECTED
    assert report["record_count"] == 3
    assert report["sequence_local_identities"] == 2
    assert report["raw_code_histograms"] == {
        "out_of_view": {"1": 1, "2": 1, "3": 1},
        "occlusion": {"1": 1, "2": 1, "4": 1},
        "category": {"1": 1, "2": 1, "3": 1},
    }
    assert hashlib.sha256(out.encode()).hexdigest() == report["converted_sha256"]
    assert (out, report) == convert_text(SAMPLE, frame_count=4)


@pytest.mark.parametrize("invalid,matching", [
    ("1,1,1,2,30,40,1,1\n", "9 UAVDT"),
    ("1,1,1,2,30,40,1,1,1,4\n", "9 UAVDT"),
    ("1.0,1,1,2,30,40,1,1,1\n", "decimal integer"),
    ("nan,1,1,2,30,40,1,1,1\n", "decimal integer"),
    ("0,1,1,2,30,40,1,1,1\n", "out of range"),
    ("5,1,1,2,30,40,1,1,1\n", "out of range"),
    ("1,0,1,2,30,40,1,1,1\n", "out of range"),
    ("1,1,1,2,0,40,1,1,1\n", "dimensions"),
    ("1,1,1,2,30,-40,1,1,1\n", "dimensions"),
    ("1,1,1,2,30,40,0,1,1\n", "out_of_view"),
    ("1,1,1,2,30,40,4,1,1\n", "out_of_view"),
    ("1,1,1,2,30,40,1,0,1\n", "occlusion"),
    ("1,1,1,2,30,40,1,5,1\n", "occlusion"),
    ("1,1,1,2,30,40,1,1,0\n", "category"),
    ("1,1,1,2,30,40,1,1,4\n", "category"),
    ("1,1,1,2,30,40,1,1,1\n1,1,1,2,30,40,1,1,1\n", "duplicate"),
    ("", "empty"),
])
def test_invalid_rows_fail_closed(invalid, matching):
    with pytest.raises(AuditError, match=matching):
        convert_text(invalid, frame_count=4)


@pytest.mark.parametrize("bad", [0, -1, 2.3, True, "4"])
def test_frame_count_type_and_range_checked(bad):
    with pytest.raises(AuditError, match="frame_count"):
        parse_source(SAMPLE, frame_count=bad)


@pytest.fixture
def fake_project(tmp_path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    config = json.loads((ROOT / "configs/uavdt_reid_split.yaml").read_text())
    (root / "configs/uavdt_reid_split.yaml").write_text(json.dumps(config))
    inventory = {"sequences": [{
        "sequence": "M0201", "split": "train", "frames": 4,
        "gt_rows": 3, "gt_columns": 9, "gt_exists": True,
    }]}
    (root / "docs/uavdt_mot_audit.json").write_text(json.dumps(inventory))
    original = tmp_path / "M0201_gt_whole.txt"
    original.write_text(SAMPLE)
    return root, original


def test_prepare_validates_original_source_inventory_and_fingerprints(fake_project):
    root, original = fake_project
    output, report = prepare_conversion(root, sequence="M0201", source=original)
    assert output == EXPECTED
    assert report["source_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert report["record_count"] == 3
    assert report["policy"]["gt_class_1_is_trackeval_pedestrian_compatibility_alias"]
    assert "hota" not in json.dumps(report).lower()
    assert "original_nine_column" in report["source_type"]
    assert report["trackeval_pinned_commit"] == "12c8791b303e0a0b50f753af204249e622d0281a"


def test_wrong_source_filename_is_rejected(fake_project):
    root, original = fake_project
    renamed = original.with_name("M0201_gt.txt")
    renamed.write_bytes(original.read_bytes())
    with pytest.raises(AuditError, match="gt_whole"):
        prepare_conversion(root, sequence="M0201", source=renamed)


def test_frozen_test_and_training_sequences_refused(fake_project):
    root, original = fake_project
    for sequence in ("M0203", "M0101"):
        with pytest.raises(AuditError, match="frozen-test|not in the frozen ReID validation"):
            prepare_conversion(root, sequence=sequence, source=original)


def test_source_count_must_match_inventory(fake_project):
    root, original = fake_project
    inventory_file = root / "docs/uavdt_mot_audit.json"
    inventory = json.loads(inventory_file.read_text())
    inventory["sequences"][0]["gt_rows"] = 4
    inventory_file.write_text(json.dumps(inventory))
    with pytest.raises(AuditError, match="row count"):
        prepare_conversion(root, sequence="M0201", source=original)


def test_exclusive_outputs_and_source_immutability(fake_project, tmp_path):
    root, original = fake_project
    out, report = prepare_conversion(root, sequence="M0201", source=original)
    results = tmp_path / "converted"
    results.mkdir()
    before = original.read_bytes()
    gt, manifest = write_conversion(results, sequence="M0201", converted=out, report=report)
    assert gt.read_text() == EXPECTED
    assert json.loads(manifest.read_text()) == report
    assert original.read_bytes() == before
    with pytest.raises(AuditError, match="already exists"):
        write_conversion(results, sequence="M0201", converted=out, report=report)


def test_output_hash_verification_and_missing_dir(fake_project, tmp_path):
    root, original = fake_project
    out, report = prepare_conversion(root, sequence="M0201", source=original)
    results = tmp_path / "results"
    with pytest.raises(AuditError, match="already exist"):
        write_conversion(results, sequence="M0201", converted=out, report=report)
    results.mkdir()
    with pytest.raises(AuditError, match="hash"):
        write_conversion(results, sequence="M0201", converted=out + "bad", report=report)
    assert list(results.iterdir()) == []
