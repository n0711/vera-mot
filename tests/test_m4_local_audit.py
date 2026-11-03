"""Pure synthetic tests for M4's read-only local file inventory."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_m4_local.py"
SPEC = importlib.util.spec_from_file_location("audit_m4_local", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def workspace(tmp_path):
    config = {
        "train_sequences": [f"A{x:02d}" for x in range(24)],
        "validation_sequences": [f"V{x:02d}" for x in range(6)],
        "frozen_test_sequences": [f"T{x:02d}" for x in range(20)],
    }
    path = tmp_path / "configs/uavdt_reid_split.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_missing_data_and_checkpoints_reported_read_only(workspace):
    before = sorted(x.relative_to(workspace).as_posix() for x in workspace.rglob("*"))
    report = MODULE.audit(workspace)
    after = sorted(x.relative_to(workspace).as_posix() for x in workspace.rglob("*"))
    assert before == after
    assert not report["all_files_present"]
    assert report["checks"]["source_sequence_directories"]["present"] == 0
    assert report["checks"]["source_ground_truth"]["required"] == 30
    assert report["checks"]["training_checkpoints"]["required"] == 6
    assert report["scope"] == "read_only_filesystem_presence_not_integrity_or_inference"


def test_placeholder_presence_does_not_claim_verification(workspace):
    report = MODULE.audit(workspace)
    directories = {"source_sequence_directories", "reid_crop_directories", "fastreid_submodule"}
    for key, group in report["checks"].items():
        for relative in group["missing"]:
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if key in directories:
                path.mkdir(exist_ok=True)
            else:
                path.touch()
    complete = MODULE.audit(workspace)
    assert complete["all_files_present"]
    assert "not_integrity_or_inference" in complete["scope"]
    assert complete["next_action"] == "validate hashes and data content"


def test_overlapping_frozen_splits_rejected(workspace):
    path = workspace / "configs/uavdt_reid_split.yaml"
    config = json.loads(path.read_text())
    config["validation_sequences"][0] = config["train_sequences"][0]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overlap"):
        MODULE.audit(workspace)


def test_missing_config_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing or invalid"):
        MODULE.audit(tmp_path)
