"""Synthetic M4.9 reference comparison and command-line regression checks."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from vera_mot.reid_ablation_audit import AuditError
from vera_mot.uavdt_gt_compare import compare_converted_text, compare_files

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "1,7,-2,20,8,6,1,1,2\n2,7,10,20,8,6,1,1,2\n"
CONVERTED = "1,7,-2,20,8,6,1,1,-1,-1\n2,7,10,20,8,6,1,1,-1,-1\n"


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "configs/uavdt_reid_split.yaml").write_bytes(
        (ROOT / "configs/uavdt_reid_split.yaml").read_bytes()
    )
    (root / "docs/uavdt_mot_audit.json").write_text(json.dumps({"sequences": [{
        "sequence": "M0201", "split": "train", "frames": 2,
        "gt_rows": 2, "gt_columns": 9, "gt_exists": True,
    }]}))
    source = tmp_path / "M0201_gt_whole.txt"
    source.write_text(SOURCE)
    reference = tmp_path / "M0201_gt.txt"
    reference.write_text(CONVERTED)
    return root, source, reference


def test_numeric_equivalence_ignores_order_and_trailing_metadata():
    reference = (
        "2.0,7,10.00,20,8,6,1,1,0.5,-1\n"
        "1,7,-2.0,20,8,6,1,1,-1,-1\n"
    )
    report = compare_converted_text(CONVERTED, reference)
    assert report["structural_match"]
    assert report["matching_rows"] == 2


def test_discrepancies_are_counted_independently():
    reference = "1,7,-2,20,9,6,1,1,-1,-1\n2,8,10,20,8,6,1,1,-1,-1\n"
    report = compare_converted_text(CONVERTED, reference)
    assert report == {
        "structural_match": False, "source_rows": 2, "reference_rows": 2,
        "matching_rows": 0, "missing_reference_rows": 1,
        "extra_reference_rows": 1, "changed_geometry_rows": 1,
    }


def test_empty_reference_reports_all_missing():
    report = compare_converted_text(CONVERTED, "")
    assert not report["structural_match"]
    assert report["missing_reference_rows"] == 2


@pytest.mark.parametrize("index,value", [
    (0, "1.5"), (0, "0"), (1, "-1"), (2, "nan"), (3, "Infinity"),
    (4, "0"), (5, "-1"), (6, "0"), (7, "2"), (9, "unknown"),
])
def test_invalid_reference_rejected(index, value):
    fields = CONVERTED.splitlines()[0].split(",")
    fields[index] = value
    with pytest.raises(AuditError):
        compare_converted_text(CONVERTED, ",".join(fields))


@pytest.mark.parametrize("reference", [CONVERTED + CONVERTED, "1,7,0,0,8,6,1,1,1"])
def test_duplicate_or_wrong_width_rejected(reference):
    with pytest.raises(AuditError):
        compare_converted_text(CONVERTED, reference)


def test_files_fingerprinted_and_immutable(inputs, tmp_path):
    root, source, reference = inputs
    ignore = tmp_path / "M0201_gt_ignore.txt"
    ignore.write_bytes(b"opaque ignore fixture\r\n")
    before = {p: p.read_bytes() for p in (source, reference, ignore)}
    report = compare_files(root, sequence="M0201", source=source,
                           reference=reference, ignore_regions=ignore)
    assert report["structural_match"]
    assert report["reference_sha256"] == hashlib.sha256(before[reference]).hexdigest()
    assert report["ignore_regions_sha256"] == hashlib.sha256(before[ignore]).hexdigest()
    assert report["policy"]["ignore_regions"] == "fingerprinted_only_not_applied"
    assert before == {p: p.read_bytes() for p in before}
    assert compare_files(root, sequence="M0201", source=source,
                         reference=reference)["ignore_regions_sha256"] is None


def test_reference_name_and_split_gates(inputs):
    root, source, reference = inputs
    with pytest.raises(AuditError, match="reference filename"):
        compare_files(root, sequence="M0201", source=source, reference=source)
    with pytest.raises(AuditError, match="frozen-test"):
        compare_files(root, sequence="M0203", source=source, reference=reference)


def test_cli_match_mismatch_and_invalid_input(inputs, monkeypatch, capsys):
    root, source, reference = inputs
    spec = importlib.util.spec_from_file_location("compare_cli", ROOT / "scripts/compare_uavdt_gt.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr("sys.argv", [
        "compare_uavdt_gt.py", "--sequence", "M0201",
        "--source", str(source), "--reference", str(reference),
    ])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["structural_match"]
    reference.write_text("")
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)["missing_reference_rows"] == 2
    reference.write_text("invalid")
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
