from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vera_mot.reid_checkpoint_sweep import (
    ARTIFACTS, CHECKPOINT_NAMES, PRIMARY_CHECKPOINT, build_plan,
    configuration_hashes, record_result, reuse_result,
)
from vera_mot.reid_data import ManifestError, sha256_file
from vera_mot.reid_temporal_eval import (
    Observation, build_temporal_splits, evaluate_sequence_local,
    split_manifest, write_evaluation_outputs,
)


@pytest.fixture
def workspace(tmp_path):
    files = {
        "configs/model.yml": "_BASE_: base.yml\nMODEL: {}\n",
        "configs/base.yml": "INPUT: {SIZE_TEST: [256, 256]}\n",
        "configs/uavdt_reid_split.yaml": "{}\n",
        "registry.csv": "synthetic registry for orchestration tests only\n",
        "scripts/evaluate_uavdt_reid_temporal.py": "# synthetic evaluator\n",
        "scripts/evaluate_uavdt_reid_checkpoints.py": "# synthetic sweep entry point\n",
        "src/vera_mot/reid_temporal_eval.py": "# synthetic source\n",
        "baselines/BoT-SORT/fast_reid/model.py": "# synthetic upstream source\n",
    }
    files.update({f"checkpoints/{name}": f"synthetic checkpoint {name}" for name in CHECKPOINT_NAMES})
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def plan(root, **options):
    defaults = dict(device="cpu", batch_size=8, num_workers=0, max_identities=None, verify_crops=True)
    defaults.update(options)
    return build_plan(root, root / "checkpoints", root / "configs/model.yml", root / "registry.csv",
                      root / "output", **defaults)


def make_result(job):
    rows = [Observation("M0201", "M0201_1", frame, f"crop{frame}.png") for frame in range(1, 5)]
    splits, rejected = build_temporal_splits(rows, ["M0201"], [])
    vectors = {row.crop_path: np.array([1., 0.]) for row in rows}
    metadata = {key: value for key, value in job["inputs"].items()
                if key not in {"configuration_dependencies", "source_sha256"}}
    write_evaluation_outputs(Path(job["output_dir"]), split_manifest(splits, rejected),
                             evaluate_sequence_local(splits, vectors), metadata)


def test_six_checkpoints_and_declared_final_primary(workspace):
    value = plan(workspace)
    assert [job["checkpoint"] for job in value["jobs"]] == list(CHECKPOINT_NAMES)
    assert [job["checkpoint"] for job in value["jobs"] if job["primary"]] == [PRIMARY_CHECKPOINT]
    assert value["evaluation_scope"] == "full_validation"
    assert not (workspace / "output").exists()


def test_missing_checkpoint_fails_before_output_creation(workspace):
    (workspace / "checkpoints/model_0029.pth").unlink()
    with pytest.raises(ManifestError, match="model_0029.pth"):
        plan(workspace)
    assert not (workspace / "output").exists()


@pytest.mark.parametrize("field,value", [("batch_size", 0), ("num_workers", -1), ("max_identities", 0)])
def test_invalid_limits_rejected(workspace, field, value):
    with pytest.raises(ManifestError, match="invalid batch"):
        plan(workspace, **{field: value})


@pytest.mark.parametrize("path", ["configs/base.yml", "src/vera_mot/reid_temporal_eval.py",
                                    "baselines/BoT-SORT/fast_reid/model.py", "checkpoints/model_final.pth"])
def test_input_changes_invalidate_resume_fingerprint(workspace, path):
    before = plan(workspace)
    with (workspace / path).open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    after = plan(workspace)
    assert before["jobs"][-1]["input_fingerprint"] != after["jobs"][-1]["input_fingerprint"]


def test_configuration_cycle_and_missing_dependency_rejected(workspace):
    (workspace / "configs/base.yml").write_text("_BASE_: model.yml\n")
    with pytest.raises(ManifestError, match="cycle"):
        configuration_hashes(workspace / "configs/model.yml")
    (workspace / "configs/base.yml").unlink()
    with pytest.raises(ManifestError, match="cannot read configuration"):
        configuration_hashes(workspace / "configs/model.yml")


def test_smoke_scope_is_explicit_and_changes_fingerprint(workspace):
    full = plan(workspace)["jobs"][0]
    smoke = plan(workspace, max_identities=2)["jobs"][0]
    assert smoke["inputs"]["evaluation_scope"] == "smoke"
    assert full["input_fingerprint"] != smoke["input_fingerprint"]


def test_completed_results_can_be_reused_with_verified_artifacts(workspace):
    job = plan(workspace)["jobs"][0]
    make_result(job)
    expected = record_result(job)
    assert reuse_result(job) == expected
    record = json.loads((Path(job["output_dir"]) / "run_record.json").read_text())
    assert set(record["artifact_sha256"]) == set(ARTIFACTS)


def test_changed_output_or_inputs_cannot_be_reused(workspace):
    job = plan(workspace)["jobs"][0]
    make_result(job)
    record_result(job)
    output = Path(job["output_dir"]) / "per_query.csv"
    output.write_text("tampered\n")
    with pytest.raises(ManifestError, match="modified evaluation artifacts"):
        reuse_result(job)
    make_result(job)
    record_result(job)
    changed = plan(workspace, max_identities=1)["jobs"][0]
    with pytest.raises(ManifestError, match="stale evaluation inputs"):
        reuse_result(changed)


def test_missing_output_and_provenance_mismatch_rejected(workspace):
    job = plan(workspace)["jobs"][0]
    make_result(job)
    output = Path(job["output_dir"])
    (output / "nearest_negative_pairs.csv").unlink()
    with pytest.raises(ManifestError, match="incomplete evaluation output"):
        record_result(job)
    make_result(job)
    summary = json.loads((output / "summary.json").read_text())
    summary["checkpoint_sha256"] = "wrong checkpoint"
    (output / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ManifestError, match="provenance mismatch"):
        record_result(job)


def load_cli(workspace, monkeypatch, *extra):
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_uavdt_reid_checkpoints.py"
    spec = importlib.util.spec_from_file_location("checkpoint_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", workspace)
    monkeypatch.setitem(module.EXPECTED["val"], "registry_sha256", sha256_file(workspace / "registry.csv"))
    monkeypatch.setattr(sys, "argv", [str(script), "--checkpoint-dir", str(workspace / "checkpoints"),
                       "--config", str(workspace / "configs/model.yml"), "--registry", str(workspace / "registry.csv"),
                       "--output-dir", str(workspace / "output"), "--device", "cpu", "--batch-size", "8",
                       "--num-workers", "0", "--verify-crops", *extra])
    return module


def test_cli_plan_has_no_inference_or_writes(workspace, monkeypatch, capsys):
    module = load_cli(workspace, monkeypatch, "--plan-only")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("inference was launched"))
    assert module.main() == 0
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 6
    assert not (workspace / "output").exists()


def test_cli_sweep_resume_and_failed_overwrite(workspace, monkeypatch):
    jobs = plan(workspace)["jobs"]
    called = []

    def fake_inference(command, **kwargs):
        directory = command[command.index("--output-dir") + 1]
        job = next(job for job in jobs if job["output_dir"] == directory)
        called.append(job["checkpoint"])
        make_result(job)

    module = load_cli(workspace, monkeypatch)
    monkeypatch.setattr(module.subprocess, "run", fake_inference)
    assert module.main() == 0
    assert called == list(CHECKPOINT_NAMES)
    comparison = json.loads((workspace / "output/comparison.json").read_text())
    assert comparison["primary_checkpoint"] == PRIMARY_CHECKPOINT
    assert len(comparison["checkpoints"]) == 6
    module = load_cli(workspace, monkeypatch, "--resume")
    called.clear()
    assert module.main() == 0
    assert not called
    module = load_cli(workspace, monkeypatch, "--overwrite")

    def fail_inference(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", fail_inference)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not (workspace / "output/comparison.json").exists()
