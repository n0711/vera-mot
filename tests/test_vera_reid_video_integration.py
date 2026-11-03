"""End-to-end synthetic video smoke for gated pinned BoT-SORT, not MOT metrics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vera_mot.reid_appearance import HSVHistogramEmbedder
from vera_mot.reid_reliability import CropQualityGate


ROOT = Path(__file__).resolve().parents[1]


def _load(filename: str, module: str):
    spec = importlib.util.spec_from_file_location(module, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.integration
def test_gated_pipeline_on_generated_synthetic_video(tmp_path):
    demo = _load("make_reid_demo.py", "make_reid_demo_for_vera")
    runner = _load("run_botsort_reid_detections.py", "run_botsort_for_vera")
    video, csv = demo.create_demo(tmp_path)
    plain_output = tmp_path / "without_gate.txt"
    gated_output = tmp_path / "with_gate.txt"

    ungated = runner.run(
        video, csv, plain_output, embedder=HSVHistogramEmbedder()
    )
    gate = CropQualityGate()
    gated = runner.run(
        video, csv, gated_output, embedder=HSVHistogramEmbedder(),
        quality_gate=gate
    )

    assert ungated["frame_count"] == gated["frame_count"] == 8
    assert ungated["input_video_sha256"] == gated["input_video_sha256"]
    assert ungated["input_detection_sha256"] == gated["input_detection_sha256"]
    assert ungated["quality_gate"] is None
    assert gated["mode"] == "vera_quality_gated_reid"
    assert gated["quality_gate"]["assessed"] == 14
    assert gated["quality_gate"]["appearance_blocked"] > 0
    assert ungated["reported_track_observations"] > 0
    assert gated["reported_track_observations"] > 0

    for output in (plain_output, gated_output):
        assert output.is_file()
        for line in output.read_text(encoding="utf-8").splitlines():
            fields = line.split(",")
            assert len(fields) == 10
            assert 1 <= int(fields[0]) <= 8
            assert int(fields[1]) > 0
