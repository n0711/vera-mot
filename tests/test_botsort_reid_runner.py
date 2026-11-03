"""Lightweight runner unit tests with an injected mock upstream tracker."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vera_mot.reid_appearance import HSVHistogramEmbedder


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_botsort_reid_detections",
    ROOT / "scripts/run_botsort_reid_detections.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_detection_csv_to_mot_runner_is_data_free_and_not_benchmark(tmp_path, monkeypatch):
    demo_spec = importlib.util.spec_from_file_location(
        "make_reid_demo", ROOT / "scripts/make_reid_demo.py"
    )
    demo = importlib.util.module_from_spec(demo_spec)
    demo_spec.loader.exec_module(demo)
    video, detections = demo.create_demo(tmp_path)

    class FakeTrack:
        is_activated = True

        def __init__(self, box, score, tid):
            self.tlwh = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])
            self.score = score
            self.track_id = tid

    class FakeBoTSORT:
        def update(self, rows, frame):
            return [FakeTrack(row[:4], float(row[4]), i + 1) for i, row in enumerate(rows)]

    def factory(project_root, *, embedder, frame_rate, name, cmc_method):
        assert frame_rate == 6
        assert cmc_method == "none"
        return FakeBoTSORT(), SimpleNamespace(calls=8, features_extracted=14)

    monkeypatch.setattr(RUNNER, "create_tracker", factory)
    out = tmp_path / "mot.txt"
    summary = RUNNER.run(video, detections, out, embedder=HSVHistogramEmbedder())
    assert summary["scope"] == "unvalidated_pinned_botsort_appearance_tracking"
    assert summary["frame_count"] == 8
    assert summary["reported_track_observations"] == 14
    assert summary["extracted_embeddings"] == 14
    assert len(out.read_text().splitlines()) == 14
    assert summary["input_video_sha256"] and summary["input_detection_sha256"]
