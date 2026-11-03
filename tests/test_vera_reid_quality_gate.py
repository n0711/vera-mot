"""Data-free VERA crop-quality tests; no claimed identity-level reliability."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vera_mot.reid_reliability import CropQualityGate, QualityGateConfig
from vera_mot.botsort_reid_bridge import attach_quality_gate


def _quality_frame() -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((80, 140, 3), 70, dtype=np.uint8)
    # Deterministic high-frequency texture and a featureless patch.
    rng = np.random.default_rng(42)
    texture = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    frame[10:50, 10:50] = texture
    frame[10:50, 80:120] = 120
    return frame, np.array([[10., 10., 50., 50.], [80., 10., 120., 50.]])


def test_texture_passes_and_featureless_patch_fails():
    frame, boxes = _quality_frame()
    gate = CropQualityGate()
    decisions = gate.allowed(frame, boxes)
    assert decisions.dtype == np.bool_
    assert decisions.tolist() == [True, False]
    first, second = gate.last_observations
    assert first.score >= .5
    assert first.area_score == 1.
    assert first.in_frame_fraction == 1.
    assert second.score == 0.
    assert second.focus_score == 0.
    summary = gate.summary()
    assert summary["assessed"] == 2
    assert summary["appearance_allowed"] == 1
    assert summary["appearance_blocked"] == 1
    assert summary["scope"] == "unvalidated_image_quality_proxy_not_identity_reliability"


def test_low_resolution_and_truncated_crop_are_conservatively_blocked():
    frame, boxes = _quality_frame()
    gate = CropQualityGate()
    tiny = np.array([[10., 10., 14., 14.]])
    quality = gate.score_detections(frame, tiny)[0]
    assert quality.area_score <= 16 / 1024
    assert not quality.appearance_allowed
    boundary = np.array([[-30., 10., 50., 50.]])
    quality2 = gate.score_detections(frame, boundary)[0]
    assert quality2.in_frame_fraction <= .625
    strict = CropQualityGate(QualityGateConfig(threshold=.8))
    assert not strict.allowed(frame, boundary)[0]


def test_counters_are_unchanged_after_validation_errors():
    frame, boxes = _quality_frame()
    gate = CropQualityGate()
    gate.allowed(frame, boxes)
    before = gate.summary()
    invalid = [
        (frame.astype(float), boxes, "uint8"),
        (frame, np.array([[1., 2., 1., 5.]]), "positive"),
        (frame, np.array([[1., 2., np.nan, 5.]]), "nonfinite"),
        (frame, np.array([[1., 2., 3.]]), "shape"),
    ]
    for img, bb, err in invalid:
        with pytest.raises(ValueError, match=err):
            gate.allowed(img, bb)
        assert gate.summary() == before
    assert len(gate.last_observations) == 2


def test_empty_detections_and_same_input_are_deterministic():
    frame, boxes = _quality_frame()
    gate = CropQualityGate()
    result = gate.allowed(frame, np.empty((0, 4)))
    assert result.shape == (0,) and result.dtype == np.bool_
    assert gate.summary()["assessed"] == 0
    a = gate.score_detections(frame, boxes)
    b = gate.score_detections(frame, boxes.copy())
    assert a == b
    assert gate.summary()["assessed"] == 0



def test_invalid_config_fails_closed():
    with pytest.raises(ValueError, match="threshold"):
        QualityGateConfig(threshold=1.1)
    with pytest.raises(ValueError, match="threshold"):
        QualityGateConfig(threshold=float("nan"))
    with pytest.raises(ValueError, match="reference_area"):
        QualityGateConfig(reference_area=0)
    with pytest.raises(ValueError, match="reference_laplacian"):
        QualityGateConfig(reference_laplacian_variance=float("inf"))


def test_fail_if_upstream_quality_patch_not_installed(tmp_path):
    root = tmp_path / "baselines/BoT-SORT/tracker"
    root.mkdir(parents=True)
    (root / "bot_sort.py").write_text("class BoTSORT: pass\n")
    (root / "matching.py").write_text("def embedding_distance(): pass\n")
    tracker = SimpleNamespace(args=SimpleNamespace(with_reid=True))
    with pytest.raises(RuntimeError, match="apply_botsort_vera_patch"):
        attach_quality_gate(tracker, CropQualityGate(), tmp_path)
    assert not hasattr(tracker, "appearance_policy")


@pytest.mark.integration
def test_real_botsort_quality_gate_protects_identity_memory():
    """Execute the actual pinned tracker; no real vehicles or GT used."""
    from vera_mot.botsort_reid_bridge import create_tracker, detections_to_botsort

    root = Path(__file__).resolve().parents[1]
    frame, boxes = _quality_frame()
    gate = CropQualityGate()
    class ConstantEmbedder:
        dimension = 2
        name = "test-only-constant-vector"
        def __init__(self):
            self.calls = 0
        def encode(self, image, coords):
            self.calls += 1
            vector = [1., 0.] if self.calls < 3 else [0., 1.]
            return np.tile(vector, (len(coords), 1))

    embedder = ConstantEmbedder()
    tracker, encoder = create_tracker(root, embedder=embedder, cmc_method="none")
    attach_quality_gate(tracker, gate, root)
    row = np.array([[10., 10., 50., 50., .95]], dtype=np.float32)
    # First observation has high detail: initialize a clean appearance memory.
    tracker.update(detections_to_botsort(row), frame)
    assert len(tracker.tracked_stracks) == 1
    original = tracker.tracked_stracks[0].smooth_feat.copy()
    assert len(tracker.tracked_stracks[0].features) == 1

    # Second observation is featureless and spatially coincident. Its match is
    # allowed via IoU only; it MUST NOT update the feature memory.
    blurred = frame.copy()
    blurred[10:50, 10:50] = 128
    tracker.update(detections_to_botsort(row), blurred)
    assert len(tracker.tracked_stracks) == 1
    assert np.allclose(tracker.tracked_stracks[0].smooth_feat, original)
    assert len(tracker.tracked_stracks[0].features) == 1

    # Third observation is high-detail again; quality can restore memory writes.
    tracker.update(detections_to_botsort(row), frame)
    assert len(tracker.tracked_stracks) == 1
    assert len(tracker.tracked_stracks[0].features) == 2
    assert not np.allclose(tracker.tracked_stracks[0].smooth_feat, original)
    assert gate.summary()["assessed"] == 3
    assert gate.summary()["appearance_blocked"] == 1
    assert gate.summary()["appearance_allowed"] == 2
    assert encoder.features_extracted == 3


@pytest.mark.integration
def test_missing_appearance_vectors_do_not_enter_distance_matrix():
    """BoT-SORT's upstream matcher must fall back to IoU for missing features."""
    from vera_mot.botsort_reid_bridge import load_pinned_botsort
    load_pinned_botsort(Path(__file__).resolve().parents[1])
    from tracker import matching
    tracks = [
        SimpleNamespace(smooth_feat=np.array([1., 0.])),
        SimpleNamespace(smooth_feat=None),
    ]
    dets = [
        SimpleNamespace(curr_feat=np.array([1., 0.])),
        SimpleNamespace(curr_feat=None),
    ]
    costs = matching.embedding_distance(tracks, dets)
    assert costs.shape == (2, 2)
    assert costs[0, 0] == pytest.approx(0, abs=1e-8)
    assert np.array_equal(costs[1], np.array([2., 2.]))
    assert np.array_equal(costs[:, 1], np.array([2., 2.]))
    assert matching.embedding_distance([], dets).shape == (0, 2)
