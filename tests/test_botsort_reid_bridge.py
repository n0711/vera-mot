"""CPU-only contract tests for injected appearance + pinned BoT-SORT interface."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vera_mot.botsort_reid_bridge import (
    BoTSORTAppearanceEncoder, attach_project_encoder,
    detections_to_botsort, make_tracker_args,
)
from vera_mot.reid_appearance import HSVHistogramEmbedder


def _frame():
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    frame[10:35, 10:40] = (0, 0, 255)
    frame[10:35, 50:80] = (255, 0, 0)
    return frame


def test_adapter_produces_normalized_features_in_input_order():
    frame = _frame()
    boxes = np.array([[10, 10, 40, 35], [50, 10, 80, 35]], dtype=np.float32)
    original = boxes.copy()
    adapter = BoTSORTAppearanceEncoder(HSVHistogramEmbedder())
    features = adapter.inference(frame, boxes)
    assert features.shape == (2, 640)
    assert features.dtype == np.float32
    assert np.allclose(np.linalg.norm(features, axis=1), 1)
    assert np.array_equal(boxes, original)
    assert float(features[0] @ features[1]) < 0.5
    assert adapter.calls == 1
    assert adapter.features_extracted == 2


def test_adapter_empty_detections_are_supported():
    adapter = BoTSORTAppearanceEncoder(HSVHistogramEmbedder())
    assert adapter.inference(_frame(), []).shape == (0, 640)
    assert adapter.inference(_frame(), np.empty((0, 4))).shape == (0, 640)
    assert adapter.calls == 2
    assert adapter.features_extracted == 0


@pytest.mark.parametrize("boxes", [
    [[10, 10, 10, 30]],
    [[10, 10, float("nan"), 30]],
    [[10, 10, 40, 30, .9]],
])
def test_adapter_fails_closed_for_bad_boxes(boxes):
    adapter = BoTSORTAppearanceEncoder(HSVHistogramEmbedder())
    with pytest.raises(ValueError):
        adapter.inference(_frame(), boxes)
    assert adapter.calls == 0


def test_adapter_rejects_nonfinite_or_wrong_dimension_embeddings():
    class InvalidEmbedder:
        dimension = 3

        def encode(self, frame, boxes):
            return np.array([[1.0, np.nan, 0.0]])

    adapter = BoTSORTAppearanceEncoder(InvalidEmbedder())
    with pytest.raises(ValueError, match="non-finite"):
        adapter.inference(_frame(), [[0, 0, 15, 15]])

    class WrongDimEmbedder:
        dimension = 5

        def encode(self, frame, boxes):
            return np.ones((1, 4))

    with pytest.raises(ValueError, match="invalid shape"):
        BoTSORTAppearanceEncoder(WrongDimEmbedder()).inference(
            _frame(), [[0, 0, 15, 15]]
        )


def test_submodule_free_attach_then_enable_reid_without_model_load():
    tracker = SimpleNamespace(args=make_tracker_args(), tracked_stracks=[])
    assert not tracker.args.with_reid
    assert not hasattr(tracker, "encoder")
    adapter = attach_project_encoder(tracker, HSVHistogramEmbedder())
    assert tracker.args.with_reid and tracker.encoder is adapter
    with pytest.raises(ValueError, match="already"):
        attach_project_encoder(tracker, HSVHistogramEmbedder())


def test_tracker_args_preserve_offline_mode_with_baseline_values():
    args = make_tracker_args(cmc_method="sparseOptFlow")
    assert args.track_high_thresh == .35
    assert args.match_thresh == .8
    assert args.track_buffer == 30
    assert args.proximity_thresh == .5
    assert args.appearance_thresh == .25
    assert args.with_reid is False
    assert args.cmc_method == "sparseOptFlow"
    with pytest.raises(ValueError, match="track_low"):
        make_tracker_args(track_high_thresh=.1)
    with pytest.raises(ValueError, match="GMC"):
        make_tracker_args(cmc_method="unknown")


def test_botsort_detections_validation_and_input_immutability():
    inp = np.array([[10., 10., 40., 40., .95]])
    out = detections_to_botsort(inp)
    assert out.shape == (1, 5)
    assert out.dtype == np.float32
    out[0, 4] = .2
    assert inp[0, 4] == .95
    assert detections_to_botsort(np.empty((0, 5))).shape == (0, 5)
    with pytest.raises(ValueError, match="invalid"):
        detections_to_botsort(np.array([[0., 0., 0., 10., .5]]))
    with pytest.raises(ValueError, match="non-finite"):
        detections_to_botsort(np.array([[0., 0., 10., 10., float("nan")]]))


def test_botsort_encoder_does_not_construct_untrusted_fastreid():
    tracker = SimpleNamespace(args=make_tracker_args(), tracked_stracks=[])
    class FakeEncoder:
        dimension = 2

        def encode(self, frame, boxes):
            return np.tile([3., 4.], (len(boxes), 1))

    adapter = attach_project_encoder(tracker, FakeEncoder())
    features = adapter.inference(_frame(), [[10, 10, 40, 35]])
    assert np.allclose(features[0], [.6, .8])
    assert adapter.features_extracted == 1


@pytest.mark.integration
def test_real_pinned_botsort_reid_path_using_synthetic_frames():
    """Requires PyTorch, full pinned submodule environment, and local patches."""
    from pathlib import Path

    from vera_mot.botsort_reid_bridge import create_tracker

    root = Path(__file__).resolve().parents[1]
    tracker, encoder = create_tracker(
        root, embedder=HSVHistogramEmbedder(), frame_rate=30, cmc_method="none"
    )
    for index in range(1, 5):
        frame = _frame()
        if index == 3:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = np.array([[10, 10, 40, 35, .95], [50, 10, 80, 35, .95]])
        tracks = tracker.update(detections_to_botsort(dets), frame)
        assert isinstance(tracks, list)
    assert encoder.calls == 4
    assert encoder.features_extracted == 6
