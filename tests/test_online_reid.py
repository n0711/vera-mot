"""CPU-only tests: no datasets, detector model, or pretrained weights required."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from vera_mot.online_reid import OnlineReID
from vera_mot.reid_appearance import HSVHistogramEmbedder, ResNet18Embedder


def _update(tracker, index, rows, features):
    boxes = np.array(rows, dtype=float).reshape(-1, 4)
    vectors = np.array(features, dtype=float).reshape(-1, 2)
    return tracker.update(index, boxes, vectors, np.full(len(boxes), .95))


def test_stable_identity_across_occlusion():
    tracker = OnlineReID(max_age=3)
    first = _update(tracker, 1, [[0, 0, 10, 10]], [[1, 0]])
    assert first[0].track_id == 1 and not first[0].matched_existing
    assert not _update(tracker, 2, [], [])
    assert not _update(tracker, 3, [], [])
    resumed = _update(tracker, 4, [[1, 0, 11, 10]], [[1, 0]])
    assert resumed[0].track_id == 1 and resumed[0].matched_existing
    after_expiry = _update(tracker, 9, [[2, 0, 12, 10]], [[1, 0]])
    assert after_expiry[0].track_id == 2


def test_embedding_recovers_swap_and_assigns_each_track_once():
    tracker = OnlineReID()
    initial = _update(tracker, 1, [[0, 0, 10, 10], [20, 0, 30, 10]], [[1, 0], [0, 1]])
    assert [x.track_id for x in initial] == [1, 2]
    swapped = _update(tracker, 2, [[20, 0, 30, 10], [0, 0, 10, 10]], [[1, 0], [0, 1]])
    assert [x.track_id for x in swapped] == [1, 2]
    assert all(x.matched_existing for x in swapped)


def test_displacement_gate_prevents_unrealistic_remote_match():
    tracker = OnlineReID(max_displacement=2)
    _update(tracker, 1, [[0, 0, 10, 10]], [[1, 0]])
    far = _update(tracker, 2, [[1000, 0, 1010, 10]], [[1, 0]])
    assert far[0].track_id == 2


def test_empty_and_filtered_frames_expire_and_reject_invalid_inputs():
    tracker = OnlineReID(max_age=1)
    _update(tracker, 1, [[0, 0, 10, 10]], [[1, 0]])
    assert tracker.update(2, np.array([[0, 0, 10, 10.]]),
                          np.array([[float('nan'), 0]]), np.array([.1])) == []
    assert tracker.update(3, np.empty((0, 4)), np.empty((0, 2)),
                          np.empty(0)) == []
    assert tracker.active_ids == ()
    with pytest.raises(ValueError, match="strictly increasing"):
        _update(tracker, 3, [], [])
    with pytest.raises(ValueError, match="positive"):
        _update(tracker, 4, [[0, 0, 0, 10]], [[1, 0]])
    with pytest.raises(ValueError, match="zero"):
        _update(tracker, 4, [[0, 0, 10, 10]], [[0, 0]])


def test_feature_dimensions_cannot_change_and_inputs_are_immutable():
    tracker = OnlineReID()
    boxes = np.array([[0, 0, 20, 20]], dtype=float)
    feats = np.array([[1., 0.]])
    scores = np.array([.8])
    tracker.update(1, boxes, feats, scores)
    assert np.array_equal(boxes, [[0, 0, 20, 20]])
    assert np.array_equal(feats, [[1., 0.]])
    assert np.array_equal(scores, [.8])
    with pytest.raises(ValueError, match="dimension"):
        tracker.update(2, boxes, np.array([[1., 0, 0]]), scores)


def test_hsv_embeds_finite_normalized_color_patches():
    frame = np.zeros((60, 80, 3), np.uint8)
    frame[:, :40] = [0, 0, 255]  # red in BGR
    frame[:, 40:] = [255, 0, 0]  # blue in BGR
    enc = HSVHistogramEmbedder()
    features = enc.encode(frame, np.array([[0, 0, 39, 60], [41, 0, 80, 60]]))
    assert features.shape == (2, 640)
    assert np.all(np.isfinite(features))
    assert np.allclose(np.linalg.norm(features, axis=1), 1)
    assert float(features[0] @ features[1]) < .5
    assert enc.encode(frame, np.empty((0, 4))).shape == (0, 640)
    with pytest.raises(ValueError, match="smaller"):
        enc.encode(frame, np.array([[0, 0, 0.1, 0.1]]))


def test_neural_encoder_fails_closed_when_weights_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="local ResNet18 weights"):
        ResNet18Embedder(tmp_path / "missing.pth")


def test_end_to_end_synthetic_video_and_detection_csv(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_reid_detections.py"
    spec = importlib.util.spec_from_file_location("run_reid_detections", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video = tmp_path / "synthetic.mp4"
    csv = tmp_path / "detections.csv"
    output = tmp_path / "result.txt"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (128, 64))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer unavailable")
    for index in range(4):
        image = np.zeros((64, 128, 3), np.uint8)
        if index != 1:
            image[10:30, 10 + index:30 + index] = [0, 0, 255]
        writer.write(image)
    writer.release()
    csv.write_text("frame,x1,y1,x2,y2,score\n"
                   "1,10,10,30,30,0.9\n"
                   "3,12,10,32,30,0.9\n"
                   "4,13,10,33,30,0.9\n", encoding="utf-8")
    summary = module.run(video, csv, output, embedder=HSVHistogramEmbedder(),
                         min_similarity=.7, max_age=5)
    ids = [int(line.split(',')[1]) for line in output.read_text().splitlines()]
    assert ids == [1, 1, 1]
    assert summary["scope"] == "appearance_association_without_ground_truth"
    assert summary["observations"] == 3
    assert summary["matched_existing"] == 2
    assert len(output.read_text().splitlines()) == 3


def test_csv_rejects_malformed_and_out_of_range_frames(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_reid_detections.py"
    spec = importlib.util.spec_from_file_location("run_reid_detections", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    file = tmp_path / "detections.csv"
    file.write_text("frame,x1,y1,x2,y2,score\nfoo,0,0,10,10,0.9\n")
    with pytest.raises(ValueError, match="CSV line 2"):
        module.read_detections(file)


def test_complete_generated_demo_preserves_two_ids_after_gap(tmp_path):
    demo_path = Path(__file__).resolve().parents[1] / "scripts/make_reid_demo.py"
    inference_path = Path(__file__).resolve().parents[1] / "scripts/run_reid_detections.py"
    one = importlib.util.spec_from_file_location("make_reid_demo", demo_path)
    demo = importlib.util.module_from_spec(one)
    one.loader.exec_module(demo)
    two = importlib.util.spec_from_file_location("run_reid_detections", inference_path)
    runner = importlib.util.module_from_spec(two)
    two.loader.exec_module(runner)
    video, csv = demo.create_demo(tmp_path)
    report = runner.run(video, csv, tmp_path / "out.txt", embedder=HSVHistogramEmbedder(),
                        min_similarity=.75, max_age=30)
    assert report["frame_count"] == 8
    assert report["observations"] == 14
    assert report["unique_ids"] == 2
    tracks = {}
    for row in (tmp_path / "out.txt").read_text().splitlines():
        values = row.split(",")
        tracks.setdefault(int(values[1]), []).append(int(values[0]))
    assert sorted(tracks.values(), key=len) == [[1, 2, 3, 6, 7, 8], list(range(1, 9))]
