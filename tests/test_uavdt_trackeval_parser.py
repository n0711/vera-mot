"""Synthetic 9-to-10-column fixture against the exact pinned TrackEval parser."""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from vera_mot.uavdt_gt_convert import convert_text

ROOT = Path(__file__).resolve().parents[1]
PINNED_SOURCE = ROOT / "baselines/TrackEval/trackeval/datasets/mot_challenge_2d_box.py"


@pytest.fixture
def pinned_trackeval(monkeypatch):
    assert PINNED_SOURCE.is_file(), "pinned TrackEval submodule must be initialized"
    # TrackEval pin predates removal of NumPy's deprecated type aliases.
    monkeypatch.setattr(np, "float", float, raising=False)
    monkeypatch.setattr(np, "int", int, raising=False)
    monkeypatch.syspath_prepend(str(ROOT / "baselines/TrackEval"))
    import trackeval

    return trackeval


def make_dataset(tmp_path, trackeval, source, tracker_rows, frames):
    converted, _ = convert_text(source, frame_count=frames)
    gt_root = tmp_path / "gt"
    tracker_root = tmp_path / "trackers"
    (tracker_root / "synthetic" / "data").mkdir(parents=True)
    gt_root.mkdir()
    (gt_root / "M0201_gt.txt").write_text(converted)
    (tracker_root / "synthetic" / "data/M0201.txt").write_text(tracker_rows)
    return trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(tracker_root),
        "OUTPUT_FOLDER": str(tmp_path / "metrics"),
        "TRACKERS_TO_EVAL": ["synthetic"],
        "CLASSES_TO_EVAL": ["pedestrian"],  # API class-name alias, not pedestrian GT.
        "BENCHMARK": "UAVDT_REID",
        "SPLIT_TO_EVAL": "val",
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "SEQ_INFO": {"M0201": frames},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}_gt.txt",
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    })


@pytest.mark.integration
def test_generated_single_vehicle_gt_is_read_by_pinned_trackeval(tmp_path, pinned_trackeval):
    source = (
        "3,2,15,25,11,18,3,4,3\n"
        "1,2,-2,5,12,16,1,1,1\n"
        "2,5,33,21,10,13,2,2,2\n"
    )
    tracker_rows = (
        "1,10,-2,5,12,16,0.9,-1,-1,-1\n"
        "2,10,33,21,10,13,0.9,-1,-1,-1\n"
        "3,10,15,25,11,18,0.9,-1,-1,-1\n"
    )

    dataset = make_dataset(tmp_path, pinned_trackeval, source, tracker_rows, frames=3)
    gt_data = dataset._load_raw_file("synthetic", "M0201", is_gt=True)
    assert gt_data["num_timesteps"] == 3
    assert [row[0] for row in gt_data["gt_ids"]] == [2, 5, 2]
    assert all(row[0] == 1 for row in gt_data["gt_classes"])
    assert all(row["zero_marked"][0] == 1 for row in gt_data["gt_extras"])
    # Explicitly verify parser preserves image box geometry in native xywh.
    assert gt_data["gt_dets"][0].tolist() == [[-2.0, 5.0, 12.0, 16.0]]
    assert gt_data["gt_dets"][2].tolist() == [[15.0, 25.0, 11.0, 18.0]]
    data = dataset.get_preprocessed_seq_data(
        dataset.get_raw_seq_data("synthetic", "M0201"), "pedestrian"
    )
    assert data["num_gt_dets"] == 3
    assert data["num_tracker_dets"] == 3
    # No metric is evaluated: a parser smoke test does not establish accuracy.


@pytest.mark.integration
@pytest.mark.parametrize(
    "ids, expected_hota, expected_assa, expected_idf1, expected_switches",
    [
        pytest.param([20, 20, 20, 20], 1, 1, 1, 0, id="perfect"),
        pytest.param([20, 20, 31, 31], sqrt(0.5), 0.5, 0.5, 1, id="identity-switch"),
        pytest.param([20, None, 20, 20], 0.75, 0.75, 6 / 7, 0, id="missed-frame"),
    ],
)
def test_synthetic_identity_scores(tmp_path, pinned_trackeval, ids, expected_hota,
                                   expected_assa, expected_idf1, expected_switches):
    source = "".join(f"{frame},7,10,20,8,6,1,1,2\n" for frame in range(1, 5))
    tracker_rows = "".join(
        f"{frame},{identity},10,20,8,6,0.9,-1,-1,-1\n"
        for frame, identity in enumerate(ids, start=1) if identity is not None
    )
    dataset = make_dataset(tmp_path, pinned_trackeval, source, tracker_rows, frames=4)
    data = dataset.get_preprocessed_seq_data(
        dataset.get_raw_seq_data("synthetic", "M0201"), "pedestrian"
    )
    metrics = pinned_trackeval.metrics
    config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    hota = metrics.HOTA().eval_sequence(data)
    # Every present box has IoU 1, so each of the 19 HOTA thresholds must agree.
    np.testing.assert_allclose(hota["HOTA"], np.full(19, expected_hota))
    np.testing.assert_allclose(hota["AssA"], np.full(19, expected_assa))
    assert metrics.Identity(config).eval_sequence(data)["IDF1"] == pytest.approx(expected_idf1)
    assert metrics.CLEAR(config).eval_sequence(data)["IDSW"] == expected_switches
