"""Synthetic 9-to-10-column fixture against the exact pinned TrackEval parser."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vera_mot.uavdt_gt_convert import convert_text

ROOT = Path(__file__).resolve().parents[1]
PINNED_SOURCE = ROOT / "baselines/TrackEval/trackeval/datasets/mot_challenge_2d_box.py"


@pytest.mark.integration
def test_generated_single_vehicle_gt_is_read_by_pinned_trackeval(tmp_path, monkeypatch):
    assert PINNED_SOURCE.is_file(), "pinned TrackEval submodule must be initialized"
    # TrackEval pin predates removal of NumPy's deprecated type aliases.
    monkeypatch.setattr(np, "float", float, raising=False)
    monkeypatch.setattr(np, "int", int, raising=False)
    monkeypatch.syspath_prepend(str(ROOT / "baselines/TrackEval"))
    import trackeval

    source = (
        "3,2,15,25,11,18,3,4,3\n"
        "1,2,-2,5,12,16,1,1,1\n"
        "2,5,33,21,10,13,2,2,2\n"
    )
    converted, summary = convert_text(source, frame_count=3)
    assert summary["record_count"] == 3
    gt_root = tmp_path / "gt"
    tracker_root = tmp_path / "trackers"
    (tracker_root / "synthetic" / "data").mkdir(parents=True)
    gt_root.mkdir()
    (gt_root / "M0201_gt.txt").write_text(converted)
    (tracker_root / "synthetic" / "data/M0201.txt").write_text(
        "1,10,-2,5,12,16,0.9,-1,-1,-1\n"
        "2,10,33,21,10,13,0.9,-1,-1,-1\n"
        "3,10,15,25,11,18,0.9,-1,-1,-1\n"
    )

    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(tracker_root),
        "OUTPUT_FOLDER": str(tmp_path / "metrics"),
        "TRACKERS_TO_EVAL": ["synthetic"],
        "CLASSES_TO_EVAL": ["pedestrian"],  # API class-name alias, not pedestrian GT.
        "BENCHMARK": "UAVDT_REID",
        "SPLIT_TO_EVAL": "val",
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "SEQ_INFO": {"M0201": 3},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}_gt.txt",
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    })
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
