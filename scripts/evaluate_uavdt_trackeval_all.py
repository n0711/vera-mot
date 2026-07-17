#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKEVAL_ROOT = PROJECT_ROOT / "baselines" / "TrackEval"

if "float" not in np.__dict__:
    np.float = float

if "int" not in np.__dict__:
    np.int = int

sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval  # noqa: E402


TEST_SEQUENCES = [
    "M0203", "M0205", "M0208", "M0209", "M0403",
    "M0601", "M0602", "M0606", "M0701", "M0801",
    "M0802", "M1001", "M1004", "M1007", "M1009",
    "M1101", "M1301", "M1302", "M1303", "M1401",
]


def main() -> None:
    frames_root = (
        PROJECT_ROOT
        / "datasets/UAVDT/full_mot/raw/UAV-benchmark-M"
    )

    gt_root = (
        PROJECT_ROOT
        / "datasets/UAVDT/full_mot/toolkit/"
        "UAV-benchmark-MOTD_v1.0/GT"
    )

    trackers_root = (
        PROJECT_ROOT
        / "experiments/tracking_baselines"
    )

    output_root = (
        trackers_root
        / "botsort_no_reid/metrics/uavdt_official"
    )

    sequence_info = {}

    for sequence in TEST_SEQUENCES:
        sequence_length = len(
            list((frames_root / sequence).glob("img*.jpg"))
        )

        if sequence_length == 0:
            raise FileNotFoundError(
                f"No frames found for {sequence}"
            )

        tracker_file = (
            trackers_root
            / "botsort_no_reid/uavdt_official"
            / f"{sequence}.txt"
        )

        if not tracker_file.is_file():
            raise FileNotFoundError(tracker_file)

        sequence_info[sequence] = sequence_length

    eval_config = trackeval.Evaluator.get_default_eval_config()

    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )

    dataset_config = (
        trackeval.datasets.MotChallenge2DBox
        .get_default_dataset_config()
    )

    dataset_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "OUTPUT_FOLDER": str(output_root),
            "TRACKERS_TO_EVAL": ["botsort_no_reid"],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "UAVDT",
            "SPLIT_TO_EVAL": "test",
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "uavdt_official",
            "OUTPUT_SUB_FOLDER": "all_test",
            "SEQ_INFO": sequence_info,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}_gt.txt",
            "SKIP_SPLIT_FOL": True,
            "PRINT_CONFIG": False,
        }
    )

    metric_config = {
        "METRICS": [
            "HOTA",
            "CLEAR",
            "Identity",
        ],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }

    metrics = [
        trackeval.metrics.HOTA(metric_config),
        trackeval.metrics.CLEAR(metric_config),
        trackeval.metrics.Identity(metric_config),
    ]

    evaluator = trackeval.Evaluator(eval_config)

    evaluator.evaluate(
        [
            trackeval.datasets.MotChallenge2DBox(
                dataset_config
            )
        ],
        metrics,
    )


if __name__ == "__main__":
    main()
