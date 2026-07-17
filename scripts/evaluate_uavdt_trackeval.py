#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKEVAL_ROOT = PROJECT_ROOT / "baselines" / "TrackEval"

# Compatibility for TrackEval with modern NumPy.
if "float" not in np.__dict__:
    np.float = float

if "int" not in np.__dict__:
    np.int = int

sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        default="M0203",
    )

    parser.add_argument(
        "--tracker",
        default="botsort_no_reid",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames_dir = (
        PROJECT_ROOT
        / "datasets/UAVDT/full_mot/raw/UAV-benchmark-M"
        / args.sequence
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
        PROJECT_ROOT
        / "experiments/tracking_baselines"
        / args.tracker
        / "metrics"
    )

    sequence_length = len(list(frames_dir.glob("img*.jpg")))

    if sequence_length == 0:
        raise FileNotFoundError(
            f"No frames found for {args.sequence}"
        )

    gt_file = gt_root / f"{args.sequence}_gt.txt"

    tracker_file = (
        trackers_root
        / args.tracker
        / "uavdt"
        / f"{args.sequence}.txt"
    )

    if not gt_file.is_file():
        raise FileNotFoundError(gt_file)

    if not tracker_file.is_file():
        raise FileNotFoundError(tracker_file)

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
            "TRACKERS_TO_EVAL": [args.tracker],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "UAVDT",
            "SPLIT_TO_EVAL": "test",
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "uavdt",
            "OUTPUT_SUB_FOLDER": args.sequence,
            "SEQ_INFO": {
                args.sequence: sequence_length,
            },
            "GT_LOC_FORMAT": (
                "{gt_folder}/{seq}_gt.txt"
            ),
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