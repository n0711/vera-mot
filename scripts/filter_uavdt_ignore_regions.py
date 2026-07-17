#!/usr/bin/env python3

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_SEQUENCES = [
    "M0203", "M0205", "M0208", "M0209", "M0403",
    "M0601", "M0602", "M0606", "M0701", "M0801",
    "M0802", "M1001", "M1004", "M1007", "M1009",
    "M1101", "M1301", "M1302", "M1303", "M1401",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply UAVDT ignore-region preprocessing."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments/tracking_baselines/"
            "botsort_no_reid/uavdt"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments/tracking_baselines/"
            "botsort_no_reid/uavdt_official"
        ),
    )

    parser.add_argument(
        "--gt-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets/UAVDT/full_mot/toolkit/"
            "UAV-benchmark-MOTD_v1.0/GT"
        ),
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--ignore-ioa",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def read_boxes(path: Path) -> dict[int, list[np.ndarray]]:
    boxes: dict[int, list[np.ndarray]] = defaultdict(list)

    if not path.is_file():
        return boxes

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        values = line.split(",")

        if len(values) < 6:
            raise ValueError(
                f"{path}:{line_number}: expected at least 6 columns"
            )

        frame = int(float(values[0]))
        box = np.asarray(
            [float(value) for value in values[2:6]],
            dtype=np.float64,
        )

        boxes[frame].append(box)

    return boxes


def read_tracker_rows(
    path: Path,
) -> dict[int, list[tuple[str, np.ndarray]]]:
    rows: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        values = line.split(",")

        if len(values) != 10:
            raise ValueError(
                f"{path}:{line_number}: expected 10 columns"
            )

        frame = int(float(values[0]))
        box = np.asarray(
            [float(value) for value in values[2:6]],
            dtype=np.float64,
        )

        rows[frame].append((line, box))

    return rows


def intersection_matrix(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)

    first_x2 = first[:, 0] + np.maximum(first[:, 2], 0)
    first_y2 = first[:, 1] + np.maximum(first[:, 3], 0)

    second_x2 = second[:, 0] + np.maximum(second[:, 2], 0)
    second_y2 = second[:, 1] + np.maximum(second[:, 3], 0)

    left = np.maximum(first[:, None, 0], second[None, :, 0])
    top = np.maximum(first[:, None, 1], second[None, :, 1])
    right = np.minimum(first_x2[:, None], second_x2[None, :])
    bottom = np.minimum(first_y2[:, None], second_y2[None, :])

    return (
        np.maximum(right - left, 0)
        * np.maximum(bottom - top, 0)
    )


def iou_matrix(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    intersection = intersection_matrix(first, second)

    first_area = (
        np.maximum(first[:, 2], 0)
        * np.maximum(first[:, 3], 0)
    )

    second_area = (
        np.maximum(second[:, 2], 0)
        * np.maximum(second[:, 3], 0)
    )

    union = (
        first_area[:, None]
        + second_area[None, :]
        - intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def tracker_ioa_matrix(
    tracker_boxes: np.ndarray,
    ignore_boxes: np.ndarray,
) -> np.ndarray:
    intersection = intersection_matrix(
        tracker_boxes,
        ignore_boxes,
    )

    tracker_area = (
        np.maximum(tracker_boxes[:, 2], 0)
        * np.maximum(tracker_boxes[:, 3], 0)
    )

    return np.divide(
        intersection,
        tracker_area[:, None],
        out=np.zeros_like(intersection),
        where=tracker_area[:, None] > 0,
    )


def filter_sequence(
    tracker_path: Path,
    gt_path: Path,
    ignore_path: Path,
    output_path: Path,
    match_iou: float,
    ignore_ioa: float,
) -> tuple[int, int]:
    tracker_rows = read_tracker_rows(tracker_path)
    gt_boxes = read_boxes(gt_path)
    ignore_boxes = read_boxes(ignore_path)

    kept_lines: list[str] = []
    input_count = 0
    removed_count = 0

    for frame in sorted(tracker_rows):
        frame_rows = tracker_rows[frame]
        input_count += len(frame_rows)

        tracks = np.asarray(
            [box for _, box in frame_rows],
            dtype=np.float64,
        )

        valid_gt = np.asarray(
            gt_boxes.get(frame, []),
            dtype=np.float64,
        ).reshape(-1, 4)

        ignored = np.asarray(
            ignore_boxes.get(frame, []),
            dtype=np.float64,
        ).reshape(-1, 4)

        matched_tracker_indices: set[int] = set()

        if len(valid_gt) and len(tracks):
            similarities = iou_matrix(valid_gt, tracks)
            matching_scores = similarities.copy()
            matching_scores[matching_scores < match_iou] = 0

            gt_indices, tracker_indices = linear_sum_assignment(
                -matching_scores
            )

            for gt_index, tracker_index in zip(
                gt_indices,
                tracker_indices,
            ):
                if matching_scores[gt_index, tracker_index] > 0:
                    matched_tracker_indices.add(int(tracker_index))

        remove_indices: set[int] = set()

        unmatched_indices = [
            index
            for index in range(len(tracks))
            if index not in matched_tracker_indices
        ]

        if unmatched_indices and len(ignored):
            unmatched_boxes = tracks[unmatched_indices]
            overlap = tracker_ioa_matrix(
                unmatched_boxes,
                ignored,
            )

            for local_index, tracker_index in enumerate(
                unmatched_indices
            ):
                if np.any(overlap[local_index] > ignore_ioa):
                    remove_indices.add(tracker_index)

        for index, (line, _) in enumerate(frame_rows):
            if index in remove_indices:
                removed_count += 1
            else:
                kept_lines.append(line + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(kept_lines),
        encoding="utf-8",
    )

    return input_count, removed_count


def main() -> None:
    args = parse_args()

    total_input = 0
    total_removed = 0

    for sequence in TEST_SEQUENCES:
        tracker_path = args.input_root / f"{sequence}.txt"
        gt_path = args.gt_root / f"{sequence}_gt.txt"
        ignore_path = args.gt_root / f"{sequence}_gt_ignore.txt"
        output_path = args.output_root / f"{sequence}.txt"

        if not tracker_path.is_file():
            raise FileNotFoundError(tracker_path)

        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)

        input_count, removed_count = filter_sequence(
            tracker_path=tracker_path,
            gt_path=gt_path,
            ignore_path=ignore_path,
            output_path=output_path,
            match_iou=args.match_iou,
            ignore_ioa=args.ignore_ioa,
        )

        total_input += input_count
        total_removed += removed_count

        print(
            f"{sequence}: input={input_count}, "
            f"ignored={removed_count}, "
            f"kept={input_count - removed_count}"
        )

    print()
    print(f"Total input detections: {total_input}")
    print(f"Removed in ignored regions: {total_removed}")
    print(f"Total kept detections: {total_input - total_removed}")
    print(f"Output: {args.output_root}")


if __name__ == "__main__":
    main()
