#!/usr/bin/env python3
"""Generate an explicitly synthetic vehicle-like ReID smoke-test clip and boxes."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def create_demo(directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / "synthetic_vehicle_demo.mp4"
    detections = directory / "synthetic_vehicle_detections.csv"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 6, (192, 96))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create synthetic demo MP4")
    records = []
    try:
        for frame in range(1, 9):
            image = np.full((96, 192, 3), 35, dtype=np.uint8)
            # Synthetic red and blue rectangles. The red one disappears briefly.
            for identity, color, x, y, visible in (
                (1, (0, 0, 245), 10 + 4 * frame, 15, frame not in (4, 5)),
                (2, (240, 35, 0), 152 - 4 * frame, 58, True),
            ):
                if not visible:
                    continue
                cv2.rectangle(image, (x, y), (x + 25, y + 16), color, -1)
                records.append((frame, x, y, x + 25, y + 16, 0.95))
            writer.write(image)
    finally:
        writer.release()
    with detections.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.writer(handle, lineterminator="\n")
        csv_writer.writerow(("frame", "x1", "y1", "x2", "y2", "score"))
        csv_writer.writerows(records)
    return video, detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/reid_online/synthetic_demo"))
    args = parser.parse_args()
    video, detections = create_demo(args.output_dir)
    print("Synthetic sample (not a real tracker benchmark):")
    print(f"Video: {video}")
    print(f"Detections: {detections}")


if __name__ == "__main__":
    main()
