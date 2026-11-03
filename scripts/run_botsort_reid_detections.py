#!/usr/bin/env python3
"""Run the pinned BoT-SORT tracker on external detections, with optional ReID.

No YOLOX checkpoint or UAVDT dataset is required. Appearance features are
*not* learned vehicle-ReID features unless a vehicle-trained encoder is supplied
and separately validated. Never write into frozen baseline output paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_reid_detections import read_detections, sha256_stream  # noqa: E402
from vera_mot.botsort_reid_bridge import (  # noqa: E402
    create_tracker, detections_to_botsort,
)
from vera_mot.reid_appearance import HSVHistogramEmbedder, ResNet18Embedder  # noqa: E402


def run(
    video: Path,
    detection_csv: Path,
    output: Path,
    *,
    embedder=None,
    cmc_method: str = "none",
    render: Path | None = None,
) -> dict:
    if not video.is_file():
        raise FileNotFoundError(f"video missing: {video}")
    frames_to_boxes = read_detections(detection_csv)
    if any(frame < 1 for frame in frames_to_boxes):
        raise ValueError("frame indices must be positive")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    writer = None
    frame_number = 0
    lines: list[str] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        tracker, encoder = create_tracker(
            ROOT, embedder=embedder, frame_rate=max(1, int(round(fps))),
            name=video.stem, cmc_method=cmc_method,
        )
        if render is not None:
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            render.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(render), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise ValueError(f"cannot write render: {render}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            rows = frames_to_boxes.get(frame_number, np.empty((0, 5), dtype=np.float64))
            detections = detections_to_botsort(rows)
            tracks = tracker.update(detections, frame)
            for track in tracks:
                if not track.is_activated:
                    continue
                x, y, width, height = map(float, track.tlwh)
                if width <= 0 or height <= 0 or not np.isfinite([x, y, width, height]).all():
                    raise ValueError("nonfinite/invalid tracker bounding box")
                track_id = int(track.track_id)
                score = float(track.score)
                if not np.isfinite(score):
                    raise ValueError("nonfinite tracking score")
                lines.append(f"{frame_number},{track_id},{x:.2f},{y:.2f},"
                             f"{width:.2f},{height:.2f},{score:.4f},-1,-1,-1\n")
                if writer is not None:
                    p1 = int(x), int(y)
                    p2 = int(x + width), int(y + height)
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            if writer is not None:
                writer.write(frame)
        if not frame_number:
            raise ValueError("video contains no frames")
        if any(frame > frame_number for frame in frames_to_boxes):
            raise ValueError("detection frame index is beyond video length")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    summary = {
        "scope": "unvalidated_pinned_botsort_appearance_tracking",
        "tracker": "pinned_BoT-SORT",
        "mode": "appearance_reid" if embedder is not None else "no_reid",
        "embedding": embedder.name if embedder is not None else None,
        "encoder_calls": encoder.calls if encoder is not None else 0,
        "extracted_embeddings": encoder.features_extracted if encoder is not None else 0,
        "frame_count": frame_number,
        "reported_track_observations": len(lines),
        "tentative_unique_ids": len({line.split(",")[1] for line in lines}),
        "input_video_sha256": sha256_stream(video),
        "input_detection_sha256": sha256_stream(detection_csv),
        "embedding_weights_sha256": getattr(embedder, "weights_sha256", None),
        "cmc_method": cmc_method,
        "note": "not ground-truth ReID or HOTA/IDF1 validation",
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", type=Path)
    p.add_argument("--mode", choices=("no-reid", "hsv-reid", "resnet18-reid"),
                   default="hsv-reid")
    p.add_argument("--cmc-method", choices=("none", "sparseOptFlow", "orb", "ecc"),
                   default="none")
    p.add_argument("--weights", type=Path)
    p.add_argument("--weights-sha256")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    if args.mode == "no-reid":
        if args.weights or args.weights_sha256:
            p.error("weights are only accepted for resnet18-reid")
        embedder = None
    elif args.mode == "hsv-reid":
        if args.weights or args.weights_sha256:
            p.error("weights are only accepted for resnet18-reid")
        embedder = HSVHistogramEmbedder()
    else:
        if args.weights is None:
            p.error("--weights required for resnet18-reid")
        embedder = ResNet18Embedder(args.weights, args.device, args.weights_sha256)
    summary = run(args.video, args.detections, args.output, embedder=embedder,
                  cmc_method=args.cmc_method, render=args.render)
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
