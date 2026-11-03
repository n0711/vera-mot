#!/usr/bin/env python3
"""Associate video detections using appearance, independent of detector weights.

Input CSV columns: frame,x1,y1,x2,y2,score (one row per vehicle detection).
Results are tracker hypotheses in MOT 10-column format, not ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from vera_mot.online_reid import OnlineReID  # noqa: E402
from vera_mot.reid_appearance import HSVHistogramEmbedder, ResNet18Embedder  # noqa: E402


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_detections(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"detections CSV missing: {path}")
    output = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"frame", "x1", "y1", "x2", "y2", "score"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(f"detections must contain exactly: {', '.join(sorted(expected))}")
        for line, row in enumerate(reader, start=2):
            try:
                raw_frame = row["frame"].strip()
                if not raw_frame.isdecimal():
                    raise ValueError("frame must be integer")
                number = int(raw_frame)
                values = [float(row[k]) for k in ("x1", "y1", "x2", "y2", "score")]
                if number < 1 or not np.all(np.isfinite(values)) or not (
                    values[2] > values[0] and values[3] > values[1] and 0 <= values[4] <= 1
                ):
                    raise ValueError("invalid geometry or score")
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid detection at CSV line {line}: {exc}") from exc
            output[number].append(values)
    return {number: np.asarray(rows, dtype=np.float64) for number, rows in output.items()}


def run(video: Path, detections: Path, output: Path, *,
        embedder, min_similarity: float, max_age: int, annotate: Path | None = None) -> dict:
    if not video.is_file():
        raise FileNotFoundError(f"video missing: {video}")
    frames_to_boxes = read_detections(detections)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    tracker = OnlineReID(min_similarity=min_similarity, max_age=max_age)
    lines = []
    frame_id = 0
    observed = 0
    reassociated = 0
    writer = None
    try:
        if annotate is not None:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else 30.0
            annotate.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(annotate), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            if not writer.isOpened():
                raise ValueError(f"cannot create annotated video: {annotate}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            rows = frames_to_boxes.get(frame_id, np.empty((0, 5), dtype=np.float64))
            # Exclude low-score rows BEFORE crop encoding to avoid unsafe crops
            # from detections ignored by the track associator.
            rows = rows[rows[:, 4] >= tracker.min_score]
            boxes, scores = rows[:, :4], rows[:, 4]
            features = embedder.encode(frame, boxes)
            assignments = tracker.update(frame_id, boxes, features, scores)
            for a in assignments:
                x1, y1, x2, y2 = a.bbox
                lines.append(f"{frame_id},{a.track_id},{x1:.2f},{y1:.2f},"
                             f"{x2-x1:.2f},{y2-y1:.2f},{a.score:.4f},-1,-1,-1\n")
                observed += 1
                reassociated += int(a.matched_existing)
                if writer is not None:
                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {a.track_id}", p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if writer is not None:
                writer.write(frame)
        if any(number > frame_id for number in frames_to_boxes):
            raise ValueError("detection frame index exceeds video length")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_id < 1:
        raise ValueError("video contains no frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    return {"scope": "appearance_association_without_ground_truth",
            "frame_count": frame_id, "observations": observed,
            "matched_existing": reassociated,
            "unique_ids": len(set(int(row.split(",")[1]) for row in lines)),
            "embedding_method": embedder.name,
            "weight_sha256": getattr(embedder, "weights_sha256", None),
            "input_video_sha256": sha256_stream(video),
            "input_detections_sha256": sha256_stream(detections)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--annotate", type=Path)
    p.add_argument("--embedding", choices=("hsv", "resnet18"), default="hsv")
    p.add_argument("--weights", type=Path)
    p.add_argument("--weights-sha256")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--min-similarity", type=float, default=0.75)
    p.add_argument("--max-age", type=int, default=30)
    args = p.parse_args()
    if args.embedding == "hsv":
        if args.weights is not None or args.weights_sha256 is not None:
            p.error("--weights only applies to resnet18")
        embedder = HSVHistogramEmbedder()
    else:
        if args.weights is None:
            p.error("resnet18 requires --weights (local torchvision state dict)")
        embedder = ResNet18Embedder(args.weights, args.device, args.weights_sha256)
    summary = run(args.video, args.detections, args.output, embedder=embedder,
                  min_similarity=args.min_similarity, max_age=args.max_age,
                  annotate=args.annotate)
    (args.output.with_suffix(".provenance.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
