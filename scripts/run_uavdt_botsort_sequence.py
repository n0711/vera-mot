#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLOX_ROOT = PROJECT_ROOT / "baselines" / "YOLOX"
BOTSORT_ROOT = PROJECT_ROOT / "baselines" / "BoT-SORT"

sys.path.insert(0, str(YOLOX_ROOT))

from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

sys.path.insert(0, str(BOTSORT_ROOT))

from tracker.bot_sort import BoTSORT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOX and BoT-SORT on one UAVDT sequence."
    )

    parser.add_argument("--sequence", default="M0203")

    parser.add_argument(
        "--frames-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets"
            / "UAVDT"
            / "full_mot"
            / "raw"
            / "UAV-benchmark-M"
        ),
    )

    parser.add_argument(
        "--exp",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "yolox_uavdt_s_1280_test.py",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "detector_runs"
            / "yolox_uavdt_s_1280"
            / "best_ckpt.pth"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "tracking_baselines"
            / "botsort_no_reid"
            / "uavdt"
        ),
    )

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--det-conf", type=float, default=0.05)
    parser.add_argument("--nms", type=float, default=0.65)
    parser.add_argument("--track-high", type=float, default=0.35)
    parser.add_argument("--track-low", type=float, default=0.10)
    parser.add_argument("--new-track", type=float, default=0.45)
    parser.add_argument("--match-thresh", type=float, default=0.80)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--min-box-area", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def make_tracker_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        track_high_thresh=args.track_high,
        track_low_thresh=args.track_low,
        new_track_thresh=args.new_track,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
        fast_reid_config="",
        fast_reid_weights="",
        device="cuda",
        cmc_method="sparseOptFlow",
        name=args.sequence,
        ablation=False,
        mot20=False,
    )


def colour_for_id(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id) % 255),
        int((17 * track_id) % 255),
        int((29 * track_id) % 255),
    )


def main() -> None:
    args = parse_args()

    sequence_dir = args.frames_root / args.sequence
    frame_paths = sorted(sequence_dir.glob("img*.jpg"))

    if not frame_paths:
        raise FileNotFoundError(
            f"No UAVDT frames found in: {sequence_dir}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    results_path = args.output_root / f"{args.sequence}.txt"
    video_path = args.output_root / f"{args.sequence}.mp4"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_fp16 = args.fp16 and device.type == "cuda"

    exp = get_exp(str(args.exp), None)
    model = exp.get_model()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    if use_fp16:
        model.half()

    transform = ValTransform(legacy=False)

    first_frame = cv2.imread(str(frame_paths[0]))

    if first_frame is None:
        raise RuntimeError(f"Cannot read: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    writer = None

    if args.render:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Cannot create: {video_path}")

    tracker = BoTSORT(
        make_tracker_args(args),
        frame_rate=args.fps,
    )

    result_lines: list[str] = []
    total_detections = 0
    total_track_observations = 0

    started = time.perf_counter()

    for index, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))

        if frame is None:
            raise RuntimeError(f"Cannot read: {frame_path}")

        frame_id = int(frame_path.stem.replace("img", ""))

        frame_height, frame_width = frame.shape[:2]

        ratio = min(
            exp.test_size[0] / frame_height,
            exp.test_size[1] / frame_width,
        )

        processed, _ = transform(
            frame,
            None,
            exp.test_size,
        )

        tensor = torch.from_numpy(processed)
        tensor = tensor.unsqueeze(0).to(device)
        tensor = tensor.half() if use_fp16 else tensor.float()

        with torch.no_grad():
            prediction = model(tensor)

            prediction = postprocess(
                prediction,
                exp.num_classes,
                args.det_conf,
                args.nms,
                class_agnostic=True,
            )[0]

        if prediction is None:
            detections = np.empty((0, 7), dtype=np.float32)
        else:
            detections = prediction.detach().cpu().numpy()
            detections[:, :4] /= ratio
            total_detections += len(detections)

        online_tracks = tracker.update(detections, frame)

        for track in online_tracks:
            if not track.is_activated:
                continue

            x, y, box_width, box_height = track.tlwh

            if box_width * box_height < args.min_box_area:
                continue

            track_id = int(track.track_id)
            score = float(track.score)

            result_lines.append(
                f"{frame_id},{track_id},"
                f"{x:.2f},{y:.2f},"
                f"{box_width:.2f},{box_height:.2f},"
                f"{score:.4f},-1,-1,-1\n"
            )

            total_track_observations += 1

            if writer is not None:
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(width - 1, int(x + box_width))
                y2 = min(height - 1, int(y + box_height))

                colour = colour_for_id(track_id)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    colour,
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    2,
                    cv2.LINE_AA,
                )

        if writer is not None:
            writer.write(frame)

        if index % 100 == 0:
            print(
                f"{args.sequence}: "
                f"{index}/{len(frame_paths)} frames"
            )

    elapsed = time.perf_counter() - started

    if writer is not None:
        writer.release()

    results_path.write_text(
        "".join(result_lines),
        encoding="utf-8",
    )

    processing_fps = (
        len(frame_paths) / elapsed if elapsed > 0 else 0.0
    )

    print()
    print(f"Sequence: {args.sequence}")
    print(f"Frames: {len(frame_paths)}")
    print(f"Raw detections: {total_detections}")
    print(f"Track observations: {total_track_observations}")
    print(f"Processing FPS: {processing_fps:.2f}")
    print(f"Results: {results_path}")

    if writer is not None:
        print(f"Video: {video_path}")


if __name__ == "__main__":
    main()