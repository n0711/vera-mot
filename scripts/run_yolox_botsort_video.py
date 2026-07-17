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

# Load the official YOLOX package first.
sys.path.insert(0, str(YOLOX_ROOT))

from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402

# Then expose the pinned BoT-SORT tracker.
sys.path.insert(0, str(BOTSORT_ROOT))

from tracker.bot_sort import BoTSORT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOX-S with BoT-SORT without ReID."
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets"
            / "local_private"
            / "uav_test"
            / "uav_cars_v1.mp4"
        ),
    )

    parser.add_argument(
        "--exp",
        type=Path,
        default=PROJECT_ROOT / "configs" / "yolox_uavdt_s_1280_test.py",
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
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "tracking_baselines"
            / "botsort_no_reid"
            / "uav_cars_v1_tracked.mp4"
        ),
    )

    parser.add_argument("--det-conf", type=float, default=0.05)
    parser.add_argument("--nms", type=float, default=0.65)

    parser.add_argument("--track-high", type=float, default=0.35)
    parser.add_argument("--track-low", type=float, default=0.10)
    parser.add_argument("--new-track", type=float, default=0.45)
    parser.add_argument("--match-thresh", type=float, default=0.80)
    parser.add_argument("--track-buffer", type=int, default=30)

    parser.add_argument(
        "--cmc-method",
        choices=["sparseOptFlow", "orb", "ecc", "none"],
        default="sparseOptFlow",
    )

    parser.add_argument("--min-box-area", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def tracker_args(args: argparse.Namespace) -> SimpleNamespace:
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
        cmc_method=args.cmc_method,
        name=args.video.stem,
        ablation=False,
        mot20=False,
    )


def colour_for_id(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id) % 255),
        int((17 * track_id) % 255),
        int((29 * track_id) % 255),
    )


def load_model(
    exp_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    fp16: bool,
):
    exp = get_exp(str(exp_path), None)
    model = exp.get_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    if fp16:
        model.half()

    return exp, model


def main() -> None:
    args = parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    if not args.exp.is_file():
        raise FileNotFoundError(f"Experiment not found: {args.exp}")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_path = args.output.with_suffix(".txt")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_fp16 = args.fp16 and device.type == "cuda"

    exp, model = load_model(
        args.exp,
        args.checkpoint,
        device,
        use_fp16,
    )

    transform = ValTransform(legacy=False)

    capture = cv2.VideoCapture(str(args.video))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create: {args.output}")

    tracker = BoTSORT(
        tracker_args(args),
        frame_rate=int(round(fps)),
    )

    frame_index = 0
    detection_total = 0
    confirmed_track_total = 0
    result_lines: list[str] = []

    started = time.perf_counter()

    while True:
        success, frame = capture.read()

        if not success:
            break

        frame_index += 1

        ratio = min(
            exp.test_size[0] / height,
            exp.test_size[1] / width,
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
            detection_total += len(detections)

        online_tracks = tracker.update(
            detections,
            frame,
        )

        confirmed_tracks = 0

        for track in online_tracks:
            # Hide unconfirmed one-frame tracks from rendered output.
            if not track.is_activated:
                continue

            x, y, box_width, box_height = track.tlwh

            if box_width * box_height < args.min_box_area:
                continue

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(width - 1, int(x + box_width))
            y2 = min(height - 1, int(y + box_height))

            track_id = int(track.track_id)
            score = float(track.score)
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
                f"ID {track_id} {score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                2,
                cv2.LINE_AA,
            )

            result_lines.append(
                f"{frame_index},{track_id},"
                f"{x:.2f},{y:.2f},"
                f"{box_width:.2f},{box_height:.2f},"
                f"{score:.4f},-1,-1,-1\n"
            )

            confirmed_tracks += 1

        confirmed_track_total += confirmed_tracks

        cv2.putText(
            frame,
            f"Frame {frame_index} | Confirmed tracks: "
            f"{confirmed_tracks}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_index % 100 == 0:
            print(
                f"Processed {frame_index}/{total_frames} frames"
            )

    elapsed = time.perf_counter() - started

    capture.release()
    writer.release()

    results_path.write_text(
        "".join(result_lines),
        encoding="utf-8",
    )

    processing_fps = frame_index / elapsed if elapsed > 0 else 0.0

    print()
    print(f"Frames: {frame_index}")
    print(f"Input FPS: {fps:.2f}")
    print(f"Processing FPS: {processing_fps:.2f}")
    print(f"Raw detections: {detection_total}")
    print(f"Confirmed track observations: {confirmed_track_total}")
    print(f"Video: {args.output}")
    print(f"MOT results: {results_path}")


if __name__ == "__main__":
    main()