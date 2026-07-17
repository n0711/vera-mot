#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLOX_ROOT = PROJECT_ROOT / "baselines" / "YOLOX"

sys.path.insert(0, str(YOLOX_ROOT))

from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import postprocess  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise YOLOX vehicle predictions on UAVDT images."
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
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets"
            / "UAVDT"
            / "coco_vehicle"
            / "test2017"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "detector_evaluation"
            / "uavdt_test_predictions"
        ),
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.65)
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if path.is_file():
        return [path]

    return sorted(
        file
        for file in path.rglob("*")
        if file.suffix.lower() in extensions
    )


def draw_predictions(
    image,
    output,
    ratio: float,
    confidence_threshold: float,
):
    if output is None:
        return image, 0

    output = output.cpu()

    boxes = output[:, :4] / ratio
    scores = output[:, 4] * output[:, 5]

    detection_count = 0

    for box, score in zip(boxes, scores):
        confidence = float(score)

        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = f"vehicle {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        detection_count += 1

    return image, detection_count


def main() -> None:
    args = parse_args()

    if not args.exp.is_file():
        raise FileNotFoundError(f"Experiment file not found: {args.exp}")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    images = collect_images(args.input)

    if not images:
        raise RuntimeError(f"No images found under: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    exp = get_exp(str(args.exp), None)
    exp.test_conf = args.confidence
    exp.nmsthre = args.nms

    model = exp.get_model()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    use_fp16 = args.fp16 and device.type == "cuda"

    if use_fp16:
        model.half()

    transform = ValTransform(legacy=False)

    selected_images = images[: args.limit]

    for index, image_path in enumerate(selected_images, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        original_height, original_width = image.shape[:2]

        ratio = min(
            exp.test_size[0] / original_height,
            exp.test_size[1] / original_width,
        )

        processed, _ = transform(
            image,
            None,
            exp.test_size,
        )

        tensor = torch.from_numpy(processed).unsqueeze(0)
        tensor = tensor.to(device)

        tensor = tensor.half() if use_fp16 else tensor.float()

        with torch.no_grad():
            output = model(tensor)

            output = postprocess(
                output,
                exp.num_classes,
                args.confidence,
                args.nms,
                class_agnostic=True,
            )[0]

        visualised, count = draw_predictions(
            image.copy(),
            output,
            ratio,
            args.confidence,
        )

        output_path = args.output / image_path.name
        cv2.imwrite(str(output_path), visualised)

        print(
            f"[{index}/{len(selected_images)}] "
            f"{image_path.name}: {count} detections"
        )

    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()