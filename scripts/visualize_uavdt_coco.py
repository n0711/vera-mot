from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "UAVDT"
    / "coco_vehicle"
)

IMAGE_DIR = DATASET_ROOT / "train2017"
ANNOTATION_FILE = (
    DATASET_ROOT
    / "annotations"
    / "instances_train2017.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "uavdt_annotation_check"
)

NUMBER_OF_IMAGES = 12


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(ANNOTATION_FILE.read_text())

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    images = data["images"]

    if len(images) < NUMBER_OF_IMAGES:
        raise SystemExit("Not enough images in dataset")

    random.seed(42)
    selected_images = random.sample(images, NUMBER_OF_IMAGES)

    for image_info in selected_images:
        image_path = IMAGE_DIR / image_info["file_name"]
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        annotations = annotations_by_image.get(image_info["id"], [])

        for annotation in annotations:
            x, y, width, height = annotation["bbox"]

            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + width))
            y2 = int(round(y + height))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                "vehicle",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        output_path = OUTPUT_DIR / image_info["file_name"]
        cv2.imwrite(str(output_path), image)

        print(
            f"Saved: {output_path.name} "
            f"({len(annotations)} boxes)"
        )

    print(f"\nOpen folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()