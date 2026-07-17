from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "UAVDT"
    / "yolo_detection_subset"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "UAVDT"
    / "coco_vehicle"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def convert_split(source_split: str, coco_split: str) -> None:
    image_source = SOURCE_ROOT / source_split / "images"
    label_source = SOURCE_ROOT / source_split / "labels"

    image_output = OUTPUT_ROOT / coco_split
    annotation_output = OUTPUT_ROOT / "annotations"

    image_output.mkdir(parents=True, exist_ok=True)
    annotation_output.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    image_paths = sorted(
        path
        for path in image_source.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        destination = image_output / image_path.name

        if not destination.exists():
            os.symlink(image_path.resolve(), destination)

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_source / f"{image_path.stem}.txt"

        if label_path.exists():
            for line_number, line in enumerate(
                label_path.read_text().splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        f"expected 5 values, found {len(parts)}"
                    )

                class_id, x_center, y_center, box_width, box_height = (
                    map(float, parts)
                )

                source_class_id = int(class_id)

                if source_class_id < 0:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        f"invalid class {class_id}"
                    )

                values = (
                    x_center,
                    y_center,
                    box_width,
                    box_height,
                )

                if not all(0.0 <= value <= 1.0 for value in values):
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        f"coordinates outside [0, 1]"
                    )

                pixel_width = box_width * width
                pixel_height = box_height * height

                x_min = (x_center * width) - (pixel_width / 2.0)
                y_min = (y_center * height) - (pixel_height / 2.0)

                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)

                pixel_width = min(pixel_width, width - x_min)
                pixel_height = min(pixel_height, height - y_min)

                if pixel_width <= 0 or pixel_height <= 0:
                    raise ValueError(
                        f"{label_path}:{line_number}: invalid box"
                    )

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "source_class_id": source_class_id,
                        "bbox": [
                            round(x_min, 3),
                            round(y_min, 3),
                            round(pixel_width, 3),
                            round(pixel_height, 3),
                        ],
                        "area": round(pixel_width * pixel_height, 3),
                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

        image_id += 1

    coco = {
        "info": {
            "description": "UAVDT vehicle detection subset",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "vehicle",
                "supercategory": "vehicle",
            }
        ],
    }

    output_path = (
        annotation_output
        / f"instances_{coco_split}.json"
    )

    output_path.write_text(json.dumps(coco, indent=2))

    print(
        f"{source_split:5s} → {coco_split:9s}: "
        f"{len(images)} images, "
        f"{len(annotations)} boxes"
    )


def main() -> None:
    convert_split("train", "train2017")
    convert_split("val", "val2017")
    convert_split("test", "test2017")

    print(f"\nCreated: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()