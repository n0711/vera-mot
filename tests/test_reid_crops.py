import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vera_mot.reid_crops import (
    dataset_checksum,
    materialize_split,
    read_manifest,
    safe_relative_path,
    verify_crop,
    write_registry,
)
from vera_mot.reid_data import FIELDS, ManifestError


class ReIDCropTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "datasets/UAVDT/full_mot/raw/UAV-benchmark-M/S1/img000001.jpg"
        source.parent.mkdir(parents=True)
        pixels = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
        Image.fromarray(pixels, "RGB").save(source, quality=95)
        self.row = {
            "sequence": "S1", "frame": 1, "identity": "S1_1", "target_id": 1,
            "category": 1, "identity_majority_category": 1,
            "identity_category_conflict": 0, "out_of_view": 1, "occlusion": 1,
            "original_left": 5, "original_top": 6, "original_width": 11,
            "original_height": 12, "clipped_left": 5, "clipped_top": 6,
            "clipped_width": 11, "clipped_height": 12, "clipped_area": 132,
            "source_image": "datasets/UAVDT/full_mot/raw/UAV-benchmark-M/S1/img000001.jpg",
            "planned_crop_path": "datasets/UAVDT/reid_crops/train/S1_1/S1_1.png",
            "split": "train", "sampling_index": 0,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _manifest(self, row=None):
        path = self.root / "manifest.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(self.row if row is None else row)
        return path

    def test_exact_png_extraction_native_size_and_verified_resume(self):
        rows = read_manifest(self._manifest(), "train", set())
        registry, stats = materialize_split(self.root, rows)
        self.assertEqual(stats, {"written_crops": 1, "resumed_crops": 0, "failed_crops": 0})
        crop_path = self.root / self.row["planned_crop_path"]
        with Image.open(crop_path) as crop, Image.open(self.root / self.row["source_image"]) as source:
            expected = source.crop((5, 6, 16, 18))
            self.assertEqual(crop.format, "PNG")
            self.assertEqual(crop.size, (11, 12))
            self.assertEqual(crop.tobytes(), expected.tobytes())
        second_registry, second_stats = materialize_split(self.root, rows)
        self.assertEqual(second_stats["resumed_crops"], 1)
        self.assertEqual(registry, second_registry)

    def test_mismatched_existing_crop_rejected(self):
        rows = read_manifest(self._manifest(), "train", set())
        crop_path = self.root / self.row["planned_crop_path"]
        crop_path.parent.mkdir(parents=True)
        Image.new("RGB", (11, 12), "black").save(crop_path)
        with self.assertRaisesRegex(ManifestError, "existing crop mismatch"):
            materialize_split(self.root, rows)
        registry, stats = materialize_split(self.root, rows, force=True)
        self.assertEqual(stats["written_crops"], 1)
        self.assertEqual(len(registry), 1)

    def test_path_traversal_absolute_test_and_split_rejected(self):
        for value in ("../escape.png", "/tmp/escape.png"):
            with self.assertRaises(ManifestError):
                safe_relative_path(value)
        with self.assertRaisesRegex(ManifestError, "frozen test"):
            read_manifest(self._manifest(), "train", {"S1"})
        bad = dict(self.row, split="test")
        with self.assertRaisesRegex(ManifestError, "invalid split"):
            read_manifest(self._manifest(bad), "train", set())

    def test_registry_and_dataset_checksum_deterministic(self):
        rows = read_manifest(self._manifest(), "train", set())
        registry, _ = materialize_split(self.root, rows)
        one, two = self.root / "one.csv", self.root / "two.csv"
        self.assertEqual(write_registry(one, registry), write_registry(two, list(reversed(registry))))
        self.assertEqual(one.read_bytes(), two.read_bytes())
        self.assertEqual(dataset_checksum(registry), dataset_checksum(list(reversed(registry))))


if __name__ == "__main__":
    unittest.main()
