import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vera_mot.reid_data import (
    ManifestError,
    build_split_rows,
    clip_box,
    composite_identity,
    parse_gt_line,
    uniform_indices,
    validate_manifest_leakage,
    validate_split_config,
    write_csv,
)


class ReIDUnitTests(unittest.TestCase):
    def test_gt_parsing_and_composite_identity(self):
        row = parse_gt_line("12,7,3,4,20,30,2,4,1")
        self.assertEqual(row, {
            "frame": 12, "target_id": 7, "left": 3, "top": 4,
            "width": 20, "height": 30, "out_of_view": 2,
            "occlusion": 4, "category": 1,
        })
        self.assertEqual(composite_identity("M0101", row["target_id"]), "M0101_7")
        with self.assertRaises(ManifestError):
            parse_gt_line("1,2,3")

    def test_clipping(self):
        row = parse_gt_line("1,1,-5,-2,20,20,1,1,1")
        self.assertEqual(clip_box(row, 10, 10), (0, 0, 10, 10))

    def test_uniform_cap_has_temporal_coverage(self):
        indices = uniform_indices(300, 200)
        self.assertEqual(len(indices), 200)
        self.assertEqual((indices[0], indices[-1]), (0, 299))
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(indices, uniform_indices(300, 200))

    def test_split_test_and_sequence_leakage_rejected(self):
        config = {
            "seed": 42,
            "train_sequences": [f"A{i}" for i in range(24)],
            "validation_sequences": [f"V{i}" for i in range(6)],
            "frozen_test_sequences": [f"T{i}" for i in range(20)],
        }
        validate_split_config(config)
        config["validation_sequences"][0] = "A0"
        with self.assertRaisesRegex(ManifestError, "sequence leakage"):
            validate_split_config(config)


class SyntheticManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.gt_root = self.root / "gt"
        self.image_root = self.root / "images"
        self.gt_root.mkdir()
        (self.image_root / "S1").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def _image(self, frame, size=(40, 30)):
        Image.new("RGB", size).save(self.image_root / "S1" / f"img{frame:06d}.jpg")

    def _build(self, lines, **kwargs):
        (self.gt_root / "S1_gt_whole.txt").write_text("\n".join(lines) + "\n")
        return build_split_rows(
            self.root, ["S1"], "train", self.gt_root, self.image_root,
            minimum_observations=kwargs.get("minimum_observations", 10),
            temporal_stride=kwargs.get("temporal_stride", 5),
            maximum_samples=kwargs.get("maximum_samples", 200),
        )

    def test_category_filter_clipping_tiny_rejection_and_minimum_track(self):
        lines = []
        for frame in range(1, 23):
            self._image(frame)
            # ID 1: twenty valid supported observations, one unsupported, one tiny.
            if frame == 21:
                lines.append(f"{frame},1,0,0,20,20,1,1,9")
            elif frame == 22:
                lines.append(f"{frame},1,0,0,9,20,1,1,1")
            else:
                lines.append(f"{frame},1,-2,-2,20,20,3,4,{1 + frame % 3}")
            # ID 2 is filtered because it has only nine valid observations.
            if frame <= 9:
                lines.append(f"{frame},2,1,1,15,15,1,2,1")
        rows, stats = self._build(lines)
        self.assertEqual([row["frame"] for row in rows], [1, 6, 11, 16])
        self.assertEqual(rows[0]["identity"], "S1_1")
        self.assertEqual(rows[0]["clipped_width"], 18)
        self.assertEqual(stats["rejection_counts"]["unsupported_category"], 1)
        self.assertEqual(stats["rejection_counts"]["clipped_width_lt_10"], 1)
        self.assertEqual(stats["rejection_counts"]["rejected_identities"], 1)
        self.assertTrue(all(row["category"] in (1, 2, 3) for row in rows))

    def test_every_fifth_then_uniform_200_cap(self):
        lines = []
        for frame in range(1, 1251):
            self._image(frame)
            lines.append(f"{frame},1,0,0,20,20,1,1,1")
        rows, _ = self._build(lines)
        self.assertEqual(len(rows), 200)
        self.assertEqual((rows[0]["frame"], rows[-1]["frame"]), (1, 1246))
        self.assertGreater(rows[100]["frame"], 500)

    def test_minimum_four_after_sampling_and_category_conflict(self):
        lines = []
        for frame in range(1, 20):
            self._image(frame)
            category = 1 if frame < 12 else 2
            lines.append(f"{frame},1,0,0,20,20,1,1,{category}")
        rows, stats = self._build(lines)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["identity_majority_category"] == 1 for row in rows))
        self.assertTrue(all(row["identity_category_conflict"] == 1 for row in rows))
        self.assertEqual(stats["category_conflicts"]["identities_with_multiple_raw_categories"], 1)

        # Ten valid observations satisfy source length but produce only two samples.
        short_lines = []
        for frame in range(1, 11):
            short_lines.append(f"{frame},2,0,0,20,20,1,1,1")
        rows, stats = self._build(short_lines)
        self.assertEqual(rows, [])
        self.assertEqual(stats["identity_accounting"]["rejection_reason_counts"]["fewer_than_4_final_planned_crops"], 1)

    def test_duplicate_rejected(self):
        self._image(1)
        line = "1,1,0,0,20,20,1,1,1"
        with self.assertRaisesRegex(ManifestError, "duplicate"):
            self._build([line, line], minimum_observations=1)

    def test_missing_frame_and_gt_rejected(self):
        with self.assertRaisesRegex(ManifestError, "missing"):
            self._build(["1,1,0,0,20,20,1,1,1"], minimum_observations=1)
        with self.assertRaisesRegex(ManifestError, "required GT"):
            build_split_rows(self.root, ["NOPE"], "train", self.gt_root, self.image_root)

    def test_identity_and_test_leakage_rejected(self):
        base = {"sequence": "S1", "frame": 1, "identity": "shared"}
        val = {"sequence": "S2", "frame": 1, "identity": "shared"}
        with self.assertRaisesRegex(ManifestError, "identity occurs"):
            validate_manifest_leakage([base], [val], [])
        with self.assertRaisesRegex(ManifestError, "frozen test"):
            validate_manifest_leakage([base], [], ["S1"])

    def test_deterministic_byte_identical_csv(self):
        lines = []
        for frame in range(1, 21):
            self._image(frame)
            lines.append(f"{frame},1,0,0,20,20,2,3,1")
        first, _ = self._build(lines)
        second, _ = self._build(lines)
        path_one, path_two = self.root / "one.csv", self.root / "two.csv"
        self.assertEqual(write_csv(path_one, first), write_csv(path_two, second))
        self.assertEqual(path_one.read_bytes(), path_two.read_bytes())
        with path_one.open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == "__main__":
    unittest.main()
