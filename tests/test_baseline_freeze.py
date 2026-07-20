from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from convert_uavdt_yolo_to_coco import parse_yolo_annotation
from filter_uavdt_ignore_regions import filter_sequence
from vera_mot.mot_io import MotRow, parse_mot_text

import baseline_artifacts


class YoloConversionTests(unittest.TestCase):
    def test_all_source_classes_collapse_to_vehicle(self):
        first = parse_yolo_annotation("0 0.5 0.5 0.2 0.4", 100, 50)
        second = parse_yolo_annotation("7 0.5 0.5 0.2 0.4", 100, 50)
        self.assertEqual(first["category_id"], 1)
        self.assertEqual(second["category_id"], 1)
        self.assertEqual(first["source_class_id"], 0)
        self.assertEqual(second["source_class_id"], 7)
        self.assertEqual(first["bbox"], second["bbox"])


class MotIoTests(unittest.TestCase):
    def test_parse_and_write_round_trip(self):
        row = MotRow.parse("1,2,10.00,20.00,30.00,40.00,0.5000,-1,-1,-1")
        self.assertEqual(MotRow.parse(row.format()), row)

    def test_empty_and_malformed_results_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_mot_text("")
        with self.assertRaisesRegex(ValueError, "10 columns"):
            parse_mot_text("1,2,3")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_mot_text("0,2,1,1,2,2,0.5,-1,-1,-1")


class IgnoreFilteringTests(unittest.TestCase):
    def test_valid_match_is_kept_and_unmatched_ignore_overlap_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = root / "tracker.txt"
            gt = root / "gt.txt"
            ignore = root / "ignore.txt"
            output = root / "filtered.txt"
            tracker.write_text(
                "1,1,0,0,10,10,0.9,-1,-1,-1\n"
                "1,2,20,20,10,10,0.8,-1,-1,-1\n",
                encoding="utf-8",
            )
            gt.write_text("1,1,0,0,10,10,1,1,-1\n", encoding="utf-8")
            ignore.write_text("1,1,20,20,10,10,1,-1,-1\n", encoding="utf-8")
            input_count, removed = filter_sequence(
                tracker, gt, ignore, output, match_iou=0.5, ignore_ioa=0.5
            )
            self.assertEqual((input_count, removed), (2, 1))
            self.assertIn("1,1,0,0,10,10", output.read_text(encoding="utf-8"))
            self.assertNotIn("1,2,20,20,10,10", output.read_text(encoding="utf-8"))


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.yaml"
        self.registry = self.root / "registry.json"
        self.result = self.root / "M0001.txt"
        self.manifest.write_text(
            json.dumps({"frozen_configuration": {"threshold": 0.5}}),
            encoding="utf-8",
        )
        self.result.write_text(
            "1,1,0.00,0.00,10.00,10.00,0.9000,-1,-1,-1\n",
            encoding="utf-8",
        )
        self.old_manifest = baseline_artifacts.MANIFEST_PATH
        self.old_registry = baseline_artifacts.REGISTRY_PATH
        baseline_artifacts.MANIFEST_PATH = self.manifest
        baseline_artifacts.REGISTRY_PATH = self.registry
        fingerprint = baseline_artifacts.configuration_fingerprint(
            baseline_artifacts.load_manifest()
        )
        self.registry.write_text(
            json.dumps(
                {
                    "configuration_fingerprint": fingerprint,
                    "results": {
                        "M0001": {
                            "sha256": hashlib.sha256(
                                self.result.read_bytes()
                            ).hexdigest()
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        baseline_artifacts.MANIFEST_PATH = self.old_manifest
        baseline_artifacts.REGISTRY_PATH = self.old_registry
        self.temporary.cleanup()

    def test_matching_result_is_accepted(self):
        valid, _ = baseline_artifacts.verify_result("M0001", self.result)
        self.assertTrue(valid)

    def test_configuration_mismatch_is_rejected(self):
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        data["frozen_configuration"]["threshold"] = 0.6
        self.manifest.write_text(json.dumps(data), encoding="utf-8")
        valid, message = baseline_artifacts.verify_result("M0001", self.result)
        self.assertFalse(valid)
        self.assertIn("fingerprint", message)

    def test_content_mismatch_is_rejected(self):
        self.result.write_text(
            "1,1,0.00,0.00,11.00,10.00,0.9000,-1,-1,-1\n",
            encoding="utf-8",
        )
        valid, message = baseline_artifacts.verify_result("M0001", self.result)
        self.assertFalse(valid)
        self.assertIn("checksum mismatch", message)


if __name__ == "__main__":
    unittest.main()
