import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines/BoT-SORT"))

from vera_mot.fastreid_uavdt import (
    load_uavdt_registry,
    validate_batch,
)
from vera_mot.reid_crops import REGISTRY_FIELDS
from vera_mot.reid_data import ManifestError


class UAVDTFastReIDAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry_root = self.root / "datasets/UAVDT/reid_manifests"
        self.crop_root = self.root / "datasets/UAVDT/reid_crops"
        self.registry_root.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def _rows(self, split="train", identities=("S1_10", "S2_3"), samples=4):
        rows = []
        for identity in identities:
            sequence = identity.split("_")[0]
            for frame in range(1, samples + 1):
                relative = f"datasets/UAVDT/reid_crops/{split}/{identity}/{sequence}_{frame}.png"
                crop = self.root / relative
                crop.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 10), (frame, 2, 3)).save(crop)
                import hashlib
                digest = hashlib.sha256(crop.read_bytes()).hexdigest()
                rows.append({
                    "split": split, "sequence": sequence, "frame": frame,
                    "identity": identity, "category": 1,
                    "identity_majority_category": 1, "identity_category_conflict": 0,
                    "crop_path": relative, "width": 12, "height": 10,
                    "file_size": crop.stat().st_size, "sha256": digest,
                    "source_image": f"datasets/UAVDT/full_mot/raw/UAV-benchmark-M/{sequence}/img{frame:06d}.jpg",
                    "source_region_sha256": "unused",
                })
        return rows

    def _write(self, rows, split="train"):
        path = self.registry_root / f"{split}_crop_registry.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        return path

    def test_registry_parsing_mapping_and_crop_verification(self):
        self._write(self._rows())
        first = load_uavdt_registry(self.root, "train", verify_crops=True, expected=None)
        second = load_uavdt_registry(self.root, "train", verify_crops=True, expected=None)
        self.assertEqual(first.identity_to_label, {"S1_10": 0, "S2_3": 1})
        self.assertEqual(first.identity_to_label, second.identity_to_label)
        self.assertEqual(first.sequence_to_camera, {"S1": 0, "S2": 1})
        self.assertEqual(len(first.records), 8)

    def test_minimum_four_and_duplicate_rejection(self):
        rows = self._rows(samples=3)
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "fewer than four"):
            load_uavdt_registry(self.root, "train", expected=None)
        rows = self._rows()
        self._write(rows + [rows[0]])
        with self.assertRaisesRegex(ManifestError, "duplicate"):
            load_uavdt_registry(self.root, "train", expected=None)

    def test_absolute_traversal_private_and_frozen_test_rejection(self):
        rows = self._rows()
        rows[0]["crop_path"] = "/tmp/bad.png"
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "unsafe"):
            load_uavdt_registry(self.root, "train", expected=None)
        rows = self._rows()
        rows[0]["crop_path"] = "datasets/UAVDT/reid_crops/train/../bad.png"
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "unsafe"):
            load_uavdt_registry(self.root, "train", expected=None)
        rows = self._rows()
        rows[0]["source_image"] = "private/video/frame.png"
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "required prefix"):
            load_uavdt_registry(self.root, "train", expected=None)
        rows = self._rows(identities=("M0203_1",), samples=4)
        self._write(rows)
        with self.assertRaisesRegex(ManifestError, "frozen-test"):
            load_uavdt_registry(self.root, "train", expected=None)

    def test_modified_crop_and_unsupported_split_rejected(self):
        rows = self._rows()
        self._write(rows)
        crop = self.root / rows[0]["crop_path"]
        Image.new("RGB", (12, 10), "black").save(crop)
        with self.assertRaisesRegex(ManifestError, "modified crop"):
            load_uavdt_registry(self.root, "train", verify_crops=True, expected=None)
        with self.assertRaisesRegex(ManifestError, "unsupported"):
            load_uavdt_registry(self.root, "test", expected=None)

    def test_batch_composition(self):
        result = validate_batch([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
        self.assertEqual(result["distinct_identities"], 4)
        with self.assertRaises(ManifestError):
            validate_batch([0] * 16)

    def test_train_validation_identity_spaces_are_separate(self):
        self._write(self._rows("train", ("S1_1",)), "train")
        self._write(self._rows("val", ("S1_1",)), "val")
        train = load_uavdt_registry(self.root, "train", expected=None)
        val = load_uavdt_registry(self.root, "val", expected=None)
        self.assertEqual(train.identity_to_label, {"S1_1": 0})
        self.assertEqual(val.identity_to_label, {"S1_1": 0})
        self.assertNotEqual(train.records[0].crop_path, val.records[0].crop_path)


class UAVDTFastReIDConfigTests(unittest.TestCase):
    def test_configuration_invariants(self):
        script = ROOT / "scripts/dry_run_uavdt_fastreid_loader.py"
        spec = importlib.util.spec_from_file_location("uavdt_loader_dry_run", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cfg = module.load_config(ROOT / "configs/fastreid/uavdt_sbs_R50_ibn.yml")
        module.validate_config(cfg)
        self.assertEqual(cfg.SOLVER.IMS_PER_BATCH, 16)
        self.assertEqual(cfg.DATALOADER.NUM_INSTANCE, 4)


if __name__ == "__main__":
    unittest.main()
