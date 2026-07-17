from pathlib import Path

from yolox.exp import Exp as BaseExp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Exp(BaseExp):
    def __init__(self) -> None:
        super().__init__()

        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1

        self.data_dir = str(
            PROJECT_ROOT
            / "datasets"
            / "UAVDT"
            / "coco_vehicle"
        )

        self.test_ann = "instances_test2017.json"

        self.test_size = (1280, 1280)
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.data_num_workers = 4

        self.exp_name = "yolox_uavdt_s_1280_test"
        self.output_dir = str(
            PROJECT_ROOT
            / "experiments"
            / "detector_evaluation"
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann,
            name="test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )