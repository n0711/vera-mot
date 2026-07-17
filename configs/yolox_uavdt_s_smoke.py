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

        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (20, 20)

        self.max_epoch = 1
        self.warmup_epochs = 0
        self.no_aug_epochs = 0
        self.eval_interval = 1
        self.data_num_workers = 2

        self.exp_name = "yolox_uavdt_s_smoke"
        self.output_dir = str(
            PROJECT_ROOT
            / "experiments"
            / "detector_runs"
        )