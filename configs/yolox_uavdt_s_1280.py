from pathlib import Path

from yolox.exp import Exp as BaseExp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Exp(BaseExp):
    def __init__(self) -> None:
        super().__init__()

        # Model
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1

        # Dataset
        self.data_dir = str(
            PROJECT_ROOT
            / "datasets"
            / "UAVDT"
            / "coco_vehicle"
        )
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # Small aerial vehicles
        self.input_size = (1280, 1280)
        self.test_size = (1280, 1280)
        self.random_size = (40, 40)

        # Training
        self.max_epoch = 50
        self.warmup_epochs = 2
        self.no_aug_epochs = 10
        self.eval_interval = 5
        self.data_num_workers = 4

        self.exp_name = "yolox_uavdt_s_1280"
        self.output_dir = str(
            PROJECT_ROOT
            / "experiments"
            / "detector_runs"
        )