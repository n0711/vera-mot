# UAVDT YOLOX-S Vehicle Detector Baseline

**Status:** trained and evaluated locally on 17 July 2026.

## Configuration

- YOLOX-S: depth 0.33, width 0.50;
- one collapsed `vehicle` class;
- 1,266 training, 271 validation, and 272 held-out test images;
- 1280×1280 training and inference;
- 50 epochs, batch size 2, FP16 on an RTX 4060 Laptop GPU;
- best checkpoint expected at
  `experiments/detector_runs/yolox_uavdt_s_1280/best_ckpt.pth`.

The frozen checkpoint SHA-256 is
`91b8c13682a3b74cdd74d45edc83899cfb8be367665f85c942c761a78189ceef`.
The checkpoint is not distributed by this repository.

## Benchmark results

Held-out COCO evaluation:

- AP@[0.50:0.95]: 50.132%;
- AP50: 83.5%;
- AP75: 56.1%;
- AR100: 55.305%;
- average reported inference time: 14.32 ms.

The compact canonical record is
`experiments/detector_evaluation/uavdt_test_summary.txt`.

## Limitations

The source/checksum of the UAVDT archives, derivation of the detection subset,
initial COCO checkpoint checksum, and a public location for the trained
checkpoint are not recorded. These omissions prevent full clean-clone
reproduction.

Private UAV-car and tank/domain-shift images are qualitative checks only. They
have no benchmark annotations here, so false-positive and missed-object
observations are unverified and must not be reported as measured detector
performance.
