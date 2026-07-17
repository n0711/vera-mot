# Current Experiment

## Detector

- Model: YOLOX-S
- Task: Ground-vehicle detection in UAV imagery
- Dataset: UAVDT detection subset
- Classes: 1 (`vehicle`)
- Training images: 1,266
- Validation images: 271
- Test images: 272
- Training boxes: 35,007
- Input resolution: 1280 × 1280
- Epochs: 50
- Batch size: 2
- Precision: FP16
- Initial weights: COCO-pretrained YOLOX-S
- Hardware: NVIDIA RTX 4060 Laptop GPU

## Output

Expected checkpoint:

`experiments/detector_runs/yolox_uavdt_s_1280/best_ckpt.pth`

## Next stages

1. Evaluate on the held-out test split.
2. Visualise predictions.
3. Connect YOLOX output to BoT-SORT.
4. Establish BoT-SORT without ReID.
5. Establish BoT-SORT with vehicle ReID.
6. Implement VERA-MOT.