# Current Experiment

## Milestone 3.1: frozen UAVDT BoT-SORT no-ReID baseline

The detector-only baseline and the 20-sequence BoT-SORT no-ReID experiment have
been executed. Milestone 3.1 preserves their configuration, provenance, compact
evidence, and safe result-reuse rules without changing any result.

Canonical configuration and results:

`experiments/tracking_baselines/botsort_no_reid/manifest.yaml`

The no-ReID benchmark result is HOTA 43.340, DetA 36.441, AssA 52.419,
IDF1 56.964, 1,180 ID switches, MOTA 31.265, MOTP 74.136, precision 67.048,
and recall 62.161 across all 20 UAVDT test sequences.

## Implemented

- YOLOX-S single-class UAVDT vehicle detector;
- held-out detector evaluation;
- BoT-SORT with Kalman motion, two-stage IoU association, sparse-optical-flow
  camera-motion compensation, and ReID disabled;
- UAVDT ignore-region filtering and TrackEval evaluation.

## Not implemented

- ByteTrack experiment;
- vehicle ReID and BoT-SORT-ReID baseline;
- VERA association or protected-memory mechanisms;
- ablations or ROS 2/edge demonstration.

## Next experiment

Establish the controlled BoT-SORT vehicle-ReID baseline.

## ReID validation implementation update (2026-10-25)

The training lifecycle is implemented in `scripts/train_uavdt_fastreid.py`.
A completed local 60-epoch run was reported during earlier project work; this
checkout has neither its checkpoints nor completion evidence, so the report
does not establish a tracked vehicle-ReID benchmark result.

The temporal validation evaluator and raw crop-quality analysis have been
integrated from their feature branches. The six-checkpoint sweep preserves
`model_final.pth` as the declared primary checkpoint and verifies shared splits,
provenance and resumable result checksums. Bounded smoke runs are explicitly
labelled. See `docs/reid_evaluation_workflow.md` for the next executable steps.

No new benchmark scores, reliability thresholds, VERA gating or protected-memory
results are claimed by this implementation update. The frozen no-ReID evidence
above remains the reference for a later controlled tracking comparison.
