# BoT-SORT Without ReID — Frozen Baseline

**Status date:** 17 July 2026
**Freeze milestone:** 3.1

## Benchmark configuration

- detector: YOLOX-S UAVDT, one vehicle class, 1280×1280;
- detector confidence 0.05 and NMS 0.65;
- track high/low/new thresholds: 0.35/0.10/0.45;
- first-stage match threshold 0.80; second-stage threshold 0.50;
- track buffer 30 and assumed frame rate 30 FPS;
- sparse-optical-flow camera-motion compensation;
- ReID disabled;
- all 20 UAVDT test sequences;
- valid-GT filtering IoU 0.50 and ignore-region IoA 0.50.

The machine-readable record is
`experiments/tracking_baselines/botsort_no_reid/manifest.yaml`.

## Benchmark results

| HOTA | DetA | AssA | IDF1 | IDSW | MOTA | MOTP | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 43.340 | 36.441 | 52.419 | 56.964 | 1,180 | 31.265 | 74.136 | 67.048 | 62.161 |

TrackEval completed 20/20 sequences with no failed or skipped evaluation
sequence.

## Qualitative observations

A private normal-UAV video produced a rendered tracking artifact. Notes report
short detection misses and a persistent non-vehicle false track. A motorcycle
was reportedly displayed with the same tracker ID after a miss, but the video
has no ground-truth identity annotation here. This is not verified recovery:
temporary loss followed by a displayed ID cannot establish identity correctness.

Military/domain-shift footage also has prediction artifacts but no quantitative
annotations. Reports of tyre false positives or a missed tank remain unverified
and are outside the frozen UAVDT benchmark.

## Limitations

- raw results and large logs remain local; their checksums are preserved in the
  compact result registry;
- datasets and detector weights are not distributed;
- custom ignore preprocessing has not been validated against the official
  UAVDT MATLAB evaluator;
- no ReID or VERA logic is part of this baseline.
