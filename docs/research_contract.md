# VERA-MOT Research Contract

## Method and research question

VERA means Visual Embedding Reliability Assessment. The thesis asks whether a
lightweight reliability assessment can prevent blurred, occluded, undersized,
or otherwise unreliable UAV vehicle embeddings from corrupting association and
long-term identity memory without unacceptable runtime overhead.

## Controlled progression

1. UAVDT vehicle detector
2. ByteTrack comparator
3. BoT-SORT without ReID
4. BoT-SORT with vehicle ReID
5. VERA reliability-gated association
6. VERA protected identity-memory updates
7. Ablations and evaluation
8. Edge/ROS 2 demonstration, later

BoT-SORT-ReID is the eventual direct baseline for VERA. The currently frozen
milestone is the preceding BoT-SORT no-ReID baseline.

## Included

- public UAV vehicle datasets and baseline reproduction;
- vehicle appearance adaptation;
- embedding reliability assessment;
- reliability-weighted association and protected identity memory;
- ablations, tracking metrics, and onboard latency/resource profiling.

## Excluded from the thesis core

- single-object tracking and heading estimation;
- gimbal or flight control;
- MAVLink command authority;
- autonomous following and real closed-loop flight testing;
- the complete ADDITESS guidance pipeline.

Claims of recovery or identity retention require annotated identity evidence.
Qualitative private-video observations are not benchmark results.
