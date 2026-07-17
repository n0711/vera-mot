# BoT-SORT Without ReID — Baseline Status

**Status date:** 17 July 2026

## Pipeline

UAV video
→ YOLOX-S ground-vehicle detections
→ BoT-SORT without ReID
→ persistent track IDs

## Configuration

- Detector: YOLOX-S UAVDT ground-vehicle baseline
- Input: `uav_cars_v1.mp4`
- Detector confidence threshold: 0.05
- Track high threshold: 0.35
- Track low threshold: 0.10
- New-track threshold: 0.45
- Matching threshold: 0.80
- Track buffer: 30 frames
- Camera-motion compensation: sparse optical flow
- ReID: disabled

## Qualitative observations

1. The motorcycle was intermittently missed by the detector.
2. When it was detected again, BoT-SORT retained the same identity.
3. A persistent non-vehicle false detection became a confirmed track.
4. Real vehicles did not show unnecessary identity changes.
5. Camera movement did not visibly destabilise the tracks.

## Interpretation

The no-ReID baseline demonstrates that Kalman prediction, IoU association and
camera-motion compensation can bridge short detector misses.

However, the tracker cannot reject persistent detector false positives. A
temporally consistent false detection may therefore become a stable track.

## Baseline decision

The unchanged BoT-SORT no-ReID configuration is accepted as the first
qualitative tracking baseline.

Formal conclusions require an annotated MOT dataset and metrics such as HOTA,
AssA, IDF1, MOTA, identity switches and fragmentation.