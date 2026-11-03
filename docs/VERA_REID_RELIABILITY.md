# M4.6 — Quality-gated appearance association and protected feature memory

Status: implemented experimental milestone with synthetic CPU tests only. This
is **not** an empirically calibrated vehicle ReID reliability score, an
occlusion estimator, or a benchmark result.

## Hypothesis

An appearance observation that is too small, blurred, or partially outside
the image should not contribute to identity matching or appearance memory.
Geometry-only matching remains permitted.

The gate extends PR #7's pinned BoT-SORT bridge; the original benchmark,
historical metrics, pinned submodule revision and detector remain unchanged.

## Heuristic definition

For original xyxy area A_raw and in-frame visible area A_vis:

- q_size = clip(A_vis / 1024, 0, 1)
- q_focus = clip(Var(Laplacian(gray crop)) / 120, 0, 1)
- q_in_frame = clip(A_vis / A_raw, 0, 1)
- q = min(q_size, q_focus, q_in_frame)
- allow appearance only if q >= 0.5 (provisional)

The 1024-pixel area, 120 Laplacian-variance normalization and 0.5 threshold
are initial engineering choices, not values tuned on annotated vehicle data.
In-frame area is a *boundary truncation* proxy, not visibility under an
occluder; focus is sensitive to scale, compression and lighting. Lack of
texture does not prove a detection is untrustworthy.

## Mechanism

An optional appearance policy runs on the upstream high-score detections
after embedding extraction. Each detection either keeps its embedding or
is constructed in STrack with no appearance feature.

The vetted upstream matching patch assigns a distance of 2.0 to a pair
with missing features. BoT-SORT divides it by two, so that appearance pair
is rejected and the existing IoU path remains available. For an otherwise
matched low-quality detection, STrack.update does not add a memory feature.
A later quality-accepted detection can update memory again.

The patch does not alter BoT-SORT's Kalman filter, IoU, association thresholds
or tracking state transitions. Appearance extraction is still done before
quality filtering; do not claim lower compute cost.

## Install on local Ubuntu

On the new development branch, with the existing Python 3.10 venv active:

    cd ~/vera-mot
    git submodule update --init --recursive
    bash scripts/apply_botsort_compatibility_patch.sh
    bash scripts/apply_botsort_vera_patch.sh
    python -m pytest -q -m 'not integration'
    python -m pytest -v -m integration tests/test_vera_reid_quality_gate.py

The VERA patch installer ensures the previous ReID and lazy import patches
are installed. It checks the exact known Git blob hashes of the pre-VERA
tracker files; unrelated local modifications are rejected, never overwritten
or reset. It remains opt-in and does not move the submodule pin.

## Side-by-side synthetic smoke experiment

    python scripts/make_reid_demo.py

    python scripts/run_botsort_reid_detections.py \
      --video experiments/reid_online/synthetic_demo/synthetic_vehicle_demo.mp4 \
      --detections experiments/reid_online/synthetic_demo/synthetic_vehicle_detections.csv \
      --mode hsv-reid \
      --output experiments/reid_online/synthetic_demo/botsort_hsv_reid.txt

    python scripts/run_botsort_reid_detections.py \
      --video experiments/reid_online/synthetic_demo/synthetic_vehicle_demo.mp4 \
      --detections experiments/reid_online/synthetic_demo/synthetic_vehicle_detections.csv \
      --mode vera-hsv-reid --quality-threshold 0.5 \
      --output experiments/reid_online/synthetic_demo/botsort_vera_hsv_reid.txt

The accompanying provenance JSON includes the heuristic gate configuration,
accepted/blocked counts and mean proxy score, not image crops or ground truth.
Synthetic rectangles are deliberately not an identity benchmark.

## Acceptance and next validation

CPU tests cover sharp vs featureless crops, truncated/small detections,
invalid inputs, shape contracts and reject-without-mutation behavior. Real
pinned BoT-SORT integration tests must show an unchanged feature-memory
vector after a blocked detection and resumed updates when accepted.

A later identity-annotated experiment must compare ungated/gated association
under fixed detections and CMC assumptions, and measure IDF1, AssA, ID switches,
false rejection, occlusion gaps and runtime. No HOTA, IDF1, or identity
recovery improvement can be claimed from this synthetic slice.
