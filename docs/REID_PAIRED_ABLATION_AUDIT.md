# M4.7 - ReID paired-run integrity audit (offline)

Status: **new data-free audit implementation**. It does not establish a vehicle
identity benchmark, a learned reliability model, or any tracker improvement.

## Purpose

Compare the *provenance and integrity* of paired, ungated HSV-ReID and
VERA-gated HSV-ReID runs from the existing pinned BoT-SORT runner. This is a
precondition for an eventual identity-annotated ablation, not a substitute for
it. The audit reads only existing MOT rows and their companion provenance JSON.
It does not read videos, extract embeddings, send data, or change the tracker.

## Fixed-input protocol

1. Use the same authorized video and detector CSV in both runs.
2. Use identical camera-motion compensation and embedding model/weights.
3. Run ungated hsv-reid, then vera-hsv-reid on the same inputs and environment.
4. Confirm both runner commands succeeded and produced MOT and provenance
   files. Run the audit with explicit paths:

    python scripts/audit_vera_reid_pair.py \
      --ungated experiments/reid_online/synthetic_demo/botsort_hsv_reid.txt \
      --gated experiments/reid_online/synthetic_demo/botsort_vera_hsv_reid.txt \
      --output experiments/reid_online/synthetic_demo/paired_audit.json

The audit refuses a mismatched input video or detection-file SHA-256,
different embedding configuration/weights, CMC method, frame count, or
embedding extraction counters. It also fails on malformed MOT rows, duplicate
track IDs within a frame, invalid gate counters, and mismatched provenance
observation/identity counts. Zero-track runs are valid.

The output includes input fingerprints, output/provenance file SHA-256 values,
quality-gate counters, observation counts, tentative track counts, and track
continuity gap counts per run. The JSON is deterministic for unchanged inputs,
and does not copy source frames or raw detections.

## Interpretation and failure boundaries

- A track ID is arbitrary within each run, so numeric IDs are **not** matched
  directly between runs. Different counts/gaps do not demonstrate improvement.
- Track gaps are changes in consecutive observed frame indices per *reported*
  ID, not ground-truth occlusion or identity recovery.
- Provenance hashes establish consistency between *reported* run metadata, not
  the authenticity of the original video, dataset, or trained weights.
- The quality score is an uncalibrated size/focus/in-frame proxy, not a
  probability of correct identity. This audit cannot calculate IDF1, HOTA,
  AssA, IDSW, association accuracy, or false acceptance/rejection rates.
- No new detector training, ReID model execution, GPU results, UAV deployment,
  robot actuation, or flight-control integration is performed here.

## To graduate to identity-based ablations

Acquire authorized identity-annotated held-out sequences without violating the
frozen UAVDT test partition. Predeclare thresholds and tracking settings,
execute no-ReID / ungated-ReID / VERA gated-ReID on identical detections and
camera-motion settings, and evaluate with a consistent ground-truth alignment
and TrackEval. Only then report IDF1, AssA, HOTA, IDSW and uncertainty with
actual runtimes and failure cases. Keep synthetic checks and real experiments
in separate evidence categories.

## Verification

    python -m pytest -q -m 'not integration' tests/test_reid_ablation_audit.py

This CPU-only test uses invented MOT/provenance fixtures. Pinned BoT-SORT
runtime execution and annotated vehicle identity evaluation remain separate,
unverified evidence gates.
