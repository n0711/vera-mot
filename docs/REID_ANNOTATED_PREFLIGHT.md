# M4.8 - Read-only annotated UAVDT validation preflight

**Evidence status:** implementation and synthetic CPU tests only. This
milestone verifies prerequisites for *future* identity-based comparisons. It
does not calculate TrackEval metrics or execute an annotated vehicle benchmark.

## Scope and research contract

M4.7 checks two reported BoT-SORT MOT/provenance outputs for paired
comparability, but its video/detection hashes are originally self-reported.

M4.8 adds a stricter, offline prerequisite check:

1. Require a sequence from the existing six-sequence ReID *validation* split.
   The immutable 20-sequence test set is refused, and the split lists must
   remain disjoint. The official sequence inventory must identify it as an
   original UAVDT train partition member.
2. Require the original 9-column UAVDT source GT filename and validate positive
   frame/identity indices, finite coordinates/codes, positive box dimensions,
   nonduplicated identities per frame, and frame bounds. Source codes in
   columns 7-9 remain raw and uninterpreted. This is **not** MOTChallenge GT
   conversion.
3. Match the GT row count and declared sequence length to the committed
   UAVDT audit (source audit is a recorded inventory, not independent proof
   of dataset authenticity).
4. Call the M4.7 paired audit to check provenance, output rows, gated-vs-ungated
   settings and matched reported video/detection hashes.
5. Rehash the **actual available video and detection CSV bytes** and require
   each to match the reported SHA-256 in both runs. Refuse a changed source.
6. Require an explicitly predeclared gate threshold and reject mismatches.
   Write a *new* deterministic aggregate JSON report with input/evidence
   hashes and GT row/identity counts, without copying raw annotations,
   frames or detections. The CLI uses exclusive creation, not overwrite.

The preflight does not decode or inspect video frames, verify timestamps, match
actual image content to annotation identities, establish dataset rights, validate
any trained model, or claim improvements. A valid source-byte hash shows only
the source file matches the run's recorded bytes.

## Invocation (once original data is available)

Use existing authorized files and prior BoT-SORT outputs for the **same**
sequence. The video and detection CSV must be the exact files supplied to both
runs; the GT must be the original 9-column UAVDT file. A single example:

~~~bash
python scripts/preflight_vera_reid_validation.py \
  --sequence M0201 \
  --ground-truth /authorized/UAVDT/GT/M0201_gt.txt \
  --video /authorized/UAVDT/videos/M0201.mp4 \
  --detections /authorized/UAVDT/detections/M0201.csv \
  --ungated /results/M0201_hsv.txt \
  --gated /results/M0201_vera.txt \
  --expected-threshold 0.5 \
  --output /results/M0201_preflight.json
~~~

Do not interpret the passed_prerequisite_consistency_only status as passing
identity evaluation. If GT was intentionally converted/filtered or contains a
different number of rows than the original recorded source, the check fails
closed. A separate, explicitly documented conversion and verification protocol
is required before using such data.

## Evidence categories

| Category | Status |
| --- | --- |
| Frozen no-ReID benchmark | Existing historical evidence, unchanged |
| Dataset-free paired audit (M4.7) | Implemented, GitHub CPU and pinned-tracker CI |
| M4.8 strict source/annotation preflight | Implemented with synthetic fixtures; see latest CI |
| Authorized real validation GT and paired video/detection files | Not present in repository |
| Full ReID, gated ReID, identity metrics on real annotated validation sequences | Blocked |
| TrackEval adaptation for source nine-column UAVDT GT | Separate unimplemented validation task |
| Held-out test/field evaluation | Not attempted in this milestone |

## Next acceptance gates

- Obtain and independently record licensed source archive fingerprints for the
  six validation sequences, the original per-sequence GT and source videos.
- Predeclare fixed detector inputs, CMC parameters, threshold(s), evaluation
  subset, annotation conversion, ignore-region rules and seed.
- Run no-ReID, ungated ReID and gated ReID under matched conditions. For an
  accuracy claim, evaluate with verified GT conversion and unchanged TrackEval
  settings. Report IDF1, AssA, HOTA, IDSW, runtime, failure cases and confidence
  intervals appropriately. No test data should be used for threshold tuning.

## Synthetic regressions (no model, video or UAVDT required)

~~~bash
python -m pytest -q -m 'not integration' tests/test_reid_eval_preflight.py
~~~

Fixtures use a 3-frame **fabricated** UAVDT-style manifest and invented track
rows plus arbitrary placeholder video bytes. They exercise strict validation
and source-byte matching only. These values are not scientific outcomes.
