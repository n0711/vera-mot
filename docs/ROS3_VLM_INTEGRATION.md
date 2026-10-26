# VERA-MOT as visual evidence for ROS3-Limassol and external VLMs

**Status: experimental offline contracts, not deployed ROS/VLM inference.**

VERA-MOT is a tracking/re-identification research repository; it is not itself
a vision-language model. The frozen BoT-SORT no-ReID benchmark is not an
evaluation of a VLM or the proposed reliability-gated VERA method.

The new source-only adapters make a narrowly scoped, read-only research path:

1. VERA-MOT produces MOTChallenge tracker rows from an authorized video.
2. `vision_evidence.py` summarizes the completed sequence without exporting
   boxes, images, per-track identifiers, location or a reconstructed trajectory.
3. A **separately installed/operated** VLM may annotate selected source frames.
   `vlm_contract.py` validates the declared inference manifest and retains
   aggregate semantic label counts. This repo does **not** run a VLM or validate
   its captions/labels against ground truth.
4. ROS3-Limassol `multimodal.py` can place both evidence reports alongside a
   telemetry receiver-observation report. It explicitly states that clock and
   robot-to-sequence alignment are **unverified**.

## Produce VERA-MOT track evidence

From the VERA-MOT repository root (Python 3.10+), with an existing single
sequence MOT-format result, e.g. from the existing baseline workflow:

```sh
PYTHONPATH=src python -m vera_mot.vision_evidence \
  experiments/tracking_baselines/botsort_no_reid/uavdt/M0203.txt \
  --sequence M0203 > M0203-vision-evidence.json
```

The exporter parses 10-column rows with monotonically increasing frame numbers,
positive integer frame/track IDs and finite row values. It records the exact
SHA-256 of the input, total output rows, sequence-local distinct track IDs,
frames **containing tracker output**, and their frame range. These are **not**
ground-truth identity or tracking accuracy measurements. Gaps between frame
numbers do not establish camera dropouts or loss of video.

It fails before printing JSON if the file is invalid, and restricts rows and
unique IDs to bounded limits. It does not access cameras or write detections.

## External VLM inference manifest v1

The VLM must be run separately with legitimately obtained frames and an
explicitly identified model/revision/prompt. Its orchestrator can emit:

```json
{
  "annotation_version": 1,
  "sequence": "M0203",
  "model_id": "research/example-vlm",
  "model_revision": "local-revision1",
  "prompt_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "frames": [
    {
      "frame": 1,
      "image_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
      "labels": ["vehicle", "water"]
    }
  ]
}
```

The hashes above are **illustrative placeholders**, not evidence. For actual
inference record the exact image bytes and model/prompt revision. Frame numbers
are sequence-local; there are no assumptions of fixed FPS, synchronized
timestamps, GNSS availability or vehicle localization.

```sh
PYTHONPATH=src python -m vera_mot.vlm_contract \
  inferred-annotations.json > M0203-vlm-evidence.json
```

The output omits frame hashes and raw video/captions, retaining the source
manifest's byte hash, declared model provenance and bounded label histogram.
Validation establishes the *format only*. It cannot attest that the model
really ran, that the input images match those hashes, or that labels are true.

## Bundle evidence offline with ROS3

From the ROS3-Limassol `robotics/ros3_limassol/` folder:

```sh
python -m ros3_limassol.multimodal \
  /path/to/ros3-evidence.json /path/to/M0203-vision-evidence.json \
  --vlm /path/to/M0203-vlm-evidence.json > multimodal-evidence.json
```

The VLM argument and anomaly model argument are optional. The bundle contains
**parallel evidence**, not synchronized cross-modal localization or validated
sensor fusion. It is not a navigation/planning/control input.

## Research evaluation required before claims

- Freeze the train/validation/test split before selecting detector, tracker,
  ReID, VLM prompts or anomaly thresholds. Do not tune on the held-out UAVDT test.
- Verify camera calibration and timestamp origin. Record ROS receiver monotonic
  arrival times separately from image capture times; establish independent
  clock synchronization and error bounds if temporal fusion is later required.
- Annotate known outage/recovery and GNSS-denied scenarios with independent
  ground truth. Compare against a no-vision baseline and tracking-only baseline.
- Report HOTA, IDF1, identity switches, ReID validation, VLM label precision
  and runtime separately. Do not attribute correlation to GNSS robustness.
- Test consent, privacy, dataset permissions and retention policies for video,
  location traces and outputs before sharing evidence.

No military use claims or guidance capability are established by this work.
