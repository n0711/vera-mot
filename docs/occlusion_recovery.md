# Experimental recovery after detection gaps

The CPU `OnlineReID` associator retains an identity until the current frame minus
its last observation exceeds `max_age`. Reusing an ID after at least one missing
frame is now reported separately from ordinary consecutive-frame association.
A detection gap is not proof of physical occlusion: missed detections and leaving
the image can also cause gaps.

## Optional ambiguity guard

Pass `--recovery-margin 0.05` to `scripts/run_reid_detections.py` to try the guard.
The default is zero (disabled), preserving the existing association baseline.
The example value is provisional, not tuned or validated on UAVDT.

For every feasible lost-track/detection pair with cost c, require every competing
cost in its row and column to be at least c + margin. Row alternatives represent
other detections; column alternatives represent other retained identities,
including identities observed on the preceding frame. Compare against the
original feasible cost matrix before removing edges or performing Hungarian
assignment. No competitor means no additional restriction. Existing similarity,
displacement, maximum-cost and age gates still apply.

Ambiguous recovery edges are removed. The assignment may use another eligible
identity; otherwise the observation receives a new ID. A rejected lost identity
keeps its previous appearance and timestamp until recovered or expired. The
policy deliberately trades possible fragmentation for fewer arbitrary recovery
choices; it does not establish an improvement in IDF1 or ID switches. It also does
not guard consecutive-frame matches or guarantee that a unique match is correct.

## Outputs and reproducibility

`AssignedDetection.missed_frames` gives the number of missing frame indices before
a reused ID, and is zero for new IDs and consecutive matches. The runner's JSON
provenance includes `gap_recoveries`, `gap_recovery_events` (frame, track ID and
missed frames), and the supplied association parameters. `matched_existing`
retains its old meaning and includes both ordinary continuity and gap recovery.
MOT output remains ten columns. Events are hypotheses, not verified recoveries.

```bash
python scripts/run_reid_detections.py \
  --video /path/to/video.mp4 --detections /path/to/detections.csv \
  --output /path/to/result.txt --embedding hsv \
  --max-age 30 --recovery-margin 0.05
python -m pytest tests/test_online_reid.py -q
```

Synthetic tests cover distinct appearances after a gap, competing lookalike IDs,
competing detections, input-order reversal, an active competing identity, baseline
compatibility, expiry boundaries, and video-to-provenance output. HSV descriptors
are a diagnostic baseline, not a trained vehicle ReID model. This change applies
to the standalone associator, not the pinned BoT-SORT pipeline.

Next evidence step: use an annotated aerial validation clip and the actual
vehicle checkpoint with provenance, compare guard off/on using identical inputs,
and count correct recovery, wrong recovery and fragmentation by gap duration.
Keep threshold selection within validation sequences. No aerial accuracy, trained
checkpoint result, or BoT-SORT/VERA benchmark improvement is claimed here.

## Verification recorded 24 November 2026

Python 3.12 with `requirements-test.txt`: `python -m pytest -m 'not integration'
-q` passed 215 tests, with 9 integration tests deselected. `git diff --check`
passed. These counts exclude the separate M4.9 reference-comparison PR (#15).
