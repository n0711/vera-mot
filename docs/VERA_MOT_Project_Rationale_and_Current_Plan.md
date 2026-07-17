# VERA-MOT — Project Rationale and Current Implementation Plan

**Date:** 17 July 2026  
**Project:** VERA-MOT  
**Meaning:** Visual Embedding Reliability Assessment for Multi-Object Tracking  
**Primary domain:** Onboard UAV tracking of ground vehicles

---

## 1. What we are building

VERA-MOT is an independent research implementation for persistent multi-object tracking of ground vehicles in UAV video.

The immediate system objective is:

UAV video
   ↓
Vehicle detector
   ↓
Camera-motion-compensated multi-object tracker
   ↓
Vehicle appearance embeddings
   ↓
VERA reliability assessment
   ↓
Protected identity memory
   ↓
Stable vehicle track identities


The central problem is not only detecting a vehicle in each frame. The system must preserve the correct identity of each vehicle across:

- UAV camera movement
- small target size
- image blur
- partial and complete occlusion
- similar-looking vehicles
- crossings and close interactions
- temporary missed detections
- reappearance after a tracking gap

A detector can repeatedly find the same vehicle while the tracker still assigns it a new or incorrect identity. VERA-MOT therefore focuses on identity reliability, not only detection accuracy.

---

## 2. Why the project is separated into three layers

The wider concept contains three related but different systems:

BSc research contribution
        ↓
ADDITESS onboard perception product
        ↓
Future closed-loop UAV guidance integration


### 2.1 BSc research contribution

The thesis investigates one narrow and defensible question:

> Can a lightweight visual-embedding reliability mechanism reduce identity switches and improve target recovery in UAV vehicle video without introducing unacceptable onboard latency?

The thesis contribution is VERA itself: deciding whether an appearance embedding is reliable enough to influence association, update identity memory, or reactivate a lost track.

### 2.2 ADDITESS perception product

The industrial system is broader:


Camera
  ↓
Vehicle detector/classifier
  ↓
Camera-motion-compensated MOT
  ↓
Reliability-aware identity management
  ↓
Target selection
  ↓
High-rate selected-target tracking
  ↓
Heading and motion estimation
  ↓
Verified target-state interface


This product may later include military-vehicle classes, oriented boxes, target locking, heading estimation, deployment optimization, and integration with the ADDITESS UAV autonomy stack.

### 2.3 Closed-loop guidance

Flight-control commands, autonomous following, gimbal control, MAVLink command authority, and real closed-loop flight testing are not part of the thesis core. During the academic phase, tracker output may be demonstrated through logging, ROS 2, or simulation without giving the research tracker direct flight authority.

This separation prevents the BSc contribution from becoming an unmanageable full UAV autonomy project.

---

## 3. Why VERA is needed

A conventional ReID-assisted tracker assumes that the current appearance feature is useful. In aerial footage, this assumption often fails.


Vehicle crop
   ↓
Embedding extracted
   ↓
Crop is blurred, tiny, occluded, or incomplete
   ↓
Embedding is unreliable
   ↓
Wrong association or contaminated identity memory
   ↓
Identity switch / wrong reactivation


Detector confidence does not fully solve this problem. A detector can be confident that a crop contains a vehicle while the crop is still too blurred, small, or occluded to represent that vehicle's identity reliably.

VERA will estimate an appearance reliability score for each candidate observation. The initial design considers:

- detector confidence
- target pixel size
- image sharpness or blur
- temporal consistency with previously trusted embeddings
- consistency with predicted motion
- later, visibility or occlusion cues if justified by experiments

The intended operating logic is:


High reliability
    → appearance strongly influences association
    → identity memory may be updated

Medium reliability
    → appearance influence is reduced
    → long-term memory is protected

Low reliability
    → appearance is suppressed
    → motion and IoU dominate
    → track is marked uncertain


The exact score and thresholds must be derived experimentally rather than fixed by assumption.

---

## 4. Why BoT-SORT is the baseline

BoT-SORT is not being treated as the final solution or as a modern production tracker. It is being used as a controlled research baseline because it already exposes the components VERA needs to study:

- Kalman motion prediction
- camera-motion compensation
- ByteTrack-style high/low-confidence association
- optional ReID appearance embeddings
- track appearance memory
- modular association logic

This gives us clear intervention points:

Detection crop
   ↓
ReID embedding
   ↓
[VERA reliability assessment]
   ↓
Association cost
   ↓
[VERA memory-update gate]
   ↓
Stored track identity


Starting from an established baseline is necessary for academic comparison. Designing a new detector, tracker, ReID network, uncertainty model, and UAV controller simultaneously would make it impossible to isolate which component caused an improvement.

ByteTrack will remain the motion-and-IoU comparator. BoT-SORT without ReID will show the value of camera-motion compensation and improved motion handling. BoT-SORT-ReID will show the benefit and risk of appearance information. VERA-MOT will then be compared against all three.

---

## 5. Why we moved directly to vehicle tracking

The original BoT-SORT checkpoints downloaded during setup were trained for MOT17 pedestrian tracking. They were useful for confirming that the official files were accessible, but a pedestrian-video demonstration would not advance the vehicle research enough to justify the time.

The project therefore moved directly to the correct domain:

UAV vehicle data
   ↓
Vehicle-capable detector
   ↓
BoT-SORT vehicle baseline
   ↓
Vehicle appearance model
   ↓
VERA-MOT


The pedestrian checkpoints are retained only as reference artifacts. They are not part of the VERA-MOT experimental results.

---

## 6. Why UAVDT is the first dataset

UAVDT is being used as the primary baseline dataset because it provides UAV video with tracked road vehicles and includes conditions directly relevant to the research problem, such as moving viewpoints, small objects, occlusion, and different camera perspectives.

The dataset allows us to test the complete tracking problem rather than isolated vehicle detection. It provides sequences and identity annotations required to measure whether a tracker preserves the same vehicle ID over time.

The first dataset role is:

1. validate the data structure and annotations
2. run a vehicle detector on UAV frames
3. reproduce ByteTrack and BoT-SORT vehicle baselines
4. identify sequences where identity switches occur
5. inspect the crops and embeddings immediately before those failures
6. design and test VERA using measurable failure cases

VisDrone can later be used as an additional public aerial dataset for external validation. Public datasets are used first so that the academic results remain reproducible and independent of proprietary ADDITESS data.

---

## 7. Work completed on 17 July 2026

### Repository and project structure

- Created the `vera-mot` research repository.
- Connected the local repository to GitHub.
- Pinned the official BoT-SORT repository as a Git submodule.
- Created separate directories for source code, datasets, experiments, documentation, patches, and tests.
- Defined the research contract and thesis scope.

### Reproducible development environment

- Created an isolated Python 3.10 virtual environment.
- Configured VS Code to use the project interpreter.
- Installed the NVIDIA driver and verified the RTX 4060 Laptop GPU.
- Installed PyTorch 2.4.1 with CUDA 12.4 support.
- Verified that PyTorch can allocate and execute tensors on the GPU.
- Installed BoT-SORT and its required runtime dependencies.
- Built the YOLOX C++ extension successfully.

### Compatibility work

The original BoT-SORT/FastReID code targets an older Python/PyTorch stack. Minimal compatibility patches were applied without changing tracking logic:

- replaced deprecated `collections.Mapping` imports;
- replaced the removed `torch._six.string_classes` dependency;
- verified successful imports of YOLOX, BoT-SORT, OpenCV, NumPy, PyTorch, and CUDA.

The patch, exact Python package lock, and GPU audit were committed to GitHub so the environment can be reproduced.

### Checkpoints and reproducibility records

- Downloaded the original MOT17 YOLOX detector checkpoint.
- Downloaded the original MOT17 FastReID checkpoint.
- Recorded SHA-256 checksums for both files.
- Retained them as baseline references, not as vehicle experiment models.

### Current operation

- Created the UAVDT dataset directory structure.
- Started downloading the UAVDT archive from a mirror after the official Google Drive link reached its public quota.
- Dataset extraction and inspection remain pending until the download completes.

---

## 8. Current project state

```text
[PASS] Repository initialized and pushed
[PASS] BoT-SORT baseline pinned
[PASS] Python environment isolated
[PASS] RTX 4060 CUDA environment operational
[PASS] BoT-SORT and YOLOX import successfully
[PASS] Compatibility patch preserved
[PASS] Environment lock and GPU audit preserved
[PASS] Reference model checksums preserved
[IN PROGRESS] UAVDT download
[PENDING] UAVDT archive validation and extraction
[PENDING] Vehicle detector baseline
[PENDING] ByteTrack vehicle tracking run
[PENDING] BoT-SORT vehicle tracking run
[PENDING] BoT-SORT-ReID vehicle baseline
[PENDING] VERA reliability analysis
```

No VERA algorithm modification has been made yet. This is intentional. A valid research contribution requires a working, measured, unchanged baseline first.

---

## 9. Planned implementation phases

### Phase 1 — Reproducible aerial vehicle baseline

1. Validate and extract UAVDT.
2. Inspect image sequences and annotation format.
3. Build a dataset adapter without altering original data.
4. establish a vehicle-capable detector.
5. Run ByteTrack.
6. Run BoT-SORT without ReID.
7. Measure baseline tracking metrics.
8. Save configurations, logs, outputs, and runtime measurements.

Reason:VERA cannot be evaluated until the existing tracker works correctly on the target domain.

### Phase 2 — Vehicle appearance baseline

1. Select or train an aerial vehicle ReID representation.
2. Integrate embeddings into the BoT-SORT association path.
3. Establish BoT-SORT-ReID vehicle results.
4. Measure whether ReID helps or harms under different conditions.

Reason: VERA controls the use of appearance information. There must first be a functioning appearance baseline.

### Phase 3 — Failure analysis

For detections and tracks preceding identity switches, record:

- bounding-box size
- detector confidence
- sharpness
- occlusion/visibility indicators
- embedding similarity to trusted history
- motion prediction error
- association cost
- whether identity memory was updated

Reason: The reliability score should be derived from observed failure patterns, not chosen arbitrarily.

### Phase 4 — VERA association gating

Add reliability to the association stage only.


appearance distance
       ↓
weighted or suppressed by reliability
       ↓
association decision


Compare against the unchanged BoT-SORT-ReID baseline.

Reason: This isolates the effect of reliability-aware association.

### Phase 5 — Protected identity memory

Allow only reliable observations to update the long-term identity representation. Less reliable observations may support short-term matching but must not overwrite trusted memory.

Reason: A single corrupted embedding can damage future associations even after the bad frame has passed.

### Phase 6 — Occlusion recovery and track state

Introduce explicit operational states such as:

- verified
- uncertain
- lost
- recovered

Evaluate reactivation after short and long occlusions.

### Phase 7 — Ablation and onboard profiling

Measure each reliability component individually and in combination. Profile:

- FPS
- mean latency
- p50 and p95 latency
- GPU and CPU utilisation
- memory consumption
- dropped frames
- power and temperature where hardware permits.

Reason: The method must improve identity reliability without making onboard deployment impractical.

---

## 10. Experimental comparison

The minimum comparison should be:

| Configuration | Purpose |
|---|---|
| ByteTrack | Motion/IoU baseline |
| BoT-SORT | Camera-motion-aware baseline without ReID |
| BoT-SORT-ReID | Appearance-assisted baseline |
| VERA association only | Tests reliability-aware matching |
| VERA memory only | Tests protected identity memory |
| Full VERA-MOT | Tests the combined contribution |

The main metrics are:

### Tracking quality

- HOTA
- AssA
- IDF1
- MOTA
- identity switches
- track fragmentation

### Occlusion and recovery

- correct recovery rate
- incorrect recovery rate
- recovery latency
- identity retention after reappearance

### Runtime

- FPS
- mean, p50, and p95 latency
- CPU/GPU utilisation
- memory use
- dropped-frame rate
- power and temperature

---

## 11. What is deliberately outside the current scope

The following are not current implementation tasks:

- full military-vehicle detector development
- oriented vehicle heading estimation
- selected-target SOT
- gimbal control
- MAVLink command generation
- autonomous following control
- real closed-loop flight testing
- complete ADDITESS product deployment

These belong to the wider engineering roadmap and may consume VERA-MOT output later. Excluding them now protects the research question and keeps experimental results interpretable.

---

## 12. Immediate next actions after the UAVDT download

1. Verify archive size and checksum if available.
2. Inspect archive structure before extraction.
3. Extract into `datasets/UAVDT/extracted/`.
4. Identify sequence directories and MOT annotations.
5. Create `docs/UAVDT_DATASET_NOTES.md` describing the exact structure.
6. Select a small sequence for the first vehicle-only pipeline test.
7. verify the COCO or UAV-trained detector on that sequence.
8. Run the first tracker without ReID.
9. Record the command, configuration, runtime, and output location.
10. Only then begin the formal baseline evaluation.

---

## 13. One-paragraph project statement

VERA-MOT develops a reproducible UAV vehicle-tracking research baseline and a lightweight mechanism for assessing the reliability of visual identity embeddings. The method is intended to prevent blurred, occluded, undersized, or otherwise unreliable observations from corrupting detection-to-track association and long-term identity memory. It will be implemented within an established camera-motion-compensated tracker, evaluated on public UAV vehicle datasets, compared against ByteTrack and BoT-SORT variants, and profiled for onboard execution. Target selection, heading estimation, SOT, MAVLink control, and closed-loop UAV guidance remain separate ADDITESS engineering modules rather than the core academic contribution.
