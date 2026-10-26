# VERA-MOT

VERA-MOT (Visual Embedding Reliability Assessment for Multi-Object Tracking)
is a project investigating whether unreliable vehicle ReID
embeddings can be prevented from corrupting association and long-term identity
memory in UAV video.

## Status

| Stage | State |
| --- | --- |
| YOLOX-S single-class vehicle detector on UAVDT | done, held-out test AP 50.1, AP50 83.5 |
| BoT-SORT without ReID, 20 UAVDT test sequences (Milestone 3.1) | done and frozen |
| UAVDT ignore-region filtering and TrackEval evaluation | done |
| Vehicle ReID data pipeline and training lifecycle (Milestone 4) | implemented; local training evidence is not distributed |
| Sequence-local temporal ReID evaluation and crop-quality analysis | implemented; benchmark execution evidence pending |
| Six-checkpoint validation comparison with verified resume | implemented; final epoch is the declared primary checkpoint |
| BoT-SORT vehicle-ReID tracking baseline | not yet established in the tracked evidence |
| ByteTrack, VERA association gating, protected memory, ablations | not implemented |
| ROS 2 or edge deployment | not implemented |

The frozen 20-sequence no-ReID result is HOTA 43.340, DetA 36.441, AssA 52.419,
IDF1 56.964, 1,180 ID switches, MOTA 31.265, MOTP 74.136, precision 67.048,
and recall 62.161. See
`experiments/tracking_baselines/botsort_no_reid/manifest.yaml`. The detector
numbers are in `experiments/detector_evaluation/uavdt_test_summary.txt`.

The training entry point is `scripts/train_uavdt_fastreid.py`. A completed local
60-epoch run was reported during project work, but its `completion.json` and
checkpoints are absent from this repository. This status distinguishes available
implementation from verified experiment evidence. Dataset QA and initialization
provenance remain prerequisites for a new training run. The historical design,
split and leakage rules are in
[`docs/reid_baseline_design.md`](docs/reid_baseline_design.md).

This repository is partially reproducible. Code, revisions, configuration,
checksums, and compact evidence are preserved, but UAVDT and the trained
detector checkpoint are not distributed.

## Repository layout

- `baselines/`: pinned BoT-SORT, YOLOX, and TrackEval submodules.
- `configs/`: YOLOX training, smoke-test and held-out-test configurations, the
  ReID sequence split, and the FastReID UAVDT config.
- `datasets/`: ignored local datasets. Only the expected layouts are documented.
- `docs/`: research scope, environment audits, baseline status, and the ReID
  design.
- `experiments/`: frozen manifests and compact canonical evidence. Generated
  results stay ignored.
- `patches/`: compatibility changes for pinned upstream code.
- `scripts/`: dataset, detector, tracking, filtering, evaluation, and ReID data
  entry points.
- `src/vera_mot/`: project-owned code (MOT file I/O, ReID manifests and crops,
  the FastReID UAVDT adapter).
- `tests/`: CPU-only, data-free unit tests.

Where to read next:

- [`docs/VERA_MOT_Project_Rationale_and_Current_Plan.md`](docs/VERA_MOT_Project_Rationale_and_Current_Plan.md): why the project exists and how it is staged
- [`docs/research_contract.md`](docs/research_contract.md): the research questions and rules
- [`docs/current_experiment.md`](docs/current_experiment.md): the frozen baseline in one page
- [`docs/detector_baseline_status.md`](docs/detector_baseline_status.md) and [`docs/botsort_no_reid_baseline_status.md`](docs/botsort_no_reid_baseline_status.md): baseline details and limitations

## Optional ROS3-Limassol and external-VLM research path

Read-only `src/vera_mot/vision_evidence.py` exports aggregate evidence from
existing, single-sequence MOT results. `src/vera_mot/vlm_contract.py` validates
a **declared** external VLM annotation manifest; it does not run or verify a
vision-language model. Neither module sends commands, publishes video, or
alters the frozen benchmark. See [ROS3 and VLM integration
protocol](docs/ROS3_VLM_INTEGRATION.md) for commands, limits and provenance.

## Environment and submodules

The recorded environment was Ubuntu 22.04, Python 3.10.12, PyTorch
2.4.1+cu124, torchvision 0.19.1+cu124, NumPy 1.26.4, and OpenCV 4.11.0.86.
CUDA execution used an RTX 4060 Laptop GPU. CUDA is required for the recorded
FP16 runtime, but not for the unit tests.

```bash
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch 2.4.1/torchvision 0.19.1 CUDA wheels first.
python -m pip install -r requirements-baseline.txt
./scripts/apply_botsort_compatibility_patch.sh
```

The scripts prepend the pinned submodules to `sys.path`, so no machine-local
editable install is required. For interactive imports:

```bash
export PYTHONPATH="$PWD/src:$PWD/baselines/YOLOX:$PWD/baselines/BoT-SORT:$PWD/baselines/TrackEval"
```

`requirements-baseline.txt` describes the focused baseline environment, and
`requirements-baseline-lock.txt` pins it exactly. `requirements-vera.txt` is
empty because VERA-specific dependencies do not exist yet.

Run the CPU project tests without datasets, checkpoints, submodules or PyTorch:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q -m 'not integration'
```

The separate FastReID configuration integration check requires the initialized
BoT-SORT submodule and its PyTorch/FastReID dependencies. Run it in the recorded
baseline environment with `python -m pytest -q -m integration`. CI runs the CPU
suite; it does not run training, GPU inference, or the benchmark.

## ReID validation and crop-quality analysis

The dedicated temporal evaluator preserves sequence-local identities and compares
each query only against its own sequence's gallery. It rejects a modified frozen
validation registry. Every output labels bounded smoke runs separately from full
validation; no frozen-test data is used for checkpoint selection or thresholds.

With the six trained checkpoints and frozen crop data present, inspect the plan
without loading a model or writing results:

```bash
python scripts/evaluate_uavdt_reid_checkpoints.py --plan-only
```

Run all six checkpoints, or resume verified completed evaluations:

```bash
python scripts/evaluate_uavdt_reid_checkpoints.py --verify-crops
python scripts/evaluate_uavdt_reid_checkpoints.py --verify-crops --resume
```

`model_final.pth` is the predeclared primary checkpoint. Earlier checkpoints are
diagnostics; the sweep does not pick a winner. All runs must use the same temporal
split. Input fingerprints cover weights, registry, split, inherited configuration,
evaluator/upstream source and runtime package versions. Resume checks hashes of
all six output artifacts. Partial or mismatched runs stop with an error.

The comparison is written to
`experiments/reid_evaluation/checkpoints/comparison.json`. The raw crop-quality
analysis can be run independently on train or validation manifests; it defines
neither a reliability score nor gating thresholds. See the
[temporal protocol](docs/reid_temporal_evaluation_protocol.md),
[crop-quality protocol](docs/reid_crop_quality_protocol.md) and
[evaluation workflow](docs/reid_evaluation_workflow.md).

## Data and weights

Place the UAVDT detection subset as:

```text
datasets/UAVDT/yolo_detection_subset/{train,val,test}/{images,labels}
```

Place the full MOT data as:

```text
datasets/UAVDT/full_mot/raw/UAV-benchmark-M/<sequence>/img*.jpg
datasets/UAVDT/full_mot/toolkit/UAV-benchmark-MOTD_v1.0/GT/
```

Convert the YOLO detection labels to the single COCO `vehicle` category:

```bash
python scripts/convert_uavdt_yolo_to_coco.py
```

The exact public source and checksum of the locally used UAVDT archives and the
derivation of the 1,266/271/272 detection split are not yet recorded. Obtain
UAVDT from an authorized source and comply with its terms. UAVDT is not covered
by the source-code licenses in this repository.

The tracker expects the trained detector at:

```text
experiments/detector_runs/yolox_uavdt_s_1280/best_ckpt.pth
```

Its frozen SHA-256 is
`91b8c13682a3b74cdd74d45edc83899cfb8be367665f85c942c761a78189ceef`.
The checkpoint is not distributed, and its public acquisition location is not
known. Verify any supplied copy before use.

## Detector commands

The recorded training used one GPU, batch size 2, FP16, a COCO-pretrained
YOLOX-S initialization, and the 50-epoch configuration:

```bash
cd baselines/YOLOX
PYTHONPATH=. ../../.venv/bin/python tools/train.py \
  -f ../../configs/yolox_uavdt_s_1280.py \
  -d 1 -b 2 --fp16 \
  -c /path/to/yolox_s_coco.pth
cd ../..
```

The initial checkpoint URL/checksum and a training seed were not recorded, so
this command documents the execution shape rather than guaranteeing identical
weights. Evaluate the held-out detector with:

```bash
bash scripts/evaluate_yolox_uavdt.sh
```

Expected local output includes
`experiments/detector_evaluation/yolox_uavdt_s_1280_test/`. The compact frozen
summary is `experiments/detector_evaluation/uavdt_test_summary.txt`.

## Frozen no-ReID tracking

Run one sequence with the frozen defaults:

```bash
python scripts/run_uavdt_botsort_sequence.py --sequence M0203 --fp16
```

This writes MOT-format output to
`experiments/tracking_baselines/botsort_no_reid/uavdt/M0203.txt`.

Run or resume the full 20-sequence workflow:

```bash
bash scripts/finish_uavdt_no_reid.sh
```

Existing results are reused only if both their content checksum and frozen
configuration fingerprint match `result_registry.json`. A mismatch stops
without deleting anything. `--force` explicitly authorizes regenerating a
mismatched sequence:

```bash
bash scripts/finish_uavdt_no_reid.sh --force
```

The full script performs tracking, then the equivalent explicit stages are:

```bash
python scripts/filter_uavdt_ignore_regions.py
python scripts/evaluate_uavdt_trackeval_all.py
```

Filtered outputs are written under `uavdt_official/` and TrackEval outputs
under `metrics/uavdt_official/`. The custom ignore filter uses valid-GT IoU
0.50 and unmatched-prediction ignore-region IoA 0.50. Its equivalence to the
official UAVDT MATLAB protocol has not been regression-tested.

## Licensing and limitations

BoT-SORT and TrackEval are MIT-licensed, YOLOX is Apache-2.0 licensed. Their
license files remain in the pinned submodules. Dataset and checkpoint licenses
must be assessed separately.

The metrics above are benchmark results. Private-video notes are qualitative
only: a temporary miss followed by the same displayed ID is not proof of
ground-truth identity recovery. Military/domain-shift footage has no
quantitative annotations in this repository and is not part of the frozen
benchmark.

## Support this project

If this project is useful to you, you can support its development with a one-off contribution via Revolut:

**[revolut.me/nadiotis](https://revolut.me/nadiotis)**
