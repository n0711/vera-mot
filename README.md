# VERA-MOT

VERA-MOT (Visual Embedding Reliability Assessment for Multi-Object Tracking)
is a BSc research project investigating whether unreliable vehicle ReID
embeddings can be prevented from corrupting association and long-term identity
memory in UAV video.

## Current milestone

Milestone 3.1 freezes the executed UAVDT BoT-SORT no-ReID baseline. The
repository currently implements a YOLOX-S single-class vehicle detector,
BoT-SORT tracking without ReID, UAVDT ignore-region filtering, and TrackEval
evaluation. ByteTrack experiments, vehicle ReID, VERA association gating,
protected memory updates, ablations, and ROS 2/edge deployment are not
implemented.

The frozen 20-sequence result is HOTA 43.340, DetA 36.441, AssA 52.419,
IDF1 56.964, 1,180 ID switches, MOTA 31.265, MOTP 74.136, precision 67.048,
and recall 62.161. See
`experiments/tracking_baselines/botsort_no_reid/manifest.yaml`.

This repository is partially reproducible: code, revisions, configuration,
checksums, and compact evidence are preserved, but UAVDT and the trained
detector checkpoint are not distributed.

## Repository layout

- `baselines/`: pinned BoT-SORT, YOLOX, and TrackEval submodules.
- `configs/`: YOLOX training, smoke-test, and held-out-test configurations.
- `datasets/`: ignored local datasets; only the expected layouts are documented.
- `docs/`: research scope, environment, and baseline status.
- `experiments/`: frozen manifests and compact canonical evidence; generated
  results remain ignored.
- `patches/`: compatibility changes for pinned upstream code.
- `scripts/`: dataset, detector, tracking, filtering, and evaluation entry points.
- `src/vera_mot/`: project-owned reusable utilities.
- `tests/`: CPU-only, data-free unit tests.

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

The scripts prepend the pinned submodules to `sys.path`; no machine-local
editable install is required. For interactive imports:

```bash
export PYTHONPATH="$PWD/src:$PWD/baselines/YOLOX:$PWD/baselines/BoT-SORT:$PWD/baselines/TrackEval"
```

`requirements-baseline.txt` describes the focused baseline environment.
VERA-specific dependencies do not exist yet.

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
`experiments/detector_evaluation/yolox_uavdt_s_1280_test/`; the compact frozen
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

Filtered outputs are written under `uavdt_official/`; TrackEval outputs are
under `metrics/uavdt_official/`. The custom ignore filter uses valid-GT IoU
0.50 and unmatched-prediction ignore-region IoA 0.50. Its equivalence to the
official UAVDT MATLAB protocol has not been regression-tested.

## Licensing and limitations

BoT-SORT and TrackEval are MIT-licensed; YOLOX is Apache-2.0 licensed. Their
license files remain in the pinned submodules. Dataset and checkpoint licenses
must be assessed separately.

The metrics above are benchmark results. Private-video notes are qualitative
only: a temporary miss followed by the same displayed ID is not proof of
ground-truth identity recovery. Military/domain-shift footage has no
quantitative annotations in this repository and is not part of the frozen
benchmark.
