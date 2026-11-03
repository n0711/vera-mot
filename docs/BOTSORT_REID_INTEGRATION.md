# M4.5 — Pinned BoT-SORT ReID integration without UAVDT

## Scope

The project-owned appearance encoder is injected into the real pinned BoT-SORT
association path using external video and detection CSV, without the historical
UAVDT data, YOLOX checkpoint, or trained vehicle ReID weights.

The frozen no-ReID benchmark scripts, configurations, and scores remain
unchanged. Any synthetic or personal-video outputs are tentative tracker IDs,
not validated identity accuracy, HOTA, IDF1, AssA or VERA results.

## Why a local compatibility patch is necessary

Pinned BoT-SORT revision 251985436d6712aaf682aaaf5f71edb4987224bd
calls STrack.update_features before creating the feature-history deque, which
breaks when ReID is enabled. The tracker and matcher also contain np.float,
which modern NumPy removed. The opt-in patches/botsort-reid-compat.patch fixes
only these two compatibility defects; it does NOT change association logic.

The adapter creates BoTSORT with with_reid=False and attaches an explicitly
validated project encoder, then switches the instance to with_reid=True.
This avoids the upstream FastReIDInterface attempting to use missing weights.
Upstream IoU association, motion filtering, ReID cost and lost-track state
remain intact. The predecessor standalone OnlineReID is a different tracker.

## Install runtime dependencies on Ubuntu WSL

The lightweight CPU unit test environment does not include pinned BoT-SORT
runtime requirements. Use your existing Python 3.10 virtual environment:

~~~bash
cd ~/vera-mot
source .venv/bin/activate
git submodule update --init --recursive
uv pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-baseline.txt
bash scripts/apply_botsort_compatibility_patch.sh
bash scripts/apply_botsort_reid_patch.sh
python -m pytest -q -m 'not integration'
python -m pytest -q -m integration tests/test_botsort_reid_bridge.py
~~~

The CPU torch wheels are for a compatibility-first run; they do not enable
CUDA acceleration. A compatible GPU runtime may be installed separately and
verified afterwards. Do not assume that 6 GB CUDA is already validated.

The ReID patch is opt-in and idempotent. It fails if submodule revision is
wrong or the two tracker files have unrelated local edits. Do not force-apply.

## Paired synthetic trial with real BoT-SORT logic

~~~bash
python scripts/make_reid_demo.py

python scripts/run_botsort_reid_detections.py \
  --video experiments/reid_online/synthetic_demo/synthetic_vehicle_demo.mp4 \
  --detections experiments/reid_online/synthetic_demo/synthetic_vehicle_detections.csv \
  --mode no-reid \
  --output experiments/reid_online/synthetic_demo/botsort_no_reid.txt

python scripts/run_botsort_reid_detections.py \
  --video experiments/reid_online/synthetic_demo/synthetic_vehicle_demo.mp4 \
  --detections experiments/reid_online/synthetic_demo/synthetic_vehicle_detections.csv \
  --mode hsv-reid \
  --output experiments/reid_online/synthetic_demo/botsort_hsv_reid.txt \
  --render experiments/reid_online/synthetic_demo/botsort_hsv_reid.mp4
~~~

Inputs have columns frame,x1,y1,x2,y2,score (1-based frame, xyxy pixels).
Both outputs are ten-column MOT files with a companion provenance JSON. The
synthetic clip uses colored rectangles with a two-frame disappearance, not
annotated vehicle videos or benchmark identities.

## Operating assumptions and limitations

- no-reid: pinned tracker with appearance disabled
- hsv-reid: pinned tracker with a 640-D weight-free appearance histogram
- resnet18-reid: optional 512-D generic ImageNet features from explicit local
  torchvision weights; not trained vehicle ReID and not GPU-tested here
- Empty detection frames return an empty appearance matrix safely.
- Nonnumeric, zero, nonfinite, or wrong-dimensional embeddings fail closed.
- ReID uses upstream appearance_thresh=0.25 and proximity_thresh=0.5, neither
  of which was calibrated on real vehicles.
- CMC defaults to none for the synthetic demo. The historical experiment used
  sparseOptFlow; set that explicitly in a future controlled comparison.
- Identity output remains a hypothesis until checked against annotated labels.
- VERA reliability gating and protected memory are still not implemented.

## Next milestone

Obtain authorized annotated vehicle detections to measure identity continuity
and switches, validate a learned vehicle ReID encoder, and compare pinned
BoT-SORT without/with ReID on identical frames and thresholds. Then develop
VERA reliability gating as a separate experimental change.
