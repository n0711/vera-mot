# Online vehicle appearance matching without UAVDT

The historical UAVDT images, detector checkpoint, and trained FastReID weights
are unavailable. This experimental slice implements appearance-conditioned
association of externally supplied vehicle detections without those artifacts.

This is a CPU-runnable **engineering prototype**, not a validated vehicle-
trained deep ReID model, a BoT-SORT-ReID benchmark, or the VERA reliability
algorithm. Existing frozen baseline code and metrics remain unchanged.

## Implementation

- **HSVHistogramEmbedder:** 640-dimensional unit-normalized HSV histogram
  over global and four spatial cells. No learned weights or PyTorch.
- **OnlineReID:** online appearance-based track matching, one-to-one Hungarian
  assignment, cosine similarity and IoU costs, position-displacement gating,
  bounded track expiry, and unguarded exponential feature averaging.
- **ResNet18Embedder:** optional 512-D generic ImageNet features from a
  locally supplied torchvision ResNet18 state dictionary. No automatic
  downloads, random initialization, or hidden model fallback.
  These generic image features are **not** vehicle-ReID-trained weights.
- **run_reid_detections.py:** video + frame-indexed bounding-box CSV ->
  MOT 10-column associations, companion provenance JSON, and optional
  annotated MP4. IDs are algorithmic hypotheses, not ground truth.
- **make_reid_demo.py:** generated synthetic red/blue rectangles with a
  two-frame gap for reproducible CPU-only smoke testing.

## Run now without an external dataset

Run under the existing Python 3.10 virtual environment with
requirements-test.txt installed.

~~~bash
cd ~/vera-mot
source .venv/bin/activate
python -m pytest -q -m 'not integration'
python scripts/make_reid_demo.py
python scripts/run_reid_detections.py \
  --video experiments/reid_online/synthetic_demo/synthetic_vehicle_demo.mp4 \
  --detections experiments/reid_online/synthetic_demo/synthetic_vehicle_detections.csv \
  --output experiments/reid_online/synthetic_demo/associated.txt \
  --annotate experiments/reid_online/synthetic_demo/annotated.mp4
~~~

Expected synthetic demo: eight frames, fourteen object observations, two
association IDs, and the red rectangle re-associated after its two-frame
absence. This is a synthetic functionality check, not a performance benchmark.

## Run with your own authorized detector outputs

Provide a video and UTF-8 CSV containing:

~~~csv
frame,x1,y1,x2,y2,score
1,20,20,80,80,0.92
3,23,20,83,80,0.93
~~~

Frame IDs are 1-based. Boxes use source-image pixel coordinates in xyxy
format, with positive width and height, scores in [0, 1]. These are
detections, not identity labels or verified ground truth.

~~~bash
python scripts/run_reid_detections.py \
  --video sample.mp4 --detections vehicles.csv \
  --output experiments/reid_online/sample.txt
~~~

Output contains frame,track_id,x,y,w,h,score,-1,-1,-1.
A JSON sidecar records the method and hashes of inputs.

## Optional pretrained generic ImageNet encoder

Install compatible torch and torchvision independently of lightweight
CPU test dependencies. Supply a verified local torchvision-format ResNet18
state dict; no network download happens automatically:

~~~bash
python scripts/run_reid_detections.py \
  --video sample.mp4 --detections vehicles.csv \
  --output experiments/reid_online/sample_resnet.txt \
  --embedding resnet18 --device cpu \
  --weights /absolute/path/to/resnet18_state_dict.pth \
  --weights-sha256 YOUR_TRUSTED_SHA256_DIGEST
~~~

Check weight source, checksum, and license separately. The ResNet18
inference path has not yet been tested with real weights on the user's GPU.

## Mathematics and limitations

For track i and detection j:

    C(i,j) = w*(1-cosine(f_i, f_j)) + (1-w)*(1-IoU(B_i,B_j))

Default w=0.85. Reject candidates below similarity 0.75, beyond the
gap-scaled position bound, or above cost 0.55. Use Hungarian one-to-one
matching. Expire tracks after 30 frames since last match. Memory is an
unguarded EMA; VERA's protected identity memory is **not implemented**.

With T active tracks, D detections and F feature dimensions, dense matching
cost is O(T D F + max(T,D)^3), plus embedding extraction. The cost matrix
uses O(T D) memory. The rule has not been tuned against real identity labels.

HSV appearance can fail for lookalike vehicles or illumination changes.
Neither encoder establishes improved HOTA/IDF1/AssA or identity recovery.
Next: controlled learned vehicle embeddings, integration into the pinned
BoT-SORT association path, annotated evaluation and VERA reliability gating.
