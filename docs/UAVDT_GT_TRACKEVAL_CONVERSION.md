# M4.9 - UAVDT source-GT conversion to pinned TrackEval

**Evidence boundary:** CPU-only synthetic implementation and parser smoke tests. No
original UAVDT images, trained ReID checkpoints or real metric results are included.

## Input schema and semantics

Original annotation file: M0201_gt_whole.txt (similarly named for other
sequences). It has exactly nine integer fields:

| Position | UAVDT field | Accepted value |
| --- | --- | --- |
| 1 | frame index | 1 through original sequence length |
| 2 | sequence-local identity | positive, unique per frame |
| 3-6 | left, top, width, height | integer xywh, width/height positive |
| 7 | out of view | original code 1, 2, or 3 |
| 8 | occlusion | original code 1, 2, 3, or 4 |
| 9 | object category | original code 1, 2, or 3 |

Source annotation code names and scheme are described by the official UAVDT
project and the TensorBay loader. The distinct M0201_gt.txt from the official
MOT toolkit is NOT the same file as the original nine-column gt_whole source.

The pinned TrackEval MotChallenge2DBox parser reads the seventh field as
ground-truth mark and the eighth as ground-truth class. It supports only
the class name pedestrian. To score a single combined VEHICLE category,
we explicitly alias every source vehicle to TrackEval class ID 1 and
mark 1, without pretending the source category was a pedestrian label.
The converter's ten-column output is:

    frame,id,left,top,width,height,1,1,-1,-1

The last two fields are unused placeholders. No source occlusion or out-of-view
codes are silently recast as confidence, visibility, or class numbers. The
source-to-output transform is lossy for raw code metadata; provenance reports
source-code histograms and canonical nine-field roundtrip separately.

References:
- Official dataset description: https://sites.google.com/view/grli-uavdt/
- Public nine-column loader: https://tensorbay-python-sdk.graviti.com/en/v1.20.0/_modules/tensorbay/opendataset/UAVDT/loader.html
- Exact pinned TrackEval parser at commit 12c8791b303e0a0b50f753af204249e622d0281a:
  https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/datasets/mot_challenge_2d_box.py

## Validation invariants

1. Only the six ReID development-validation sequences are allowed. The frozen
   20-sequence test set is refused.
2. Input filename must be <sequence>_gt_whole.txt; its row count and sequence
   length must match the committed original UAVDT inventory.
3. Non-integer/nonfinite values, unknown attribute codes, duplicate
   frame/target pairs, nonpositive dimensions and out-of-range frames fail.
4. Canonically sort by frame then ID. Confirm every converted row matches the
   source identity and exact integer geometry, with a typed nine-field
   roundtrip and a ten-field output hash.
5. Generate a separate aggregate provenance manifest: actual file-byte
   SHA-256, split/inventory SHA-256, code distributions, pinned parser commit,
   class alias declaration and limitations.
6. Output must be fresh, in a separate directory. Existing results and
   original UAVDT files are not overwritten.

## Usage with authorized original annotation files

    mkdir -p experiments/reid_evaluation/gt_conversion/M0201

    python scripts/convert_uavdt_gt_trackeval.py \
      --sequence M0201 \
      --source /authorized/UAVDT/GT/M0201_gt_whole.txt \
      --output-dir experiments/reid_evaluation/gt_conversion/M0201

The output directory receives M0201_gt.txt and M0201_gt.conversion.json.
This creates TrackEval-compatible inputs, NOT ground-truth identity metrics.

Run data-free tests:

    python -m pytest -q -m 'not integration' tests/test_uavdt_gt_convert.py

Run the exact pinned TrackEval parser smoke test (synthetic GT and tracker):

    git submodule update --init baselines/TrackEval
    python -m pytest -q -m integration tests/test_uavdt_trackeval_parser.py

GitHub CI performs this separate pinned parser check.

## Outstanding research gates

- Independent human review of a small sample of original UAVDT annotations
  and code semantics against official dataset documentation.
- Explicitly validate official MOT toolkit ground truth vs source GT conversion,
  especially any ignore-region handling and ground-truth retention decisions.
- Actual authorized validation sequence sources, detector inputs and paired
  tracker results for M4.8 preflight.
- Complete TrackEval metric protocol and real annotated identity evaluation.
  Only then may HOTA, IDF1, AssA and IDSW be reported; parser smoke tests cannot
  provide tracking performance evidence.

The historical 20-sequence frozen no-ReID baseline is unchanged.
