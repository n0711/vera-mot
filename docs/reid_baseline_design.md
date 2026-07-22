# Milestone 4.1 UAVDT ReID baseline design

## Data and leakage boundary

Only the authoritative 30-sequence UAVDT MOT training pool may supply ReID training or validation observations. The frozen 20-sequence test set is forbidden from training, validation, checkpoint selection, and threshold tuning. Splits are sequence-disjoint and identities use the composite key `<sequence>_<target_id>` because UAVDT target IDs are sequence-local.

The earlier 107-identity figure was based on non-composite identity handling and is rejected. Across the 30 training sequences, `*_gt_whole.txt` contains 1,366 raw composite identities. The Milestone 4.1 difference of 23 identities is now fully reconciled: 19 train identities had fewer than 10 valid observations and four train identities had no valid observations after box filtering. Milestone 4.2 additionally requires at least four final samples after temporal sampling, rejecting 27 more train identities and four validation identities. The final accounting is 1,098 raw = 1,048 accepted + 50 rejected for train, and 268 raw = 264 accepted + 4 rejected for validation. Category-specific identity totals can overlap when a target has more than one raw category code, so they must not be summed as a unique-identity count.

Vehicle categories 1, 2, and 3 are retained. Category, occlusion, and out-of-view values remain raw numeric codes. Each observation retains its original category alongside a deterministic identity-majority category and a conflict flag. Boxes are clipped to the source image; missing frames and invalid boxes fail generation, while clipped boxes below 10 pixels in width, 10 pixels in height, or 100 pixels in area are rejected. An identity needs at least 10 valid observations. Every fifth valid temporal observation is selected, followed by an endpoint-preserving uniform cap at 200 samples, and identities with fewer than four final samples are rejected.

## Representative validation split

The earlier “last six sorted sequences” proposal is rejected because lexical position has no relationship to scene representativeness. The fixed split is recorded with its evidence in `configs/uavdt_reid_split.yaml`.

The six validation sequences are `M0201`, `M0202`, `M0501`, `M0603`, `M1002`, and `M1306`. They were selected deterministically by exhaustive evaluation of six-sequence subsets. Eligible subsets had to cover every official sequence attribute present in the 30-sequence pool. The objective minimized mean absolute standardized deviation from the full pool using official weather, altitude, view, and long-term attributes plus observed vehicle density, category mix, box-size medians, raw occlusion/out-of-view distributions, identity count, observation count, and observations per identity. The authoritative local attribute README does not document a distinct camera-motion code, so none is invented.

## Planned model and training defaults

The planned starting point is the pinned vehicle configuration `baselines/BoT-SORT/fast_reid/configs/VeRi/sbs_R50-ibn.yml`: FastReID VeRi SBS ResNet-50-IBN with 256×256 input. Initialization must be ImageNet, not the available MOT17 pedestrian weights. The identity sampler uses four images per identity, initial batch size 16, and seed 42.

These are future training settings only. Milestone 4.1 does not train, download weights, generate crops, enable ReID tracking, implement VERA, tune thresholds, or evaluate the frozen test sequences. VERA remains disabled.

## FastReID UAVDT adapter and loader gate

The project-owned adapter is `src/vera_mot/fastreid_uavdt.py`. It reads only the frozen train and validation crop registries, pins their registry and dataset checksums, confines paths to the approved crop roots, optionally hashes and decodes every crop, rejects duplicate/short/forbidden records, and maps lexically sorted composite identities to contiguous labels. Training labels are exactly `0..1047`; validation has a separate `0..263` label space. Deterministically sorted sequence names provide sequence-stream metadata IDs. These IDs do not claim undocumented physical camera relationships.

The explicit pinned FastReID dataset names are `UAVDTReIDTrain` and `UAVDTReIDVal`. Their configuration is `configs/fastreid/uavdt_sbs_R50_ibn.yml`, inheriting from the pinned VeRi SBS ResNet-50-IBN configuration. It fixes 256×256 inputs, batch size 16, four images and four identities per batch, seed 42, ImageNet backbone initialization with no checkpoint path, classification plus triplet losses, AMP, and two loader workers. The 60-epoch value is provisional until a later GPU throughput smoke test.

Run the data-only integration gate with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/dry_run_uavdt_fastreid_loader.py --verify-crops --batches 1
```

The expected tensor shape is `[16, 3, 256, 256]` with exactly four identities and four samples per identity. This command does not construct a model or trainer and performs no forward, backward, or optimisation step.

Standard validation evaluation remains blocked. UAVDT composite identities are sequence-local and their observations belong to one sequence stream. The pinned FastReID evaluator removes same-identity/same-camera matches; assigning fabricated camera IDs would produce misleading cross-camera mAP and Rank-1. `DATASETS.TESTS` is therefore empty, and validation is registered for count/data-loader checks only until a defensible same-sequence temporal retrieval protocol and evaluator semantics are designed and reviewed.

All seven ignored visual-QA contact sheets require human review before any training. Automated generation and one assistant inspection do not constitute the required human confirmation. Training remains prohibited, and VERA remains unimplemented.

## Generated and tracked evidence

The ignored manifests are `datasets/UAVDT/reid_manifests/train.csv` and `datasets/UAVDT/reid_manifests/val.csv`. They contain source-image and lossless PNG crop paths relative to the project root. Native-size crops live under `datasets/UAVDT/reid_crops/`; FastReID will resize them to 256×256 during loading. Full crop registries and contact sheets remain ignored. The tracked compact evidence files are `experiments/reid_data/uavdt_reid_manifest_summary.json` and `experiments/reid_data/uavdt_reid_crop_summary.json`.

## Next task boundary

Before any GPU smoke test, a human must review all seven visual-QA contact sheets and approve the dataset. The next implementation task after that approval is a bounded GPU smoke-test harness that constructs the pinned model and measures memory/throughput on a few training batches without an optimisation step. ImageNet initialization provenance must be resolved without using MOT17 pedestrian weights. Full training, tracker integration, VERA, threshold tuning, and frozen-test evaluation remain separate later milestones.
