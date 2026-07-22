# UAVDT temporal vehicle-ReID validation protocol

## Boundary and purpose

Protocol `uavdt-temporal-reid-v1` measures retrieval quality on the six fixed validation sequences in `configs/uavdt_reid_split.yaml`: `M0201`, `M0202`, `M0501`, `M0603`, `M1002`, and `M1306`. The frozen test sequences are forbidden. A row from a frozen or non-validation sequence is a fatal error. Identity is always the registry's composite `<sequence>_<target_id>` value; numeric target IDs are never joined across sequences.

This is a project-owned evaluator. It does not use FastReID's VeRi evaluator, whose same-camera exclusion is invalid for sequence-local UAVDT identities, and it does not change tracking or training.

## Deterministic temporal split

Observations are grouped by composite identity and sorted by `(sequence, identity, frame, crop_path)`. Identities with fewer than four observations are recorded as rejected. For an accepted identity, `k = min(5, floor(observation_count / 2))`; the earliest `k` crops form its gallery and the latest `k` form its queries. These sets cannot overlap. Unused middle observations are excluded. `protocol_manifest.json` records every selected and excluded observation, as well as rejected identities. `--max-identities` takes the first identities in the same deterministic ordering and exists only for bounded smoke tests.

## Retrieval and metrics

Each query is ranked only against gallery crops from its own sequence. Cross-sequence vehicles are not negatives. Embeddings are produced in inference mode by the pinned FastReID configuration and explicitly supplied checkpoint, then L2-normalized. Similarity is the normalized dot product and cosine distance is `1 - similarity`. Equal distances are resolved by lexical gallery crop path.

For each sequence, the evaluator reports identity, query, and gallery counts; Rank-1; Rank-5; mean average precision; mean and median positive cosine distance; mean nearest-negative cosine distance; and mean positive/negative margin. A query's margin is its nearest-negative distance minus its mean positive distance. If a sequence has no negative identity, negative distance and margin are unavailable and omitted from those means. Macro metrics average sequence metrics equally. Micro metrics pool all valid queries, while retrieval remains sequence-local.

Average precision is the mean precision at each rank containing a relevant gallery crop. Rank-k is one when the first relevant crop appears at or before rank `k` (even when the gallery contains fewer than `k` samples).

## Provenance and outputs

The output directory contains `protocol_manifest.json`, `summary.json`, `per_sequence.csv`, `per_query.csv`, `hardest_positive_pairs.csv`, and `nearest_negative_pairs.csv`. Every JSON document and every CSV row includes the checkpoint, configuration, and registry paths and SHA-256 values; Git commit when available; protocol version; UTC creation timestamp; and validation sequence list. CSV rows are deterministically ordered. Existing non-empty output directories require `--overwrite`; generated evaluation directories are ignored by Git by default.

`--verify-crops` hashes each selected crop against the registry. All selected crops must exist below the approved validation crop root regardless of that option. The checkpoint must exist, contain a model state dictionary, and load strictly into the pinned architecture; initialization or missing/incompatible weights are never accepted as a fallback.

## Invocation

```bash
python scripts/evaluate_uavdt_reid_temporal.py \
  --config configs/fastreid/uavdt_sbs_R50_ibn.yml \
  --checkpoint PATH_TO_CHECKPOINT \
  --registry datasets/UAVDT/reid_manifests/val_crop_registry.csv \
  --output-dir experiments/reid_evaluation/CHECKPOINT_NAME \
  --device cuda \
  --batch-size 64 \
  --num-workers 2
```

Use `--device cpu` for CPU inference. Add `--verify-crops` for full selected-crop integrity checking, `--max-identities N` for a bounded smoke test, or `--overwrite` to replace named output files in a non-empty evaluation directory.
