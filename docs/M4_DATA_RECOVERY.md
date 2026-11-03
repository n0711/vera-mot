# M4 dataset recovery and local readiness

## Boundary

The GitHub repository provides code and historical compact summaries, not
the UAVDT images, ground-truth source, generated ReID crops, trained weights,
or original training metadata. Synthetic tests never establish model accuracy.

## Check local prerequisites without writing

Run: python scripts/audit_m4_local.py

Exit 1 indicates missing paths; exit 0 only means all expected paths exist.
Neither outcome proves integrity or that training was performed.
The script checks the 24 frozen training and six validation sequences without
visiting the frozen 20-sequence test set.

## Authorized data

Obtain UAVDT-Benchmark-M and the DET/MOT toolkit from the original authors:
https://sites.google.com/view/daweidu/projects/uavdt

The authors link Google Drive and Baidu sources; if Google Drive is quota-
blocked, use an authorized source or contact the authors. Do not assume that
unofficial mirrors carry distribution permission or preserve GT semantics.
Record archive origin, download date, byte sizes and SHA-256 values; a local
hash is not independent authenticity proof.

## Rebuild order

1. Inspect archives before extracting into the documented full_mot paths.
2. Rebuild train/val manifests with build_uavdt_reid_manifest.py.
3. Compare reconstructed summaries with frozen tracked evidence before
   overwriting any historical tracked file.
4. Use materialize_uavdt_reid_crops.py --preflight-only, then materialize
   crops after checksum and disk validation. Branch names no longer gate
   this script: source checksums, split invariants and path rules do.
5. Review seven visual QA contact sheets before training.
6. Verify generated registries, crop hashes, initialization provenance,
   and the pinned FastReID environment.
7. On a 6 GB GPU, conduct bounded memory preflight. Any changed training
   settings require a new documented experiment, not a historical claim.
8. Generate new training checkpoints and their completion records.
9. Run evaluate_uavdt_reid_checkpoints.py --plan-only
   --verify-training-provenance and then a clearly labelled smoke run.
10. Run the full temporal evaluation, preserving the fixed primary final
    checkpoint and test-set embargo.

No new HOTA, IDF1, ReID retrieval, or VERA improvement is claimed.
