# UAVDT ReID crop-quality protocol

Protocol version: 1.0.0

This analysis records deterministic raw image-quality features and crop geometry for later study of temporal ReID retrieval errors. It does not define a reliability score, quality classes, weights, thresholds, association gating, or interpretations of UAVDT's raw occlusion and out-of-view codes.

## Data contract and leakage controls

Each manifest row is joined one-to-one with its crop-registry row on `split`, `sequence`, `frame`, `identity`, and the manifest `planned_crop_path`/registry `crop_path`. Duplicate, missing, or mismatched joins are fatal. Split, category, majority category, category-conflict flag, source image, crop dimensions, and crop path must agree. Frozen UAVDT test sequences are rejected using the project-owned frozen sequence list.

The source and crop datasets are read-only. Crop dimensions are always checked against the registry; `--verify-crops` additionally checks each crop file's SHA-256 against the registry. No embeddings or CUDA are used. `--max-records N` selects the first N records after deterministic join-key sorting, making bounded runs reproducible.

## Measurements

Geometry is calculated from the manifest's clipped box and the source image dimensions. `distance_right` is `source_width - (clipped_left + width)` and `distance_bottom` is analogous. The minimum border distance is the minimum of left, top, right, and bottom distances. A border-touch flag is true only at distance zero. `original_box_was_clipped` records whether any original box coordinate or dimension differs from its clipped counterpart.

Images are converted to 8-bit grayscale with Pillow. Mean, population standard deviation, and RMS contrast use grayscale values divided by 255. Laplacian variance uses `cv2.Laplacian(gray, cv2.CV_64F)`. Tenengrad is the mean of squared 3x3 Sobel-x plus squared 3x3 Sobel-y responses. Entropy is Shannon entropy in bits from the 256-bin grayscale histogram. Dark and bright saturation fractions count values at most 5 and at least 250, respectively. Laplacian variance and Tenengrad remain unnormalized.

Identity summaries use medians and preserve raw annotation-code counts as compact JSON objects. Ranked CSVs and labelled contact sheets are deterministic, with crop path breaking metric ties. “Border risk” is only a deterministic inspection ordering by ascending raw minimum border distance; it is not a score or classification.

## Invocation and provenance

```bash
python scripts/analyze_uavdt_reid_quality.py \
  --manifest datasets/UAVDT/reid_manifests/train.csv \
  --registry datasets/UAVDT/reid_manifests/train_crop_registry.csv \
  --output-dir experiments/reid_quality/train \
  --verify-crops
```

The output directory contains per-crop and per-identity CSVs, three ranked CSVs, optional contact sheets, and `summary.json`. The summary records protocol version, UTC creation time, Git SHA, input paths and hashes, selected crop-dataset checksum, counts, sequences, split, numeric percentiles, verification mode, bounded-selection value, rejected-row count, and frozen-test count. Invalid input aborts without reporting partial records as accepted; successful summaries therefore record zero rejected and zero frozen-test rows.
