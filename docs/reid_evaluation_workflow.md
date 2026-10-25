# ReID evaluation workflow

This workflow integrates temporal retrieval and raw crop-quality measurement.
It prepares the evidence needed before VERA reliability thresholds or tracker
changes. It does not establish a BoT-SORT-ReID tracking result.

## Required local artifacts

- Initialize the pinned BoT-SORT submodule and use the recorded FastReID runtime.
- Preserve the frozen validation registry, crops and source images.
- Preserve `completion.json`, `launch_manifest.json` and all six trained
  checkpoints from `experiments/reid_training/uavdt_sbs_R50_ibn/`.
- Check checkpoint hashes against the original training completion evidence.
  This sweep records the supplied weights' hashes; it does not independently
  prove that those weights came from the reported training run.

Required names are `model_0009.pth`, `model_0019.pth`, `model_0029.pth`,
`model_0039.pth`, `model_0049.pth` and `model_final.pth`. The fixed epoch-60
`model_final.pth` is the primary checkpoint. Earlier epochs are diagnostic only;
the command does not select a model using validation or frozen-test scores.

## Execution order

Run from the repository root, with its baseline virtual environment active:

```bash
git submodule update --init baselines/BoT-SORT
python scripts/evaluate_uavdt_reid_checkpoints.py --plan-only
```

Planning checks file availability, the frozen registry hash and inherited YAML
dependencies, then prints the complete six-job plan. It imports no PyTorch,
loads no weights and writes nothing. A successful plan is not an inference test.

A bounded smoke sweep uses a separate output directory:

```bash
python scripts/evaluate_uavdt_reid_checkpoints.py \
  --max-identities 8 --num-workers 0 --verify-crops \
  --output-dir experiments/reid_evaluation/smoke
```

The first eight identities in deterministic order may cover only part of the
validation split. Every output is labelled `smoke`, even if a large limit would
include every identity. Run full validation separately:

```bash
python scripts/evaluate_uavdt_reid_checkpoints.py --verify-crops
python scripts/evaluate_uavdt_reid_checkpoints.py --verify-crops --resume
```

Each checkpoint gets a subdirectory containing the temporal evaluator's six
artifacts plus a `run_record.json`. The sweep root contains `plan.json` and,
only after every checkpoint succeeds and uses the same query/gallery split,
`comparison.json`. Read Rank-1, Rank-5, mAP and margins from the macro and micro
sections; these are retrieval metrics, not HOTA or IDF1 tracking scores.

Resume requires the same checkpoint, frozen registry, split, complete inherited
configuration, evaluator/upstream Python sources, runtime package versions,
device and execution options. All six output artifacts must still match their
recorded SHA-256 values. Stale, edited or partial outputs stop before inference.
Use a fresh output directory or explicit `--overwrite` for a replacement run.
A replacement removes the older comparison report before starting, so a failed
replacement cannot leave that report looking current. Unrelated files are kept.

After retrieval validation, measure raw crop quality independently:

```bash
python scripts/analyze_uavdt_reid_quality.py \
  --manifest datasets/UAVDT/reid_manifests/val.csv \
  --registry datasets/UAVDT/reid_manifests/val_crop_registry.csv \
  --output-dir experiments/reid_quality/val --verify-crops
```

Use the recorded crop paths and composite identities to study retrieval failures
against quality features. No reliability score or threshold is selected by these
commands. Frozen test sequences remain excluded from all development analyses.

## Verification and remaining work

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q -m 'not integration'
```

The CPU suite covers retrieval formulas, sequence isolation, deterministic
splits, malformed inputs, extreme feature magnitudes, crop measurements,
checkpoint planning, synthetic orchestration and safe resume/failure behavior.
Synthetic checkpoint fixtures test orchestration only. The FastReID integration
check (`python -m pytest -q -m integration`) requires the pinned full runtime.

The development workspace has no training crops, trained weights or PyTorch.
Full inference, the integration check, retrieval-error/quality analysis and the
controlled BoT-SORT vehicle-ReID baseline remain unverified here. VERA gating,
protected identity memory and ablations follow that evidence.
