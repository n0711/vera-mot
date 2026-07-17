cat > docs/research_contract.md <<'EOF'
# VERA-MOT Research Contract

## Method

VERA: Visual Embedding Reliability Assessment.

## Thesis problem

Appearance embeddings extracted from aerial vehicle crops may become
unreliable because of blur, occlusion, small target size, illumination
change and UAV camera motion.

Using these corrupted embeddings can:

1. produce incorrect detection-to-track associations;
2. contaminate long-term identity memory;
3. cause incorrect reactivation after occlusion;
4. increase identity switches and track fragmentation.

## Research question

Can lightweight visual-embedding reliability assessment reduce identity
switches and improve target recovery in UAV video without introducing
significant onboard computational overhead?

## Baseline

BoT-SORT-ReID.

## Comparator

ByteTrack.

## Proposed contribution

VERA will estimate the reliability of every candidate ReID embedding
before it is allowed to:

- influence association;
- update identity memory;
- reactivate a lost track.

## Thesis scope

Included:

- baseline reproduction;
- UAV dataset adaptation;
- embedding reliability assessment;
- reliability-weighted association;
- protected identity memory;
- ablation experiments;
- tracking metrics;
- onboard latency and resource profiling.

Excluded from the thesis core:

- single-object tracking;
- heading estimation;
- gimbal control;
- MAVLink commands;
- UAV following control;
- real closed-loop flight testing.
EOF
