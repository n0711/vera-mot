#!/usr/bin/env bash
# Explicit, idempotent LOCAL patches for the pinned BoT-SORT submodule.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE="$PROJECT_ROOT/baselines/BoT-SORT"
REID_PATCH="$PROJECT_ROOT/patches/botsort-reid-compat.patch"
LAZY_PATCH="$PROJECT_ROOT/patches/botsort-lazy-fastreid.patch"
BASELINE_PATCH="$PROJECT_ROOT/patches/botsort-python310-pytorch24.patch"
PINNED_REVISION="251985436d6712aaf682aaaf5f71edb4987224bd"

if [[ ! -e "$SUBMODULE/.git" ]]; then
  echo "Initialize submodule: git submodule update --init --recursive" >&2
  exit 1
fi
actual_revision="$(git -C "$SUBMODULE" rev-parse HEAD)"
if [[ "$actual_revision" != "$PINNED_REVISION" ]]; then
  echo "Pinned BoT-SORT revision mismatch: $actual_revision" >&2
  exit 1
fi

# Preserve the prior ReID compatibility patch, if already installed locally.
if git -C "$SUBMODULE" apply --reverse --check "$REID_PATCH" >/dev/null 2>&1; then
  echo "BoT-SORT ReID compatibility patch already applied."
else
  if ! git -C "$SUBMODULE" apply --reverse --check "$BASELINE_PATCH" >/dev/null 2>&1; then
    echo "First run: bash scripts/apply_botsort_compatibility_patch.sh" >&2
    exit 1
  fi

  # Do not overwrite unrelated edits when first applying the ReID patch.
  if [[ -n "$(git -C "$SUBMODULE" status --porcelain -- tracker/bot_sort.py tracker/matching.py)" ]]; then
    echo "BoT-SORT tracker files have local changes; refusing to overwrite." >&2
    git -C "$SUBMODULE" status --short -- tracker/bot_sort.py tracker/matching.py >&2
    exit 1
  fi

  git -C "$SUBMODULE" apply --check "$REID_PATCH"
  git -C "$SUBMODULE" apply "$REID_PATCH"
  git -C "$SUBMODULE" apply --reverse --check "$REID_PATCH"
  echo "Applied pinned BoT-SORT ReID compatibility patch locally."
fi

# Upstream imports its full FastReID training dependencies at module load,
# including when an external HSV encoder is used. Defer that import until
# the upstream constructor itself requests FastReID.
if git -C "$SUBMODULE" apply --reverse --check "$LAZY_PATCH" >/dev/null 2>&1; then
  echo "Optional FastReID lazy-import patch already applied."
else
  # Apply only if the exact pinned-source hunk is present. No --force.
  if ! git -C "$SUBMODULE" apply --check "$LAZY_PATCH"; then
    echo "FastReID lazy-import patch does not match pinned source; stopping." >&2
    exit 1
  fi
  git -C "$SUBMODULE" apply "$LAZY_PATCH"
  git -C "$SUBMODULE" apply --reverse --check "$LAZY_PATCH"
  echo "Applied optional FastReID lazy-import patch."
fi

echo "Both ReID patches verified; changes stay local to the submodule."
