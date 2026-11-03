#!/usr/bin/env bash
# Explicit, idempotent LOCAL patch for the pinned BoT-SORT submodule.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE="$PROJECT_ROOT/baselines/BoT-SORT"
REID_PATCH="$PROJECT_ROOT/patches/botsort-reid-compat.patch"
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

# Allow repeat calls without re-running other compatibility patch scripts.
if git -C "$SUBMODULE" apply --reverse --check "$REID_PATCH" >/dev/null 2>&1; then
  echo "BoT-SORT ReID compatibility patch is already applied."
  exit 0
fi

if ! git -C "$SUBMODULE" apply --reverse --check "$BASELINE_PATCH" >/dev/null 2>&1; then
  echo "First run: bash scripts/apply_botsort_compatibility_patch.sh" >&2
  exit 1
fi

# Never overwrite unrelated local edits in the two patched tracker files.
if [[ -n "$(git -C "$SUBMODULE" status --porcelain -- tracker/bot_sort.py tracker/matching.py)" ]]; then
  echo "BoT-SORT tracker files have local changes; refusing to overwrite." >&2
  git -C "$SUBMODULE" status --short -- tracker/bot_sort.py tracker/matching.py >&2
  exit 1
fi

git -C "$SUBMODULE" apply --check "$REID_PATCH"
git -C "$SUBMODULE" apply "$REID_PATCH"
git -C "$SUBMODULE" apply --reverse --check "$REID_PATCH"
echo "Applied pinned BoT-SORT ReID compatibility patch locally."
echo "Changes stay inside the uncommitted submodule working tree."
