#!/usr/bin/env bash
# Opt-in VERA crop-quality gate for the EXACT pinned BoT-SORT working tree.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE="$PROJECT_ROOT/baselines/BoT-SORT"
VERA_PATCH="$PROJECT_ROOT/patches/botsort-vera-quality-gate.patch"
PINNED_REVISION="251985436d6712aaf682aaaf5f71edb4987224bd"
EXPECTED_BOT_SORT="73c5a1717753c298b56281c79e0855070edab590"
EXPECTED_MATCHING="74b7bb74432a6e24b5997240f8a8b19637960213"

if [[ ! -e "$SUBMODULE/.git" ]]; then
  echo "Initialize pinned BoT-SORT: git submodule update --init --recursive" >&2
  exit 1
fi
actual="$(git -C "$SUBMODULE" rev-parse HEAD)"
if [[ "$actual" != "$PINNED_REVISION" ]]; then
  echo "Unexpected BoT-SORT revision: $actual" >&2
  exit 1
fi

# Refuses unrelated changes; this script does not reset or force the submodule.
bash "$PROJECT_ROOT/scripts/apply_botsort_reid_patch.sh"

if git -C "$SUBMODULE" apply --reverse --check "$VERA_PATCH" >/dev/null 2>&1; then
  echo "VERA reliability gate patch is already applied."
  exit 0
fi

actual_bot="$(git -C "$SUBMODULE" hash-object tracker/bot_sort.py)"
actual_matching="$(git -C "$SUBMODULE" hash-object tracker/matching.py)"
if [[ "$actual_bot" != "$EXPECTED_BOT_SORT" ||
      "$actual_matching" != "$EXPECTED_MATCHING" ]]; then
  echo "BoT-SORT files differ from vetted pre-VERA patch state." >&2
  echo "Expected bot_sort $EXPECTED_BOT_SORT, found $actual_bot" >&2
  echo "Expected matching $EXPECTED_MATCHING, found $actual_matching" >&2
  echo "Refusing to overwrite local modifications." >&2
  exit 1
fi

git -C "$SUBMODULE" apply --check "$VERA_PATCH"
git -C "$SUBMODULE" apply "$VERA_PATCH"
git -C "$SUBMODULE" apply --reverse --check "$VERA_PATCH"
echo "VERA quality gating applied to local pinned BoT-SORT."
echo "No benchmark files, submodule pin, or third-party source history changed."
