#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="$PROJECT_ROOT/baselines/BoT-SORT"
PATCH_FILE="$PROJECT_ROOT/patches/botsort-python310-pytorch24.patch"
EXPECTED_REVISION="251985436d6712aaf682aaaf5f71edb4987224bd"

if [[ ! -e "$SUBMODULE/.git" ]]; then
  echo "BoT-SORT submodule is not initialized. Run: git submodule update --init --recursive" >&2
  exit 1
fi

actual_revision="$(git -C "$SUBMODULE" rev-parse HEAD)"
if [[ "$actual_revision" != "$EXPECTED_REVISION" ]]; then
  echo "BoT-SORT revision mismatch: expected $EXPECTED_REVISION, found $actual_revision" >&2
  exit 1
fi

if git -C "$SUBMODULE" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  patch_paths="$(sed -n 's#^+++ b/##p' "$PATCH_FILE")"
  modified_paths="$(git -C "$SUBMODULE" status --porcelain --untracked-files=no | sed -E 's/^.. //' | sort)"
  while IFS= read -r modified_path; do
    [[ -z "$modified_path" ]] && continue
    if ! grep -Fxq "$modified_path" <<<"$patch_paths" && [[ "$modified_path" != "tracker/bot_sort.py" && "$modified_path" != "tracker/matching.py" ]]; then
      echo "BoT-SORT has unrelated tracked modification: $modified_path" >&2
      echo "Refusing to continue without overwriting user work." >&2
      exit 1
    fi
  done <<<"$modified_paths"
  echo "BoT-SORT compatibility patch is already applied and verified."
  exit 0
fi

if [[ -n "$(git -C "$SUBMODULE" status --porcelain --untracked-files=no)" ]]; then
  echo "BoT-SORT has tracked modifications that are not the complete compatibility patch." >&2
  echo "Refusing to overwrite user work:" >&2
  git -C "$SUBMODULE" status --short --untracked-files=no >&2
  exit 1
fi

if ! git -C "$SUBMODULE" apply --check "$PATCH_FILE"; then
  echo "Compatibility patch does not apply cleanly to the expected revision." >&2
  exit 1
fi

git -C "$SUBMODULE" apply "$PATCH_FILE"
git -C "$SUBMODULE" apply --reverse --check "$PATCH_FILE"
echo "Applied and verified BoT-SORT compatibility patch."
