#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="$PROJECT_ROOT/baselines/BoT-SORT"
PATCH_FILE="$PROJECT_ROOT/patches/botsort-python310-pytorch24.patch"
EXPECTED_REVISION="251985436d6712aaf682aaaf5f71edb4987224bd"
PATCH_FILES=(
  "fast_reid/fastreid/data/build.py"
  "fast_reid/fastreid/engine/train_loop.py"
  "fast_reid/fastreid/engine/hooks.py"
  "fast_reid/fastreid/evaluation/testing.py"
)
EXPECTED_PATCH_SHA256=(
  "21338d63c99266befe15fca1df48bc4c9441e800bad73d7fd6406fbf726ee17d"
  "469395a03eddda4bde5b4c25c09df2c4ab0e6f4c70a399a82b7383e5297d4476"
  "47009135f6711b8be6f6d8fab13474f6af2eac6cbd8a0e233bbdcf922ba2b49e"
  "81e5352d5eab92c8211a8097b8d6f7c1ed1bed1a1ae89c1e7e484d592f5ebcf6"
)
EXPECTED_LOCAL_COMPAT_FILES=("tracker/bot_sort.py" "tracker/matching.py")
EXPECTED_LOCAL_COMPAT_SHA256=(
  "0e8b520af221e318ee64ddb2a2d2ae91434c691de7633f65796dd2df247c11df"
  "a45fe4081cca1e2a98d274273eee350ee0b578a79c44b8b20f7ece246e56f434"
)

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
  modified_paths="$(git -C "$SUBMODULE" status --porcelain --untracked-files=no | sed -E 's/^.. //' | sort)"
  while IFS= read -r modified_path; do
    [[ -z "$modified_path" ]] && continue
    allowed=0
    for path in "${PATCH_FILES[@]}" "${EXPECTED_LOCAL_COMPAT_FILES[@]}"; do
      [[ "$modified_path" == "$path" ]] && allowed=1
    done
    if [[ "$allowed" -ne 1 ]]; then
      echo "BoT-SORT has unrelated tracked modification: $modified_path" >&2
      echo "Refusing to continue without overwriting user work." >&2
      exit 1
    fi
  done <<<"$modified_paths"
  for i in "${!PATCH_FILES[@]}"; do
    actual_sha="$(sha256sum "$SUBMODULE/${PATCH_FILES[$i]}" | awk '{print $1}')"
    [[ "$actual_sha" == "${EXPECTED_PATCH_SHA256[$i]}" ]] || { echo "Patched file checksum mismatch: ${PATCH_FILES[$i]}" >&2; exit 1; }
  done
  for i in "${!EXPECTED_LOCAL_COMPAT_FILES[@]}"; do
    path="${EXPECTED_LOCAL_COMPAT_FILES[$i]}"
    if [[ -n "$(git -C "$SUBMODULE" status --porcelain -- "$path")" ]]; then
      actual_sha="$(sha256sum "$SUBMODULE/$path" | awk '{print $1}')"
      [[ "$actual_sha" == "${EXPECTED_LOCAL_COMPAT_SHA256[$i]}" ]] || { echo "Local compatibility checksum mismatch: $path" >&2; exit 1; }
    fi
  done
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
