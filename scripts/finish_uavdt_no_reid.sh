#!/usr/bin/env bash
set -eEo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Projects/vera-mot}"
cd "$PROJECT_ROOT"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing virtual environment: $PROJECT_ROOT/.venv" >&2
  exit 1
fi

source .venv/bin/activate

RESULT_ROOT="experiments/tracking_baselines/botsort_no_reid/uavdt"
FILTERED_ROOT="experiments/tracking_baselines/botsort_no_reid/uavdt_official"
LOG_ROOT="experiments/tracking_baselines/botsort_no_reid/logs"
METRIC_ROOT="experiments/tracking_baselines/botsort_no_reid/metrics"

mkdir -p "$RESULT_ROOT" "$FILTERED_ROOT" "$LOG_ROOT" "$METRIC_ROOT/uavdt_official"

SEQUENCES=(
  M0203 M0205 M0208 M0209 M0403
  M0601 M0602 M0606 M0701 M0801
  M0802 M1001 M1004 M1007 M1009
  M1101 M1301 M1302 M1303 M1401
)

for sequence in "${SEQUENCES[@]}"; do
  result_file="$RESULT_ROOT/${sequence}.txt"

  if [[ -s "$result_file" ]]; then
    echo "[SKIP] $sequence already has a non-empty result file"
    continue
  fi

  echo
  echo "========== RUNNING $sequence =========="
  python scripts/run_uavdt_botsort_sequence.py \
    --sequence "$sequence" \
    --fp16

done

completed_count="$(find "$RESULT_ROOT" -maxdepth 1 -type f -name 'M*.txt' -size +0c | wc -l)"

if [[ "$completed_count" -ne 20 ]]; then
  echo "Expected 20 completed sequence result files, found $completed_count" >&2
  exit 1
fi

echo
echo "All 20 UAVDT test sequences completed."

echo
echo "========== FILTERING IGNORE REGIONS =========="
python scripts/filter_uavdt_ignore_regions.py \
  2>&1 | tee "$LOG_ROOT/ignore_filter.log"

filtered_count="$(find "$FILTERED_ROOT" -maxdepth 1 -type f -name 'M*.txt' -size +0c | wc -l)"

if [[ "$filtered_count" -ne 20 ]]; then
  echo "Expected 20 filtered result files, found $filtered_count" >&2
  exit 1
fi

echo
echo "========== RUNNING TRACKEVAL =========="
python scripts/evaluate_uavdt_trackeval_all.py \
  2>&1 | tee "$METRIC_ROOT/uavdt_official_test.log"

grep '^COMBINED' "$METRIC_ROOT/uavdt_official_test.log" \
  > "$METRIC_ROOT/uavdt_official_combined.txt" || true

{
  echo "Completed: $(date --iso-8601=seconds)"
  echo "Sequences: 20"
  echo "Raw result directory: $RESULT_ROOT"
  echo "Filtered result directory: $FILTERED_ROOT"
  echo "TrackEval log: $METRIC_ROOT/uavdt_official_test.log"
  echo "Combined metrics: $METRIC_ROOT/uavdt_official_combined.txt"
} > "$METRIC_ROOT/UAVDT_NO_REID_BASELINE_DONE.txt"

echo
echo "========== COMPLETE =========="
cat "$METRIC_ROOT/uavdt_official_combined.txt"
echo
echo "Done marker: $METRIC_ROOT/UAVDT_NO_REID_BASELINE_DONE.txt"
