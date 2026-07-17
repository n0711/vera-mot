#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLOX_DIR="${PROJECT_ROOT}/baselines/YOLOX"
CHECKPOINT="${PROJECT_ROOT}/experiments/detector_runs/yolox_uavdt_s_1280/best_ckpt.pth"
CONFIG="${PROJECT_ROOT}/configs/yolox_uavdt_s_1280_test.py"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found:"
    echo "${CHECKPOINT}"
    echo "Wait until training finishes."
    exit 1
fi

cd "${YOLOX_DIR}"

PYTHONPATH=. python tools/eval.py \
    -f "${CONFIG}" \
    -c "${CHECKPOINT}" \
    -d 1 \
    -b 2 \
    --fp16