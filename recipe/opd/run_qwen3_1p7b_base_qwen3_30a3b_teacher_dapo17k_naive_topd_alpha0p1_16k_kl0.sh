#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

ADV_ESTIMATOR=naive_topd
TOPD_ALPHA=0.1
TOPD_ALPHA_TAG=alpha0p1
METHOD_TAG=naive_topd_${TOPD_ALPHA_TAG}
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=32

source "${SCRIPT_DIR}/run_qwen3_1p7b_base_qwen3_teacher_dapo17k_common.sh"
