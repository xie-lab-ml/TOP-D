#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ADV_ESTIMATOR=topd
TOPD_ALPHA=${TOPD_ALPHA:-0.1}
METHOD_TAG=topd_alpha${TOPD_ALPHA//./p}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
source "${SCRIPT_DIR}/run_qwen3_1p7b_base_qwen3_teacher_dapo17k_common.sh"
