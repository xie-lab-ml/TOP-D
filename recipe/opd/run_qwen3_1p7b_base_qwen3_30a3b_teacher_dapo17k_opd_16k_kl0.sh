#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

ADV_ESTIMATOR=opd
TOPD_ALPHA=
METHOD_TAG=opd
PPO_MINI_BATCH_SIZE=128

source "${SCRIPT_DIR}/run_qwen3_1p7b_base_qwen3_teacher_dapo17k_common.sh"
