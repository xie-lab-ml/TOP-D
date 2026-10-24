#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
export MODEL_DIR=${MODEL_DIR:-${ROOT}/model}
export STUDENT_REPO=${STUDENT_REPO:-Qwen/Qwen3-1.7B-Base}
export TEACHER_REPO=${TEACHER_REPO:-Qwen/Qwen3-30B-A3B-Instruct-2507}
# Optional STUDENT_REVISION / TEACHER_REVISION override a saved models.lock.json.
# Without overrides, the downloader reuses the previously resolved revisions.
"${PYTHON_BIN}" "${ROOT}/scripts/download_models.py"
