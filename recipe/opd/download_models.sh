#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to the release root so the script is portable after cloning.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# Download complete student and teacher checkpoints into model/ by default.
# Keep the weight shards, tokenizer files, and model configuration together.
# STUDENT_DIR / TEACHER_DIR override download destinations only; if customized,
# also set MODEL_PATH / TEACHER_MODEL_PATH when launching a training recipe.
# Existing complete checkpoints can be reused without running this helper.
MODEL_DIR=${MODEL_DIR:-${RELEASE_ROOT}/model}
STUDENT_REPO=${STUDENT_REPO:-Qwen/Qwen3-1.7B-Base}
TEACHER_REPO=${TEACHER_REPO:-Qwen/Qwen3-30B-A3B-Instruct-2507}
STUDENT_DIR=${STUDENT_DIR:-${MODEL_DIR}/Qwen3-1.7B-Base}
TEACHER_DIR=${TEACHER_DIR:-${MODEL_DIR}/Qwen3-30B-A3B-Instruct-2507}

mkdir -p "${MODEL_DIR}"

python3 - <<PY_DOWNLOAD
from huggingface_hub import snapshot_download

models = [
    ("student", "${STUDENT_REPO}", "${STUDENT_DIR}"),
    ("teacher", "${TEACHER_REPO}", "${TEACHER_DIR}"),
]
for name, repo_id, local_dir in models:
    print(f"Downloading {name} model: {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,
    )
PY_DOWNLOAD

for model_path in "${STUDENT_DIR}" "${TEACHER_DIR}"; do
    if [ ! -f "${model_path}/config.json" ]; then
        echo "Model directory is invalid or missing config.json: ${model_path}" >&2
        exit 1
    fi
done

python3 - <<PY_CHECK
import json
from pathlib import Path

for name, path in [("student", Path("${STUDENT_DIR}")), ("teacher", Path("${TEACHER_DIR}"))]:
    cfg = json.loads((path / "config.json").read_text())
    print(f"{name}_path={path}")
    print(f"{name}_model_type={cfg.get('model_type')}")
    print(f"{name}_architectures={cfg.get('architectures')}")
PY_CHECK
