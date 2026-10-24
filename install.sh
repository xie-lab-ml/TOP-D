#!/usr/bin/env bash
# Install the bundled source. This does not clone another verl or patch it.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODE=${1:---install}
case "${MODE}" in --install|--existing-env|--check) ;; *) echo 'Usage: bash install.sh [--install|--existing-env|--check]' >&2; exit 2 ;; esac
if [ "$(uname -s)" != "Linux" ]; then
    echo 'The original FSDP/vLLM training environment requires Linux.' >&2
    exit 1
fi
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 to match the original environment"'
"${PYTHON_BIN}" "${ROOT}/recipe/verification/verify_release.py"
if [ "${MODE}" = "--check" ]; then
    "${PYTHON_BIN}" "${ROOT}/recipe/environment/check_environment.py"
    exit 0
fi
if [ "${MODE}" = "--install" ]; then
    "${PYTHON_BIN}" -c 'import sys; assert sys.prefix != sys.base_prefix or "conda" in sys.prefix.lower(), "Create and activate a dedicated venv/conda environment first"'
    "${PYTHON_BIN}" -m pip install 'pip>=24' wheel ninja packaging==25.0 setuptools==80.10.2
    "${PYTHON_BIN}" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129
    "${PYTHON_BIN}" -m pip install -r "${ROOT}/recipe/environment/requirements.txt"
    # A matching CUDA toolkit/compiler is needed when a binary wheel is unavailable.
    MAX_JOBS=${MAX_JOBS:-8} "${PYTHON_BIN}" -m pip install flash-attn==2.8.3 --no-build-isolation
fi
# Do not use [vllm]: the inherited upstream extra has an older version cap.
"${PYTHON_BIN}" -m pip install --no-deps --no-build-isolation -e "${ROOT}/verl"
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" "${ROOT}/recipe/environment/check_environment.py"
"${PYTHON_BIN}" -c 'import pathlib,sys,verl; root=pathlib.Path(sys.argv[1]).resolve(); assert pathlib.Path(verl.__file__).resolve().is_relative_to(root / "verl"); print("Imported bundled verl:", verl.__file__)' "${ROOT}"
echo 'Environment installed. Model downloads and training have not been started.'
