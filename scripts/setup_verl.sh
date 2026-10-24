#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
VERL_DIR=${VERL_DIR:-${ROOT}/third_party/verl}
PIN=$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "${ROOT}/upstream.lock.json")

if [ ! -e "${VERL_DIR}" ]; then
    mkdir -p -- "$(dirname -- "${VERL_DIR}")"
    git clone --filter=blob:none --no-checkout https://github.com/verl-project/verl.git "${VERL_DIR}"
    git -C "${VERL_DIR}" -c core.autocrlf=false checkout --detach "${PIN}"
else
    if [ "$(git -C "${VERL_DIR}" rev-parse HEAD)" != "${PIN}" ]; then
        printf '%s\n' "Existing VERL_DIR is not the pinned commit; nothing was reset or overwritten." >&2
        exit 1
    fi
fi

"${PYTHON_BIN}" "${ROOT}/scripts/apply_patch.py" --verl-dir "${VERL_DIR}"
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    "${PYTHON_BIN}" -m pip install --no-deps -e "${VERL_DIR}"
fi
printf 'Pinned verl ready: %s\n' "${VERL_DIR}"
