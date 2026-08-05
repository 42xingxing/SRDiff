#!/usr/bin/env bash

# Shared, machine-independent defaults for all launch scripts.
SRDIFF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${SRDIFF_ROOT}}"
PYTHON="${PYTHON:-python}"

export SRDIFF_ROOT WORK_DIR PYTHON

validate_gpu_selection() {
    local gpu_list=$1
    local expected_count=$2
    local selected_gpus
    IFS=',' read -r -a selected_gpus <<< "${gpu_list}"
    if [[ ${#selected_gpus[@]} -ne ${expected_count} ]]; then
        echo "GPU_LIST contains ${#selected_gpus[@]} entries, but NUM_GPUS is ${expected_count}"
        exit 2
    fi
}
